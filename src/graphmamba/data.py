from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from graphmamba.graph import iter_tempo_files, load_graph_npz, parse_tempo_time


@dataclass(frozen=True)
class AlignedScan:
    tempo_path: Path
    scan_time: pd.Timestamp
    target_time: pd.Timestamp
    delta_t_hours: float


class RunningStandardizer:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values: np.ndarray) -> None:
        flat = values[np.isfinite(values)].astype(np.float64)
        for value in flat:
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (value - self.mean)

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0
        return float(math.sqrt(max(self.m2 / (self.count - 1), 1e-12)))

    def state_dict(self) -> dict[str, float | int]:
        return {"count": int(self.count), "mean": float(self.mean), "std": float(self.std)}


def _airnow_file_for_time(airnow_dir: str | Path, timestamp: pd.Timestamp) -> Path:
    date = timestamp.strftime("%Y%m%d")
    return Path(airnow_dir) / f"airnow_no2_{date}.nc"


def _split_variable_path(path: str | None) -> tuple[str | None, str]:
    """Split a ``group/name`` reference; ``group`` may be ``None`` for root vars."""
    if not path:
        return None, ""
    parts = str(path).strip("/").split("/")
    if len(parts) == 1:
        return None, parts[0]
    return "/".join(parts[:-1]), parts[-1]


def _read_tempo_variable(
    path: Path,
    group: str | None,
    name: str,
    lat_idx: np.ndarray,
    lon_idx: np.ndarray,
) -> np.ndarray | None:
    """Read a TEMPO L3 variable at station-neighbour grid indices; returns ``None`` if absent."""
    open_kwargs = {"decode_times": False, "mask_and_scale": True}
    if group:
        open_kwargs["group"] = group
    with xr.open_dataset(path, **open_kwargs) as ds:
        if name not in ds.variables:
            return None
        data = ds[name]
        if "time" in data.dims:
            data = data.isel(time=0)
        grid = np.asarray(data.values)
    return np.asarray(grid[lat_idx, lon_idx], dtype=np.float32)


def _read_tempo_variables_multi(
    path: Path,
    requests: tuple[tuple[str | None, str], ...],
    lat_idx: np.ndarray,
    lon_idx: np.ndarray,
) -> tuple[np.ndarray | None, ...]:
    """Read multiple TEMPO variables grouped by NetCDF group, one open per group."""
    results: dict[int, np.ndarray | None] = {}
    by_group: dict[str | None, list[tuple[int, str]]] = {}
    for i, (group, name) in enumerate(requests):
        by_group.setdefault(group, []).append((i, name))

    for group, items in by_group.items():
        open_kwargs: dict[str, Any] = {"decode_times": False, "mask_and_scale": True}
        if group:
            open_kwargs["group"] = group
        with xr.open_dataset(path, **open_kwargs) as ds:
            for i, name in items:
                if name not in ds.variables:
                    results[i] = None
                    continue
                data = ds[name]
                if "time" in data.dims:
                    data = data.isel(time=0)
                results[i] = np.asarray(np.asarray(data.values)[lat_idx, lon_idx], dtype=np.float32)

    return tuple(results[i] for i in range(len(requests)))


def discover_aligned_scans(
    tempo_dir: str | Path,
    airnow_dir: str | Path,
    start_date: str | None,
    end_date: str | None,
    max_time_snap_hours: float,
    lead_time: float = 0.0,
) -> list[AlignedScan]:
    """Snap each TEMPO scan to an AirNow target timestamp after lead-time shifting."""
    candidates: dict[pd.Timestamp, tuple[Path, pd.Timestamp, float]] = {}
    lead_delta = pd.Timedelta(hours=float(lead_time))
    for path in iter_tempo_files(tempo_dir, start_date, end_date):
        scan_time = parse_tempo_time(path)
        predictor_time = scan_time.round("h")
        snap_hours = abs((scan_time - predictor_time).total_seconds()) / 3600.0
        if snap_hours > max_time_snap_hours:
            continue
        target_time = predictor_time + lead_delta
        airnow_path = _airnow_file_for_time(airnow_dir, target_time)
        if not airnow_path.exists():
            continue
        current = candidates.get(target_time)
        if current is None or snap_hours < current[2]:
            candidates[target_time] = (path, scan_time, snap_hours)

    rows = sorted((target_time, *values) for target_time, values in candidates.items())
    aligned: list[AlignedScan] = []
    previous_scan_time: pd.Timestamp | None = None
    for target_time, path, scan_time, _snap_hours in rows:
        if previous_scan_time is None:
            delta = 1.0
        else:
            delta = max((scan_time - previous_scan_time).total_seconds() / 3600.0, 1e-3)
        aligned.append(AlignedScan(path, scan_time, target_time, float(delta)))
        previous_scan_time = scan_time
    return aligned


def split_indices(count: int, train_fraction: float, val_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(count * train_fraction)
    val_end = train_end + int(count * val_fraction)
    all_idx = np.arange(count, dtype=np.int64)
    return all_idx[:train_end], all_idx[train_end:val_end], all_idx[val_end:]


class TempoAirNowDataset(Dataset):
    """Windowed TEMPO/AirNow graph samples.

    Nodes are ordered as AirNow stations first, then selected TEMPO cells. The
    model predicts only the AirNow rows at the final time step in each window.
    """

    def __init__(
        self,
        graph_path: str | Path,
        tempo_dir: str | Path,
        airnow_dir: str | Path,
        start_date: str | None,
        end_date: str | None,
        context_steps: int,
        max_time_snap_hours: float,
        max_column: float,
        min_valid_targets: int = 1,
        scan_indices: np.ndarray | None = None,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        lead_time: float = 0.0,
        tempo_variable: str = "product/vertical_column_troposphere",
        tempo_quality_variable: str = "product/main_data_quality_flag",
        tempo_cloud_variable: str = "support_data/eff_cloud_fraction",
        tempo_quality_flag_valid: int = 0,
        tempo_cloud_fraction_max: float = 0.2,
    ) -> None:
        if context_steps < 1:
            raise ValueError("context_steps must be positive.")
        self.graph = load_graph_npz(graph_path)
        self.tempo_dir = Path(tempo_dir)
        self.airnow_dir = Path(airnow_dir)
        self.context_steps = context_steps
        self.max_column = float(max_column)
        self.lead_time = float(lead_time)
        self.target_mean = float(target_mean)
        self.target_std = float(max(target_std, 1e-6))
        self.tempo_value_group, self.tempo_value_name = _split_variable_path(tempo_variable)
        self.tempo_quality_group, self.tempo_quality_name = _split_variable_path(tempo_quality_variable)
        self.tempo_cloud_group, self.tempo_cloud_name = _split_variable_path(tempo_cloud_variable)
        self.tempo_quality_flag_valid = int(tempo_quality_flag_valid)
        self.tempo_cloud_fraction_max = float(tempo_cloud_fraction_max)
        self.aligned_scans = discover_aligned_scans(
            tempo_dir=tempo_dir,
            airnow_dir=airnow_dir,
            start_date=start_date,
            end_date=end_date,
            max_time_snap_hours=max_time_snap_hours,
            lead_time=self.lead_time,
        )
        if len(self.aligned_scans) < context_steps:
            raise ValueError(f"Only {len(self.aligned_scans)} aligned scans found; need at least {context_steps}.")
        self.site_ids = self.graph["airnow_site_ids"].astype(str)
        self.site_to_idx = {site: i for i, site in enumerate(self.site_ids)}
        self.num_airnow = int(self.graph["metadata"]["num_airnow"])
        self.num_tempo = int(self.graph["metadata"]["num_tempo"])
        self.num_nodes = int(self.graph["metadata"]["num_nodes"])
        # Cache static per-node arrays once instead of re-extracting every __getitem__ call.
        self._node_lat = self.graph["node_lat"].astype(np.float32)
        self._node_lon = self.graph["node_lon"].astype(np.float32)
        self._node_type = self.graph["node_type"].astype(np.float32)
        self._tempo_lat_idx = self.graph["tempo_lat_idx"].astype(np.int64)
        self._tempo_lon_idx = self.graph["tempo_lon_idx"].astype(np.int64)
        # Build the variable request tuple used by the multi-open reader.
        _requests: list[tuple[str | None, str]] = [(self.tempo_value_group, self.tempo_value_name)]
        if self.tempo_quality_name:
            _requests.append((self.tempo_quality_group, self.tempo_quality_name))
        if self.tempo_cloud_name:
            _requests.append((self.tempo_cloud_group, self.tempo_cloud_name))
        self._tempo_requests = tuple(_requests)
        self._tempo_cache_maxsize = max(context_steps * 4, 32)
        # Per-process dict cache; populated lazily in _load_tempo_values.
        # Dropped during pickling so each DataLoader worker starts with an empty cache.
        self._tempo_file_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        end_indices = np.arange(context_steps - 1, len(self.aligned_scans), dtype=np.int64)
        self.end_indices = end_indices if scan_indices is None else np.asarray(scan_indices, dtype=np.int64)
        if min_valid_targets > 0:
            self.end_indices = np.array(
                [
                    idx
                    for idx in self.end_indices
                    if self._count_valid_targets(self.aligned_scans[int(idx)].target_time) >= min_valid_targets
                ],
                dtype=np.int64,
            )
        if self.end_indices.size == 0:
            raise ValueError("No windows remain after filtering by min_valid_targets.")

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_tempo_file_cache"] = {}
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if "_tempo_file_cache" not in self.__dict__:
            self._tempo_file_cache = {}

    @property
    def feature_dim(self) -> int:
        return 6

    def with_target_scaler(self, mean: float, std: float) -> "TempoAirNowDataset":
        self.target_mean = float(mean)
        self.target_std = float(max(std, 1e-6))
        return self

    def subset(self, end_indices: np.ndarray) -> "TempoAirNowDataset":
        other = object.__new__(TempoAirNowDataset)
        other.__dict__.update(self.__dict__)
        other.end_indices = np.asarray(end_indices, dtype=np.int64)
        return other

    def fit_target_scaler(self, max_samples: int | None = None) -> RunningStandardizer:
        stats = RunningStandardizer()
        indices = self.end_indices if max_samples is None else self.end_indices[:max_samples]
        for end_idx in indices:
            target, mask = self._load_airnow_target(self.aligned_scans[int(end_idx)].target_time)
            stats.update(target[mask])
        self.target_mean = float(stats.mean)
        self.target_std = float(stats.std)
        return stats

    def __len__(self) -> int:
        return int(self.end_indices.size)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | float]:
        end_idx = int(self.end_indices[index])
        start_idx = end_idx - self.context_steps + 1
        scans = self.aligned_scans[start_idx : end_idx + 1]
        x_steps = []
        dt_steps = []
        for scan in scans:
            x_steps.append(self._load_node_features(scan))
            dt_steps.append(scan.delta_t_hours)
        y, y_mask = self._load_airnow_target(scans[-1].target_time)
        y_scaled = (y - self.target_mean) / self.target_std
        return {
            "x": torch.from_numpy(np.stack(x_steps, axis=0)).float(),
            "dt": torch.tensor(dt_steps, dtype=torch.float32),
            "y": torch.from_numpy(y_scaled.astype(np.float32)),
            "y_raw": torch.from_numpy(y.astype(np.float32)),
            "y_mask": torch.from_numpy(y_mask.astype(bool)),
            "target_time": str(scans[-1].target_time),
            "scan_time": str(scans[-1].scan_time),
            "lead_time": float(self.lead_time),
        }

    def _load_node_features(self, scan: AlignedScan) -> np.ndarray:
        features = np.zeros((self.num_nodes, self.feature_dim), dtype=np.float32)
        features[:, 2] = self._node_type
        features[:, 3] = (self._node_lat - 40.0) / 15.0
        features[:, 4] = (self._node_lon + 100.0) / 30.0
        features[:, 5] = min(max(scan.delta_t_hours, 0.0), 48.0) / 24.0

        tempo_values, valid = self._load_tempo_values(scan.tempo_path)
        tempo_offset = self.num_airnow
        clipped = np.clip(tempo_values, 0.0, self.max_column)
        features[tempo_offset:, 0] = clipped / self.max_column
        features[tempo_offset:, 1] = valid.astype(np.float32)
        return features

    def _load_tempo_values(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        key = str(path)
        if key not in self._tempo_file_cache:
            arrays = _read_tempo_variables_multi(path, self._tempo_requests, self._tempo_lat_idx, self._tempo_lon_idx)
            values = arrays[0]
            valid = np.isfinite(values)
            idx = 1
            if len(self._tempo_requests) > idx and self._tempo_requests[idx][1]:
                quality = arrays[idx]
                idx += 1
                if quality is not None:
                    valid = valid & (quality == self.tempo_quality_flag_valid)
            if len(self._tempo_requests) > idx and self._tempo_requests[idx][1]:
                cloud = arrays[idx]
                if cloud is not None:
                    valid = valid & np.isfinite(cloud) & (cloud <= self.tempo_cloud_fraction_max)
            values = np.where(valid, values, 0.0).astype(np.float32)
            if len(self._tempo_file_cache) >= self._tempo_cache_maxsize:
                self._tempo_file_cache.pop(next(iter(self._tempo_file_cache)))
            self._tempo_file_cache[key] = (values, valid)
        return self._tempo_file_cache[key]

    def _load_airnow_target(self, timestamp: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        path = _airnow_file_for_time(self.airnow_dir, timestamp)
        target = np.full(self.num_airnow, np.nan, dtype=np.float32)
        with xr.open_dataset(path, decode_times=True, mask_and_scale=True) as ds:
            times = pd.to_datetime(ds["time"].values, utc=True)
            matches = np.flatnonzero(times == timestamp)
            if matches.size == 0:
                return target, np.zeros(self.num_airnow, dtype=bool)
            time_idx = int(matches[0])
            site_ids = np.asarray(ds["site"].values).astype(str)
            values = np.asarray(ds["no2"].isel(time=time_idx).values, dtype=np.float32)
            for src_idx, site_id in enumerate(site_ids):
                dst_idx = self.site_to_idx.get(site_id)
                if dst_idx is not None:
                    target[dst_idx] = values[src_idx]
        mask = np.isfinite(target)
        target = np.where(mask, target, self.target_mean).astype(np.float32)
        return target, mask

    def _count_valid_targets(self, timestamp: pd.Timestamp) -> int:
        path = _airnow_file_for_time(self.airnow_dir, timestamp)
        if not path.exists():
            return 0
        with xr.open_dataset(path, decode_times=True, mask_and_scale=True) as ds:
            times = pd.to_datetime(ds["time"].values, utc=True)
            matches = np.flatnonzero(times == timestamp)
            if matches.size == 0:
                return 0
            values = np.asarray(ds["no2"].isel(time=int(matches[0])).values, dtype=np.float32)
            return int(np.count_nonzero(np.isfinite(values)))


def collate_samples(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = ["x", "dt", "y", "y_raw", "y_mask"]
    out: dict[str, Any] = {key: torch.stack([item[key] for item in batch], dim=0) for key in tensor_keys}
    out["target_time"] = [item["target_time"] for item in batch]
    out["scan_time"] = [item["scan_time"] for item in batch]
    out["lead_time"] = [item["lead_time"] for item in batch]
    return out
