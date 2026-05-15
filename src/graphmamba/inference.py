from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader

from graphmamba.config import ensure_dir
from graphmamba.config import resolve_run_paths
from graphmamba.data import TempoAirNowDataset, collate_samples
from graphmamba.graph import load_graph_npz
from graphmamba.model import GraphBoundPredictor, UnifiedGraphMambaPredictor
from graphmamba.train import graph_tensors, make_progress_bar, runtime_device_and_gpu_ids


@dataclass
class InferenceBundle:
    model: torch.nn.Module
    base_model: UnifiedGraphMambaPredictor
    graph: dict[str, Any]
    target_mean: float
    target_std: float
    device: torch.device
    adj: torch.Tensor
    order: torch.Tensor
    inverse_order: torch.Tensor
    config: dict[str, Any]
    device_ids: list[int]


def load_inference_bundle(
    checkpoint_path: str | Path,
    graph_path: str | Path | None = None,
    device: str = "auto",
    multi_gpu: object = "auto",
    gpu_min_free_memory_gb: float = 0.0,
    gpu_max_count: int | None = None,
) -> InferenceBundle:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    graph_file = Path(graph_path or resolve_run_paths(config)["graph_path"])
    graph = load_graph_npz(graph_file)
    model_cfg = config["model"]
    data_cfg = config["data"]
    torch_device, device_ids = runtime_device_and_gpu_ids(
        device,
        multi_gpu=multi_gpu,
        min_free_memory_gb=gpu_min_free_memory_gb,
        max_gpus=gpu_max_count,
    )

    base_model = UnifiedGraphMambaPredictor(
        input_dim=int(model_cfg.get("input_dim", data_cfg.get("input_dim", 6))),
        hidden_dim=int(model_cfg["hidden_dim"]),
        state_dim=int(model_cfg["state_dim"]),
        num_airnow=int(graph["metadata"]["num_airnow"]),
        spatial_layers=int(model_cfg.get("spatial_layers", 2)),
        temporal_layers=int(model_cfg.get("temporal_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(torch_device)
    base_model.load_state_dict(checkpoint.get("base_model", checkpoint["model"]))
    adj, order, inverse_order = graph_tensors(graph, torch_device)
    model: torch.nn.Module = GraphBoundPredictor(base_model, adj, order, inverse_order).to(torch_device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
    model.eval()

    target_scaler = checkpoint.get("target_scaler", {})
    target_mean = float(target_scaler.get("mean", 0.0))
    target_std = float(target_scaler.get("std", 1.0))
    return InferenceBundle(
        model=model,
        base_model=base_model,
        graph=graph,
        target_mean=target_mean,
        target_std=max(target_std, 1e-6),
        device=torch_device,
        adj=adj,
        order=order,
        inverse_order=inverse_order,
        config=config,
        device_ids=device_ids,
    )


@torch.no_grad()
def predict_dataframe(
    dataset: TempoAirNowDataset,
    bundle: InferenceBundle,
    batch_size: int = 1,
    num_workers: int = 0,
    show_progress: bool = True,
) -> pd.DataFrame:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_samples,
    )
    site_ids = dataset.graph["airnow_site_ids"].astype(str)
    site_names = dataset.graph["airnow_site_names"].astype(str)
    agencies = dataset.graph["airnow_agencies"].astype(str)
    station_lat = dataset.graph["airnow_lat"].astype(float)
    station_lon = dataset.graph["airnow_lon"].astype(float)
    nearest_dist = dataset.graph["nearest_tempo_distance_km"].astype(float)

    rows: list[dict[str, object]] = []
    iterator = make_progress_bar(
        loader,
        desc="Running inference",
        show_progress=show_progress,
        position=0,
        leave=True,
        total=len(loader),
        unit="batch",
        colour="blue",
    )
    for batch in iterator:
        x = batch["x"].to(bundle.device)
        dt = batch["dt"].to(bundle.device)
        pred_scaled = bundle.model(x, dt).cpu().numpy()
        pred = pred_scaled * bundle.target_std + bundle.target_mean
        observed = batch["y_raw"].numpy()
        mask = batch["y_mask"].numpy().astype(bool)
        final_dt = batch["dt"][:, -1].numpy()

        for batch_idx, target_time in enumerate(batch["target_time"]):
            for station_idx, site_id in enumerate(site_ids):
                has_target = bool(mask[batch_idx, station_idx])
                observed_value = float(observed[batch_idx, station_idx]) if has_target else np.nan
                predicted_value = float(pred[batch_idx, station_idx])
                rows.append(
                    {
                        "target_time": target_time,
                        "predictor_scan_time": batch["scan_time"][batch_idx],
                        "lead_time": float(batch["lead_time"][batch_idx]),
                        "site_id": site_id,
                        "site_name": site_names[station_idx],
                        "agency": agencies[station_idx],
                        "latitude": float(station_lat[station_idx]),
                        "longitude": float(station_lon[station_idx]),
                        "predicted_no2_ppb": predicted_value,
                        "observed_no2_ppb": observed_value,
                        "has_observation": has_target,
                        "absolute_error_ppb": abs(predicted_value - observed_value) if has_target else np.nan,
                        "delta_t_hours": float(final_dt[batch_idx]),
                        "nearest_tempo_distance_km": float(nearest_dist[station_idx]),
                    }
                )
    return pd.DataFrame(rows)


def predictions_to_airnow_dataset(
    predictions: pd.DataFrame,
    run_name: str | None = None,
    attrs: dict[str, object] | None = None,
) -> xr.Dataset:
    """Convert station-time predictions into an AirNow-like NetCDF dataset.

    The core shape mirrors the AirNow files: ``no2(time, site)`` plus station
    metadata. Here ``no2`` contains the model prediction, and observed AirNow
    values are kept in ``observed_no2`` when available.
    """
    if predictions.empty:
        raise ValueError("Cannot build a NetCDF dataset from an empty predictions table.")

    frame = predictions.copy()
    frame["time_key"] = pd.to_datetime(frame["target_time"], utc=True).dt.tz_convert(None)
    times = pd.DatetimeIndex(sorted(frame["time_key"].unique()))
    site_ids = list(dict.fromkeys(frame["site_id"].astype(str).tolist()))
    time_index = {pd.Timestamp(value): idx for idx, value in enumerate(times)}
    site_index = {value: idx for idx, value in enumerate(site_ids)}
    shape = (len(times), len(site_ids))

    predicted = np.full(shape, np.nan, dtype=np.float32)
    observed = np.full(shape, np.nan, dtype=np.float32)
    absolute_error = np.full(shape, np.nan, dtype=np.float32)
    has_observation = np.zeros(shape, dtype=np.int8)
    delta_t_hours = np.full(len(times), np.nan, dtype=np.float32)

    site_meta = frame.drop_duplicates("site_id").set_index("site_id")
    latitude = np.array([site_meta.loc[site_id, "latitude"] for site_id in site_ids], dtype=np.float32)
    longitude = np.array([site_meta.loc[site_id, "longitude"] for site_id in site_ids], dtype=np.float32)
    site_name = np.array([str(site_meta.loc[site_id, "site_name"]) for site_id in site_ids], dtype=object)
    agency = np.array([str(site_meta.loc[site_id, "agency"]) for site_id in site_ids], dtype=object)
    nearest_distance = np.array(
        [site_meta.loc[site_id, "nearest_tempo_distance_km"] for site_id in site_ids],
        dtype=np.float32,
    )

    for row in frame.itertuples(index=False):
        time_pos = time_index[pd.Timestamp(row.time_key)]
        site_pos = site_index[str(row.site_id)]
        predicted[time_pos, site_pos] = np.float32(row.predicted_no2_ppb)
        observed[time_pos, site_pos] = np.float32(row.observed_no2_ppb)
        absolute_error[time_pos, site_pos] = np.float32(row.absolute_error_ppb)
        has_observation[time_pos, site_pos] = 1 if bool(row.has_observation) else 0
        delta_t_hours[time_pos] = np.float32(row.delta_t_hours)

    ds = xr.Dataset(
        data_vars={
            "no2": (
                ("time", "site"),
                predicted,
                {
                    "long_name": "Predicted NO2 concentration",
                    "units": "PPB",
                    "source_field": "GraphMamba prediction",
                },
            ),
            "observed_no2": (
                ("time", "site"),
                observed,
                {
                    "long_name": "Observed AirNow NO2 concentration",
                    "units": "PPB",
                    "source_field": "AirNow RawConcentration",
                },
            ),
            "absolute_error": (
                ("time", "site"),
                absolute_error,
                {
                    "long_name": "Absolute prediction error",
                    "units": "PPB",
                },
            ),
            "has_observation": (
                ("time", "site"),
                has_observation,
                {
                    "long_name": "Observed AirNow target availability",
                    "flag_values": np.array([0, 1], dtype=np.int8),
                    "flag_meanings": "missing available",
                },
            ),
            "latitude": (
                ("site",),
                latitude,
                {
                    "units": "degrees_north",
                    "long_name": "site latitude",
                },
            ),
            "longitude": (
                ("site",),
                longitude,
                {
                    "units": "degrees_east",
                    "long_name": "site longitude",
                },
            ),
            "site_name": (("site",), site_name),
            "agency": (("site",), agency),
            "delta_t_hours": (
                ("time",),
                delta_t_hours,
                {
                    "long_name": "Actual gap between consecutive TEMPO observations",
                    "units": "hours",
                },
            ),
            "nearest_tempo_distance_km": (
                ("site",),
                nearest_distance,
                {
                    "long_name": "Distance to nearest selected TEMPO grid point",
                    "units": "km",
                },
            ),
        },
        coords={
            "time": ("time", times.to_numpy(dtype="datetime64[ns]")),
            "site": ("site", np.asarray(site_ids, dtype=object)),
        },
        attrs={
            "title": "Compiled GraphMamba AirNow NO2 predictions",
            "source": "GraphMamba inference over TEMPO NO2 L3 and AirNow NO2",
            "run_name": run_name or "",
            "created": pd.Timestamp.utcnow().isoformat(),
            "time_coverage_start": pd.Timestamp(times[0]).isoformat(),
            "time_coverage_end": pd.Timestamp(times[-1]).isoformat(),
            "temporal_dimension": "compiled over inference range",
        },
    )
    if attrs:
        ds.attrs.update({key: str(value) for key, value in attrs.items()})
    return ds


def write_predictions_netcdf(
    predictions: pd.DataFrame,
    output_path: str | Path,
    run_name: str | None = None,
    attrs: dict[str, object] | None = None,
) -> Path:
    output = Path(output_path)
    ensure_dir(output.parent)
    ds = predictions_to_airnow_dataset(predictions, run_name=run_name, attrs=attrs)
    encoding = {
        "time": {"units": "hours since 1970-01-01 00:00:00", "calendar": "proleptic_gregorian"},
        "no2": {"dtype": "float32", "_FillValue": np.float32(np.nan)},
        "observed_no2": {"dtype": "float32", "_FillValue": np.float32(np.nan)},
        "absolute_error": {"dtype": "float32", "_FillValue": np.float32(np.nan)},
        "latitude": {"dtype": "float32", "_FillValue": np.float32(np.nan)},
        "longitude": {"dtype": "float32", "_FillValue": np.float32(np.nan)},
        "delta_t_hours": {"dtype": "float32", "_FillValue": np.float32(np.nan)},
        "nearest_tempo_distance_km": {"dtype": "float32", "_FillValue": np.float32(np.nan)},
        "has_observation": {"dtype": "int8", "_FillValue": np.int8(-1)},
    }
    ds.to_netcdf(output, encoding=encoding)
    return output
