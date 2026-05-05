from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from graphmamba.config import ensure_dir, load_config, resolve_run_paths
from graphmamba.hilbert import hilbert_order

EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class AirNowSites:
    site_ids: np.ndarray
    names: np.ndarray
    agencies: np.ndarray
    lat: np.ndarray
    lon: np.ndarray


def parse_date(value: str | None) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    return pd.Timestamp(value, tz="UTC")


def iter_airnow_files(airnow_dir: str | Path, start: str | None = None, end: str | None = None) -> list[Path]:
    base = Path(airnow_dir)
    files = sorted(base.glob("airnow_no2_*.nc"))
    start_ts = parse_date(start)
    end_ts = parse_date(end)
    selected: list[Path] = []
    for path in files:
        date_token = path.stem.rsplit("_", 1)[-1]
        ts = pd.Timestamp(date_token, tz="UTC")
        if start_ts is not None and ts < start_ts.normalize():
            continue
        if end_ts is not None and ts > end_ts.normalize():
            continue
        selected.append(path)
    return selected


def iter_tempo_files(tempo_dir: str | Path, start: str | None = None, end: str | None = None) -> list[Path]:
    base = Path(tempo_dir)
    files = sorted(base.glob("TEMPO_NO2_L3_V03_*_subsetted.nc4"))
    start_ts = parse_date(start)
    end_ts = parse_date(end)
    selected: list[Path] = []
    for path in files:
        ts = parse_tempo_time(path)
        if start_ts is not None and ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts + pd.Timedelta(days=1):
            continue
        selected.append(path)
    return selected


def parse_tempo_time(path: str | Path) -> pd.Timestamp:
    name = Path(path).name
    for token in name.split("_"):
        if len(token) == 16 and token.endswith("Z") and "T" in token:
            return pd.Timestamp(token, tz="UTC")
    raise ValueError(f"Could not parse TEMPO timestamp from {path}")


def _strings(values: xr.DataArray | np.ndarray, count: int, default: str = "") -> np.ndarray:
    arr = np.asarray(values)
    if arr.shape == ():
        return np.full(count, default, dtype=object)
    return arr.astype(str)


def load_airnow_sites(airnow_dir: str | Path, start: str | None = None, end: str | None = None) -> AirNowSites:
    """Load and de-duplicate AirNow station metadata over the requested period."""
    records: dict[str, dict[str, object]] = {}
    files = iter_airnow_files(airnow_dir, start, end)
    if not files:
        raise FileNotFoundError(f"No AirNow files found in {airnow_dir}")

    for path in files:
        with xr.open_dataset(path, decode_times=True) as ds:
            site_count = int(ds.sizes["site"])
            site_ids = _strings(ds["site"], site_count)
            names = _strings(ds.get("site_name", np.full(site_count, "")), site_count)
            agencies = _strings(ds.get("agency", np.full(site_count, "")), site_count)
            lat = np.asarray(ds["latitude"].values, dtype=np.float64)
            lon = np.asarray(ds["longitude"].values, dtype=np.float64)
            for idx, site_id in enumerate(site_ids):
                if not site_id or site_id.lower() == "nan":
                    continue
                if not np.isfinite(lat[idx]) or not np.isfinite(lon[idx]):
                    continue
                row = records.setdefault(
                    site_id,
                    {
                        "name": names[idx],
                        "agency": agencies[idx],
                        "lat": [],
                        "lon": [],
                    },
                )
                row["lat"].append(float(lat[idx]))
                row["lon"].append(float(lon[idx]))

    site_ids = np.array(sorted(records), dtype=object)
    names = np.array([records[s]["name"] for s in site_ids], dtype=object)
    agencies = np.array([records[s]["agency"] for s in site_ids], dtype=object)
    lat = np.array([np.nanmedian(records[s]["lat"]) for s in site_ids], dtype=np.float64)
    lon = np.array([np.nanmedian(records[s]["lon"]) for s in site_ids], dtype=np.float64)
    return AirNowSites(site_ids=site_ids, names=names, agencies=agencies, lat=lat, lon=lon)


def load_tempo_grid(tempo_dir: str | Path) -> tuple[np.ndarray, np.ndarray, Path]:
    files = sorted(Path(tempo_dir).glob("TEMPO_NO2_L3_V03_*_subsetted.nc4"))
    if not files:
        raise FileNotFoundError(f"No TEMPO L3 files found in {tempo_dir}")
    with xr.open_dataset(files[0], decode_times=False) as ds:
        lat = np.asarray(ds["latitude"].values, dtype=np.float64)
        lon = np.asarray(ds["longitude"].values, dtype=np.float64)
    return lat, lon, files[0]


def lonlat_to_unit_xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    cos_lat = np.cos(lat_rad)
    return np.column_stack((cos_lat * np.cos(lon_rad), cos_lat * np.sin(lon_rad), np.sin(lat_rad)))


def chord_to_km(chord: np.ndarray) -> np.ndarray:
    clipped = np.clip(chord / 2.0, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(clipped)


def grid_points_from_1d(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_idx, lon_idx = np.meshgrid(np.arange(lat.size), np.arange(lon.size), indexing="ij")
    grid_lat = lat[lat_idx.ravel()]
    grid_lon = lon[lon_idx.ravel()]
    flat_idx = np.ravel_multi_index((lat_idx.ravel(), lon_idx.ravel()), (lat.size, lon.size))
    return grid_lat, grid_lon, flat_idx


def _add_symmetric_edges(edges: set[tuple[int, int]], src: int, dst: int) -> None:
    if src == dst:
        return
    edges.add((src, dst))
    edges.add((dst, src))


def normalized_adjacency(num_nodes: int, edge_index: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return COO indices/values for D^-1/2 (A + I) D^-1/2."""
    edges = {(int(i), int(i)) for i in range(num_nodes)}
    for src, dst in edge_index.T:
        edges.add((int(src), int(dst)))
    rows = np.fromiter((e[0] for e in edges), dtype=np.int64)
    cols = np.fromiter((e[1] for e in edges), dtype=np.int64)
    degree = np.bincount(rows, minlength=num_nodes).astype(np.float64)
    values = 1.0 / np.sqrt(np.maximum(degree[rows], 1.0) * np.maximum(degree[cols], 1.0))
    order = np.lexsort((cols, rows))
    return rows[order], cols[order], values[order].astype(np.float32)


def build_unified_graph(
    tempo_dir: str | Path,
    airnow_dir: str | Path,
    graph_path: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
    local_context_k: int = 5,
    station_neighbor_k: int = 5,
    tempo_neighbor_k: int = 4,
) -> dict[str, object]:
    """Build and persist a station-centered unified graph.

    AirNow nodes are stored first, followed by the unique TEMPO grid cells that
    are among each station's nearest ``local_context_k`` satellite neighbors.
    """
    if local_context_k < 1:
        raise ValueError("local_context_k must be positive.")

    sites = load_airnow_sites(airnow_dir, start_date, end_date)
    tempo_lat_1d, tempo_lon_1d, grid_source = load_tempo_grid(tempo_dir)
    grid_lat, grid_lon, flat_idx = grid_points_from_1d(tempo_lat_1d, tempo_lon_1d)
    grid_tree = cKDTree(lonlat_to_unit_xyz(grid_lon, grid_lat))
    site_xyz = lonlat_to_unit_xyz(sites.lon, sites.lat)
    k_query = max(local_context_k, 1)
    dists_chord, grid_neighbor_pos = grid_tree.query(site_xyz, k=k_query)
    if k_query == 1:
        dists_chord = dists_chord[:, None]
        grid_neighbor_pos = grid_neighbor_pos[:, None]
    dists_km = chord_to_km(dists_chord)

    selected_flat = np.unique(flat_idx[grid_neighbor_pos.ravel()])
    flat_to_tempo_node = {int(flat): i for i, flat in enumerate(selected_flat)}
    num_airnow = sites.site_ids.size
    num_tempo = selected_flat.size
    num_nodes = num_airnow + num_tempo

    selected_lat_idx, selected_lon_idx = np.unravel_index(selected_flat, (tempo_lat_1d.size, tempo_lon_1d.size))
    selected_lat = tempo_lat_1d[selected_lat_idx]
    selected_lon = tempo_lon_1d[selected_lon_idx]

    edges: set[tuple[int, int]] = set()
    airnow_to_tempo = np.empty((num_airnow, local_context_k), dtype=np.int64)
    airnow_to_tempo_distance_km = np.empty((num_airnow, local_context_k), dtype=np.float32)

    for station_idx in range(num_airnow):
        for rank in range(local_context_k):
            flat = int(flat_idx[int(grid_neighbor_pos[station_idx, rank])])
            tempo_node = flat_to_tempo_node[flat]
            unified_tempo_node = num_airnow + tempo_node
            airnow_to_tempo[station_idx, rank] = tempo_node
            airnow_to_tempo_distance_km[station_idx, rank] = float(dists_km[station_idx, rank])
            _add_symmetric_edges(edges, station_idx, unified_tempo_node)

    nearest_tempo_distance_km = airnow_to_tempo_distance_km[:, 0]
    context_radius_km = float(np.nanmax(airnow_to_tempo_distance_km[:, -1]))
    average_nearest_distance_km = float(np.nanmean(nearest_tempo_distance_km))

    if station_neighbor_k > 0 and num_airnow > 1:
        site_tree = cKDTree(site_xyz)
        k_station = min(station_neighbor_k + 1, num_airnow)
        _, neighbor_idx = site_tree.query(site_xyz, k=k_station)
        if neighbor_idx.ndim == 1:
            neighbor_idx = neighbor_idx[:, None]
        for src in range(num_airnow):
            for dst in neighbor_idx[src, 1:]:
                _add_symmetric_edges(edges, src, int(dst))

    if tempo_neighbor_k > 0 and num_tempo > 1:
        tempo_xyz = lonlat_to_unit_xyz(selected_lon, selected_lat)
        tempo_tree = cKDTree(tempo_xyz)
        k_tempo = min(tempo_neighbor_k + 1, num_tempo)
        _, neighbor_idx = tempo_tree.query(tempo_xyz, k=k_tempo)
        if neighbor_idx.ndim == 1:
            neighbor_idx = neighbor_idx[:, None]
        for src in range(num_tempo):
            for dst in neighbor_idx[src, 1:]:
                _add_symmetric_edges(edges, num_airnow + src, num_airnow + int(dst))

    edge_index = np.array(sorted(edges), dtype=np.int64).T
    adj_row, adj_col, adj_value = normalized_adjacency(num_nodes, edge_index)

    node_lat = np.concatenate([sites.lat, selected_lat]).astype(np.float32)
    node_lon = np.concatenate([sites.lon, selected_lon]).astype(np.float32)
    node_type = np.concatenate([np.ones(num_airnow), np.zeros(num_tempo)]).astype(np.float32)
    order = hilbert_order(node_lon.astype(np.float64), node_lat.astype(np.float64)).astype(np.int64)
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(order.size)

    graph_path = Path(graph_path)
    ensure_dir(graph_path.parent)
    np.savez_compressed(
        graph_path,
        edge_index=edge_index,
        adj_row=adj_row,
        adj_col=adj_col,
        adj_value=adj_value,
        node_lat=node_lat,
        node_lon=node_lon,
        node_type=node_type,
        hilbert_order=order,
        inverse_hilbert_order=inverse_order,
        airnow_site_ids=sites.site_ids.astype(str),
        airnow_site_names=sites.names.astype(str),
        airnow_agencies=sites.agencies.astype(str),
        airnow_lat=sites.lat.astype(np.float32),
        airnow_lon=sites.lon.astype(np.float32),
        tempo_lat_idx=selected_lat_idx.astype(np.int64),
        tempo_lon_idx=selected_lon_idx.astype(np.int64),
        tempo_lat=selected_lat.astype(np.float32),
        tempo_lon=selected_lon.astype(np.float32),
        airnow_to_tempo=airnow_to_tempo,
        airnow_to_tempo_distance_km=airnow_to_tempo_distance_km,
        nearest_tempo_distance_km=nearest_tempo_distance_km.astype(np.float32),
        metadata=json.dumps(
            {
                "num_nodes": int(num_nodes),
                "num_airnow": int(num_airnow),
                "num_tempo": int(num_tempo),
                "local_context_k": int(local_context_k),
                "station_neighbor_k": int(station_neighbor_k),
                "tempo_neighbor_k": int(tempo_neighbor_k),
                "average_station_to_nearest_tempo_km": average_nearest_distance_km,
                "adaptive_context_radius_km": context_radius_km,
                "tempo_grid_source": str(grid_source),
            }
        ),
    )

    return {
        "path": str(graph_path),
        "num_nodes": int(num_nodes),
        "num_airnow": int(num_airnow),
        "num_tempo": int(num_tempo),
        "average_station_to_nearest_tempo_km": average_nearest_distance_km,
        "adaptive_context_radius_km": context_radius_km,
        "edge_count": int(edge_index.shape[1]),
    }


def load_graph_npz(path: str | Path) -> dict[str, np.ndarray | dict[str, object]]:
    data = np.load(path, allow_pickle=False)
    graph: dict[str, np.ndarray | dict[str, object]] = {key: data[key] for key in data.files if key != "metadata"}
    graph["metadata"] = json.loads(str(data["metadata"]))
    return graph


def _config_value(config: dict[str, object], *keys: str, default: object = None) -> object:
    node: object = config
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the unified TEMPO/AirNow graph.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    run_paths = resolve_run_paths(config)
    result = build_unified_graph(
        tempo_dir=str(_config_value(config, "paths", "tempo_dir")),
        airnow_dir=str(_config_value(config, "paths", "airnow_dir")),
        graph_path=run_paths["graph_path"],
        start_date=_config_value(config, "data", "start_date"),
        end_date=_config_value(config, "data", "end_date"),
        local_context_k=int(_config_value(config, "data", "local_context_k", default=5)),
        station_neighbor_k=int(_config_value(config, "data", "station_neighbor_k", default=5)),
        tempo_neighbor_k=int(_config_value(config, "data", "tempo_neighbor_k", default=4)),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
