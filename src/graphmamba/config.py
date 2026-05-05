from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and expand user home markers in path-like fields."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config {config_path} did not contain a mapping.")
    return config


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    resolved = Path(path)
    if resolved.is_absolute() or base_dir is None:
        return resolved
    return Path(base_dir) / resolved


def resolve_run_paths(config: dict[str, Any], base_dir: str | Path | None = None) -> dict[str, Path | str]:
    """Resolve run-scoped paths from ``run_name``.

    Outputs are written under ``output/<run_name>/`` and model checkpoints under
    ``checkpoints/<run_name>/`` unless the corresponding roots are changed in
    the YAML config.
    """
    run_name = str(config.get("run_name", "default"))
    paths = config.get("paths", {})
    checkpoint_root = resolve_path(paths.get("checkpoint_root", "checkpoints"), base_dir)
    output_root = resolve_path(paths.get("output_root", "output"), base_dir)
    checkpoint_dir = checkpoint_root / run_name
    output_dir = output_root / run_name
    checkpoint_filename = str(paths.get("checkpoint_filename", "best_model.pt"))
    graph_filename = str(paths.get("graph_filename", "unified_graph.npz"))
    metrics_filename = str(paths.get("metrics_filename", "metrics.json"))
    return {
        "run_name": run_name,
        "checkpoint_dir": checkpoint_dir,
        "output_dir": output_dir,
        "checkpoint_path": checkpoint_dir / checkpoint_filename,
        "graph_path": output_dir / graph_filename,
        "metrics_path": output_dir / metrics_filename,
    }


def resolve_inference_paths(config: dict[str, Any], base_dir: str | Path | None = None) -> dict[str, Path]:
    run_paths = resolve_run_paths(config, base_dir)
    inference = config.get("inference", {})
    output_csv_filename = str(inference.get("output_csv_filename", inference.get("output_filename", "predictions.csv")))
    output_netcdf_filename = str(inference.get("output_netcdf_filename", "predictions.nc"))
    return {
        "checkpoint_path": Path(run_paths["checkpoint_path"]),
        "graph_path": Path(run_paths["graph_path"]),
        "output_csv": Path(run_paths["output_dir"]) / output_csv_filename,
        "output_netcdf": Path(run_paths["output_dir"]) / output_netcdf_filename,
    }
