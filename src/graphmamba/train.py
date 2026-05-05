from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from graphmamba.config import ensure_dir, load_config, resolve_run_paths
from graphmamba.data import TempoAirNowDataset, collate_samples, split_indices
from graphmamba.graph import build_unified_graph, load_graph_npz
from graphmamba.metrics import masked_r2, masked_rmse, pearson_corr
from graphmamba.model import GraphBoundPredictor, UnifiedGraphMambaPredictor, make_sparse_adjacency


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _multi_gpu_mode(value: object) -> str:
    if isinstance(value, bool):
        return "auto" if value else "off"
    mode = str(value or "off").lower()
    if mode in {"true", "yes", "on", "enabled"}:
        return "on"
    if mode in {"auto", "idle"}:
        return "auto"
    return "off"


def select_cuda_device_ids(min_free_memory_gb: float = 0.0, max_gpus: int | None = None) -> list[int]:
    """Select visible CUDA devices, preferring devices with more free memory."""
    if not torch.cuda.is_available():
        return []
    selected: list[tuple[int, int]] = []
    min_free_bytes = int(max(min_free_memory_gb, 0.0) * 1024**3)
    for device_id in range(torch.cuda.device_count()):
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(device_id)
        except TypeError:
            with torch.cuda.device(device_id):
                free_bytes, _total_bytes = torch.cuda.mem_get_info()
        if free_bytes >= min_free_bytes:
            selected.append((device_id, int(free_bytes)))
    selected.sort(key=lambda item: item[1], reverse=True)
    if max_gpus is not None and max_gpus > 0:
        selected = selected[:max_gpus]
    return [device_id for device_id, _free_bytes in selected]


def runtime_device_and_gpu_ids(
    device_value: str,
    multi_gpu: object = "off",
    min_free_memory_gb: float = 0.0,
    max_gpus: int | None = None,
) -> tuple[torch.device, list[int]]:
    requested = choose_device(device_value)
    if requested.type != "cuda" or not torch.cuda.is_available():
        return requested, []

    mode = _multi_gpu_mode(multi_gpu)
    if mode == "off" or torch.cuda.device_count() < 2:
        if requested.index is not None:
            return requested, []
        return torch.device("cuda:0"), []

    device_ids = select_cuda_device_ids(min_free_memory_gb=min_free_memory_gb, max_gpus=max_gpus)
    if len(device_ids) < 2 and mode == "on":
        fallback_ids = list(range(torch.cuda.device_count()))
        if max_gpus is not None and max_gpus > 0:
            fallback_ids = fallback_ids[:max_gpus]
        device_ids = fallback_ids
    if len(device_ids) < 2:
        if requested.index is not None:
            return requested, []
        first = device_ids[0] if device_ids else 0
        return torch.device(f"cuda:{first}"), []
    return torch.device(f"cuda:{device_ids[0]}"), device_ids


def masked_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, loss_name: str, huber_delta: float) -> torch.Tensor:
    valid = mask.bool()
    if not torch.any(valid):
        return pred.sum() * 0.0
    if loss_name == "mse":
        return torch.mean((pred[valid] - target[valid]) ** 2)
    if loss_name == "huber":
        return torch.nn.functional.huber_loss(pred[valid], target[valid], delta=huber_delta)
    raise ValueError(f"Unsupported loss: {loss_name}")


def graph_tensors(graph: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    adj = make_sparse_adjacency(graph, device)
    order = torch.as_tensor(graph["hilbert_order"], dtype=torch.long, device=device)
    inverse_order = torch.as_tensor(graph["inverse_hilbert_order"], dtype=torch.long, device=device)
    return adj, order, inverse_order


def save_loss_plot(history: list[dict[str, float]], output_path: str | Path) -> Path:
    """Save an epoch-vs-training-loss plot."""
    import matplotlib.pyplot as plt

    if not history:
        raise ValueError("Cannot plot an empty training history.")
    output = Path(output_path)
    ensure_dir(output.parent)
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_loss, marker="o", linewidth=1.8, label="Train loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss by Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def make_progress_bar(
    iterable: Any,
    desc: str,
    show_progress: bool,
    position: int = 0,
    leave: bool = True,
    total: int | None = None,
) -> Any:
    if not show_progress:
        return iterable
    bar = tqdm(
        iterable,
        desc=desc,
        total=total,
        position=position,
        leave=leave,
        dynamic_ncols=True,
    )
    bar.refresh()
    return bar


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: float,
    target_std: float,
    nearest_tempo_distance_km: np.ndarray,
    desc: str = "evaluate",
    show_progress: bool = True,
) -> dict[str, float]:
    model.eval()
    pred_raw_all = []
    y_raw_all = []
    mask_all = []
    dt_all = []
    dist_all = []
    iterator = make_progress_bar(loader, desc=desc, show_progress=show_progress, position=2, leave=True, total=len(loader))
    for batch in iterator:
        x = batch["x"].to(device)
        dt = batch["dt"].to(device)
        y_raw = batch["y_raw"].numpy()
        mask = batch["y_mask"].numpy().astype(bool)
        pred_scaled = model(x, dt).cpu().numpy()
        pred_raw = pred_scaled * target_std + target_mean
        pred_raw_all.append(pred_raw)
        y_raw_all.append(y_raw)
        mask_all.append(mask)
        dt_np = batch["dt"][:, -1].numpy()
        dt_all.append(np.repeat(dt_np[:, None], mask.shape[1], axis=1))
        dist_all.append(np.repeat(nearest_tempo_distance_km[None, :], mask.shape[0], axis=0))

    pred = np.concatenate(pred_raw_all, axis=0)
    truth = np.concatenate(y_raw_all, axis=0)
    mask = np.concatenate(mask_all, axis=0)
    abs_err = np.abs(pred - truth)
    dt_values = np.concatenate(dt_all, axis=0)
    dist_values = np.concatenate(dist_all, axis=0)
    return {
        "rmse": masked_rmse(truth, pred, mask),
        "r2": masked_r2(truth, pred, mask),
        "abs_error_delta_t_corr": pearson_corr(dt_values[mask], abs_err[mask]),
        "abs_error_nearest_tempo_distance_corr": pearson_corr(dist_values[mask], abs_err[mask]),
        "samples": int(pred.shape[0]),
        "observations": int(np.count_nonzero(mask)),
    }


def train(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    run_paths = resolve_run_paths(config)

    checkpoint_dir = ensure_dir(run_paths["checkpoint_dir"])
    output_dir = ensure_dir(run_paths["output_dir"])
    graph_path = Path(run_paths["graph_path"])
    if not graph_path.exists():
        graph_summary = build_unified_graph(
            tempo_dir=paths["tempo_dir"],
            airnow_dir=paths["airnow_dir"],
            graph_path=graph_path,
            start_date=data_cfg.get("start_date"),
            end_date=data_cfg.get("end_date"),
            local_context_k=int(data_cfg.get("local_context_k", 5)),
            station_neighbor_k=int(data_cfg.get("station_neighbor_k", 5)),
            tempo_neighbor_k=int(data_cfg.get("tempo_neighbor_k", 4)),
        )
        print(json.dumps({"built_graph": graph_summary}, indent=2))

    set_seed(int(train_cfg.get("seed", 7)))
    base_dataset = TempoAirNowDataset(
        graph_path=graph_path,
        tempo_dir=paths["tempo_dir"],
        airnow_dir=paths["airnow_dir"],
        start_date=data_cfg.get("start_date"),
        end_date=data_cfg.get("end_date"),
        context_steps=int(data_cfg.get("context_steps", 8)),
        max_time_snap_hours=float(data_cfg.get("max_time_snap_hours", 0.75)),
        max_column=float(data_cfg.get("max_column", 5.0e16)),
        min_valid_targets=int(data_cfg.get("min_valid_targets", 1)),
    )

    train_end, val_end, test_end = split_indices(
        len(base_dataset),
        float(train_cfg.get("train_fraction", 0.70)),
        float(train_cfg.get("val_fraction", 0.15)),
    )
    train_dataset = base_dataset.subset(base_dataset.end_indices[train_end])
    val_dataset = base_dataset.subset(base_dataset.end_indices[val_end])
    test_dataset = base_dataset.subset(base_dataset.end_indices[test_end])
    target_stats = train_dataset.fit_target_scaler()
    val_dataset.with_target_scaler(train_dataset.target_mean, train_dataset.target_std)
    test_dataset.with_target_scaler(train_dataset.target_mean, train_dataset.target_std)

    graph = load_graph_npz(graph_path)
    max_gpus = train_cfg.get("gpu_max_count")
    max_gpus = int(max_gpus) if max_gpus not in (None, "null", "") else None
    device, device_ids = runtime_device_and_gpu_ids(
        str(train_cfg.get("device", "auto")),
        multi_gpu=train_cfg.get("multi_gpu", "auto"),
        min_free_memory_gb=float(train_cfg.get("gpu_min_free_memory_gb", 0.0)),
        max_gpus=max_gpus,
    )
    adj, order, inverse_order = graph_tensors(graph, device)
    base_model = UnifiedGraphMambaPredictor(
        input_dim=int(model_cfg.get("input_dim", base_dataset.feature_dim)),
        hidden_dim=int(model_cfg["hidden_dim"]),
        state_dim=int(model_cfg["state_dim"]),
        num_airnow=int(graph["metadata"]["num_airnow"]),
        spatial_layers=int(model_cfg.get("spatial_layers", 2)),
        temporal_layers=int(model_cfg.get("temporal_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)
    model = GraphBoundPredictor(base_model, adj, order, inverse_order).to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
        print(json.dumps({"multi_gpu": True, "device_ids": device_ids}, indent=2))
    else:
        print(json.dumps({"multi_gpu": False, "device": str(device)}, indent=2))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate_samples,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg.get("batch_size", 2)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate_samples,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(train_cfg.get("batch_size", 2)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate_samples,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )

    best_val_rmse = float("inf")
    best_path = Path(run_paths["checkpoint_path"])
    history = []
    nearest_dist = graph["nearest_tempo_distance_km"].astype(np.float64)

    epochs = int(train_cfg.get("epochs", 25))
    show_progress = bool(train_cfg.get("show_progress", True))
    epoch_progress = make_progress_bar(
        range(1, epochs + 1),
        desc="epochs",
        show_progress=show_progress,
        position=0,
        leave=True,
        total=epochs,
    )
    for epoch in epoch_progress:
        model.train()
        losses = []
        progress = make_progress_bar(
            train_loader,
            desc=f"epoch {epoch}/{epochs} train batches",
            show_progress=show_progress,
            position=1,
            leave=True,
            total=len(train_loader),
        )
        for batch in progress:
            x = batch["x"].to(device)
            dt = batch["dt"].to(device)
            y = batch["y"].to(device)
            mask = batch["y_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, dt)
            loss = masked_loss(
                pred,
                y,
                mask,
                str(train_cfg.get("loss", "huber")),
                float(train_cfg.get("huber_delta", 1.0)),
            )
            loss.backward()
            clip_norm = float(train_cfg.get("grad_clip_norm", 0.0))
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if show_progress:
                progress.set_postfix(loss=np.mean(losses))

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            train_dataset.target_mean,
            train_dataset.target_std,
            nearest_dist,
            desc=f"epoch {epoch}/{epochs} val",
            show_progress=show_progress,
        )
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        if show_progress:
            epoch_progress.set_postfix(train_loss=row["train_loss"], val_rmse=val_metrics["rmse"])
        print(json.dumps(row))
        if np.isfinite(val_metrics["rmse"]) and val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            torch.save(
                {
                    "model": base_model.state_dict(),
                    "base_model": base_model.state_dict(),
                    "config": config,
                    "target_scaler": target_stats.state_dict(),
                    "graph_metadata": graph["metadata"],
                    "device_ids": device_ids,
                },
                best_path,
            )

    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        base_model.load_state_dict(checkpoint.get("base_model", checkpoint["model"]))
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        train_dataset.target_mean,
        train_dataset.target_std,
        nearest_dist,
        desc="test",
        show_progress=show_progress,
    )
    loss_plot_path = save_loss_plot(history, Path(output_dir) / "loss_curve.png") if history else None
    result = {
        "target_scaler": target_stats.state_dict(),
        "best_model": str(best_path),
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "graph_path": str(graph_path),
        "loss_plot": str(loss_plot_path) if loss_plot_path is not None else "",
        "device": str(device),
        "device_ids": device_ids,
        "history": history,
        "test": test_metrics,
        "graph": graph["metadata"],
    }
    with Path(run_paths["metrics_path"]).open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the unified Graph-Mamba predictor.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    result = train(load_config(args.config))
    print(json.dumps({"test": result["test"], "best_model": result["best_model"]}, indent=2))


if __name__ == "__main__":
    main()
