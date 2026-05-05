# Unified Graph-Mamba Predictor

This repository builds a supervised spatio-temporal regression model for AirNow
NO2 using TEMPO `vertical_column_troposphere` and a unified graph containing
satellite grid nodes plus AirNow ground station nodes.

## Setup

Use the local Prithvi mamba environment:

```bash
/mnt/data2/kyo/.mamba/envs/Prithvi/bin/python -m pip install -e .
```

On this machine, calling the environment Python directly is more reliable than
`conda run` or `conda activate`.

The default paths assume:

```text
/mnt/data3/TEMPO/NO2_L3_V03
/mnt/data3/AirNow
```

## Build The Adaptive Graph

```bash
/mnt/data2/kyo/.mamba/envs/Prithvi/bin/graphmamba-build-graph --config configs/default.yaml
```

The graph builder:

- loads AirNow station coordinates;
- loads the TEMPO latitude/longitude grid from one L3 file;
- computes each station's nearest TEMPO distance;
- chooses enough local TEMPO neighbors to give every AirNow station at least
  `local_context_k` satellite neighbors;
- records the adaptive radius needed to satisfy that condition;
- writes graph edges, node metadata, station-to-TEMPO distances, and a Hilbert
  node order.

## Train

```bash
/mnt/data2/kyo/.mamba/envs/Prithvi/bin/graphmamba-train --config configs/default.yaml
```

All notebook and CLI settings live in `configs/default.yaml`. `run_name` controls
where a run writes files:

```yaml
run_name: default

paths:
  checkpoint_root: checkpoints
  output_root: output
```

Training writes checkpoints to `checkpoints/<run_name>/` and graph/metrics
outputs to `output/<run_name>/`.

You can set the training date range and chronological validation split there:

```yaml
data:
  start_date: "2023-08-01"
  end_date: "2024-09-30"

training:
  train_fraction: 0.70
  val_fraction: 0.15
```

The test fraction is the remaining part of the aligned windows.

The trainer aligns TEMPO scans to the nearest AirNow hourly UTC timestamp,
computes actual scan-to-scan `delta_t_hours`, uses only the hours where both
TEMPO and AirNow data are present, and predicts AirNow NO2 from the hidden states
of AirNow nodes only.

Metrics include RMSE, R2, and error correlations with both `delta_t_hours` and
station-to-nearest-TEMPO distance.

Training uses notebook-friendly progress bars for epochs, batches, validation,
and test evaluation. It also writes an epoch-vs-loss plot to:

```text
output/<run_name>/loss_curve.png
```

Multi-GPU can be enabled from the same YAML file:

```yaml
training:
  device: auto
  multi_gpu: auto
  gpu_min_free_memory_gb: 2.0
  gpu_max_count:

inference:
  device: auto
  multi_gpu: auto
  gpu_min_free_memory_gb: 2.0
  gpu_max_count:
```

When multiple visible CUDA devices have enough free memory, the model uses
`torch.nn.DataParallel` over the batch dimension. Set `batch_size` at least as
large as the number of GPUs you want to use.

## Notebooks

- `notebooks/training.ipynb`: loads `configs/default.yaml` and trains with the
  `run_name`, `paths`, `data`, `model`, and `training` sections.
- `notebooks/inference.ipynb`: loads `configs/default.yaml`, reads the
  `inference` section, and exports station-time predictions under
  `output/<run_name>/`.

Inference writes a compiled AirNow-like NetCDF file by default:

```text
output/<run_name>/predictions.nc
```

The core prediction variable is `no2(time, site)`, matching the AirNow file
shape while compiling the temporal dimension across the requested inference
range. The file also includes `observed_no2`, `absolute_error`,
`has_observation`, `delta_t_hours`, and `nearest_tempo_distance_km`.
Inference also shows a progress bar over prediction batches.
