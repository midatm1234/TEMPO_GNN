from __future__ import annotations

import numpy as np


def _rot(n: int, x: int, y: int, rx: int, ry: int) -> tuple[int, int]:
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y


def hilbert_index(x: int, y: int, order: int) -> int:
    """Return the Hilbert distance for integer coordinates in a 2**order grid."""
    n = 1 << order
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) else 0
        ry = 1 if (y & s) else 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = _rot(s, x, y, rx, ry)
        s //= 2
    return d


def hilbert_order(lon: np.ndarray, lat: np.ndarray, order: int = 16) -> np.ndarray:
    """Linearize lon/lat points with a Hilbert curve over their bounding box."""
    if lon.shape != lat.shape:
        raise ValueError("lon and lat must have the same shape.")
    if lon.ndim != 1:
        raise ValueError("lon and lat must be one-dimensional arrays.")

    lon_min = float(np.nanmin(lon))
    lon_max = float(np.nanmax(lon))
    lat_min = float(np.nanmin(lat))
    lat_max = float(np.nanmax(lat))
    scale = (1 << order) - 1

    lon_span = max(lon_max - lon_min, 1e-12)
    lat_span = max(lat_max - lat_min, 1e-12)
    xs = np.clip(np.rint((lon - lon_min) / lon_span * scale), 0, scale).astype(np.int64)
    ys = np.clip(np.rint((lat - lat_min) / lat_span * scale), 0, scale).astype(np.int64)
    keys = np.fromiter((hilbert_index(int(x), int(y), order) for x, y in zip(xs, ys)), dtype=np.int64)
    return np.argsort(keys, kind="stable")

