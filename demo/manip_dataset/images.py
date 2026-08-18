"""Decode sensor_msgs/Image to numpy without cv_bridge."""

from __future__ import annotations

from typing import Optional

import numpy as np


def image_to_rgb(msg) -> Optional[np.ndarray]:
    """Convert sensor_msgs/Image to HxWx3 uint8 RGB. Returns None if unsupported/empty."""
    if msg is None:
        return None
    h, w = int(msg.height), int(msg.width)
    if h <= 0 or w <= 0:
        return None
    enc = (msg.encoding or "").lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ("rgb8", "rgb"):
        arr = data.reshape(h, w, 3)
        return np.ascontiguousarray(arr)
    if enc in ("bgr8", "bgr"):
        arr = data.reshape(h, w, 3)
        return np.ascontiguousarray(arr[:, :, ::-1])
    if enc in ("rgba8", "bgra8"):
        arr = data.reshape(h, w, 4)
        rgb = arr[:, :, :3] if enc == "rgba8" else arr[:, :, 2::-1]
        return np.ascontiguousarray(rgb)
    if enc in ("mono8", "8uc1"):
        arr = data.reshape(h, w)
        return np.ascontiguousarray(np.stack([arr, arr, arr], axis=-1))
    return None


def image_to_depth(msg) -> Optional[np.ndarray]:
    """Convert sensor_msgs/Image depth to HxW float32 meters. Returns None if unsupported."""
    if msg is None:
        return None
    h, w = int(msg.height), int(msg.width)
    if h <= 0 or w <= 0:
        return None
    enc = (msg.encoding or "").lower()

    if enc in ("32fc1", "32fc"):
        arr = np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)
        return np.ascontiguousarray(arr)
    if enc in ("16uc1", "mono16"):
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w).astype(np.float32)
        # Common ROS convention: mm -> m
        return np.ascontiguousarray(arr * 0.001)
    return None


def depth_roi_stat(
    depth: np.ndarray,
    frac: float = 0.2,
    percentile: float = 20.0,
) -> Optional[float]:
    """Near-biased depth in a central ROI (meters).

    Uses a lower percentile (default p20) instead of the median so small/near
    objects in a mostly-floor frame still pull the statistic down.
    """
    if depth is None or depth.size == 0:
        return None
    h, w = depth.shape[:2]
    frac = float(np.clip(frac, 0.05, 1.0))
    fh = max(1, int(h * frac))
    fw = max(1, int(w * frac))
    y0 = (h - fh) // 2
    x0 = (w - fw) // 2
    roi = depth[y0:y0 + fh, x0:x0 + fw]
    valid = roi[np.isfinite(roi) & (roi > 0.05) & (roi < 20.0)]
    if valid.size == 0:
        return None
    p = float(np.clip(percentile, 0.0, 100.0))
    return float(np.percentile(valid, p))


def depth_center_surround(
    depth: np.ndarray,
    inner_frac: float = 0.18,
    outer_frac: float = 0.40,
    percentile: float = 30.0,
    min_valid: int = 4,
) -> Optional[dict]:
    """Center-vs-surround depth contrast (meters).

    Returns a dict with ``d_inner``, ``d_ring``, ``contrast`` (= ring − center,
    positive when the center is closer), plus valid-pixel counts and the reason
    when a region could not be measured.

    A pitch change shifts center and ring together, so contrast stays flat; an
    object entering the center lowers only ``d_inner`` and contrast rises.
    """
    if depth is None or depth.size == 0:
        return None
    h, w = depth.shape[:2]
    inner_frac = float(np.clip(inner_frac, 0.04, 0.9))
    outer_frac = float(np.clip(max(outer_frac, inner_frac + 0.05), 0.1, 1.0))
    p = float(np.clip(percentile, 0.0, 100.0))
    min_valid = max(1, int(min_valid))

    def _block(frac: float):
        fh = max(1, int(h * frac))
        fw = max(1, int(w * frac))
        y0 = (h - fh) // 2
        x0 = (w - fw) // 2
        return depth[y0:y0 + fh, x0:x0 + fw], y0, x0, fh, fw

    inner, iy0, ix0, ih, iw = _block(inner_frac)
    outer, oy0, ox0, _, _ = _block(outer_frac)

    ring = np.array(outer, dtype=np.float64, copy=True)
    ry0 = max(0, iy0 - oy0)
    rx0 = max(0, ix0 - ox0)
    ring[ry0:ry0 + ih, rx0:rx0 + iw] = np.nan

    def _pct(block: np.ndarray):
        valid = block[np.isfinite(block) & (block > 0.05) & (block < 20.0)]
        if valid.size < min_valid:
            return None, int(valid.size)
        return float(np.percentile(valid, p)), int(valid.size)

    d_inner, n_inner = _pct(inner)
    d_ring, n_ring = _pct(ring)

    out = {
        "d_inner": d_inner,
        "d_ring": d_ring,
        "n_inner": n_inner,
        "n_ring": n_ring,
        "shape": (int(h), int(w)),
        "contrast": None,
        "reason": "",
    }
    if d_inner is None or d_ring is None:
        out["reason"] = (
            f"inner_valid={n_inner} ring_valid={n_ring} min={min_valid}"
        )
        return out
    out["contrast"] = float(d_ring - d_inner)
    return out


def depth_lower_stat(
    depth: np.ndarray,
    y0_frac: float = 0.70,
    y1_frac: float = 1.00,
    percentile: float = 30.0,
    min_m: float = 0.05,
    max_m: float = 20.0,
    min_valid: int = 8,
) -> Optional[float]:
    """Near-biased depth in a bottom image band (meters).

    A downward-tilted base camera puts a close object in the lower rows once it
    has left the center patch. Uses a low percentile so a small object still
    registers against floor and self-parts in the same band.
    """
    if depth is None or depth.size == 0:
        return None
    h, w = depth.shape[:2]
    if h < 2 or w < 2:
        return None
    y0 = int(np.clip(min(y0_frac, y1_frac), 0.0, 1.0) * h)
    y1 = int(np.clip(max(y0_frac, y1_frac), 0.0, 1.0) * h)
    if y1 <= y0:
        y0, y1 = int(0.70 * h), h
    band = depth[y0:y1, :]
    valid = band[np.isfinite(band) & (band > float(min_m)) & (band < float(max_m))]
    if valid.size < max(1, int(min_valid)):
        return None
    p = float(np.clip(percentile, 0.0, 100.0))
    return float(np.percentile(valid, p))


def depth_lateral_error(
    depth: np.ndarray,
    y0_frac: float = 0.30,
    y1_frac: float = 0.85,
    percentile: float = 30.0,
    min_valid: int = 4,
    closer_m: float = 0.04,
    min_m: float = 0.05,
    max_m: float = 20.0,
) -> Optional[dict]:
    """Signed lateral error of the nearest object vs image center.

    Restricts to a mid-height band, takes a near percentile per column, treats the
    median of those columns as floor, and centroids columns that are closer than
    floor by ``closer_m``. Only depths inside ``[min_m, max_m]`` count, which keeps
    self-parts (arm in view) from being measured as the floor.

    Returns ``error`` in roughly ``[-1, 1]`` (image-right positive), plus extras,
    or ``None`` if the band has no usable columns. ``error`` is 0.0 when a floor
    is measured but no object-like columns exist (caller should not yaw).
    """
    if depth is None or depth.size == 0:
        return None
    h, w = depth.shape[:2]
    if w < 2 or h < 2:
        return None
    y0 = int(np.clip(min(y0_frac, y1_frac), 0.0, 1.0) * h)
    y1 = int(np.clip(max(y0_frac, y1_frac), 0.0, 1.0) * h)
    if y1 <= y0:
        y0, y1 = 0, h
    band = depth[y0:y1, :]
    p = float(np.clip(percentile, 0.0, 100.0))
    min_valid = max(1, int(min_valid))
    col_d = np.full(w, np.nan, dtype=np.float64)
    for x in range(w):
        col = band[:, x]
        valid = col[np.isfinite(col) & (col > min_m) & (col < max_m)]
        if valid.size >= min_valid:
            col_d[x] = float(np.percentile(valid, p))
    usable = np.isfinite(col_d)
    if not np.any(usable):
        return None
    floor = float(np.median(col_d[usable]))
    closer = usable & (col_d < (floor - closer_m))
    if not np.any(closer):
        return {
            "error": 0.0,
            "col": None,
            "floor": floor,
            "n_obj": 0,
            "n_cols": int(np.count_nonzero(usable)),
        }
    xs = np.arange(w, dtype=np.float64)
    weight = floor - col_d
    weight = np.where(closer, np.clip(weight, 0.0, None), 0.0)
    wsum = float(weight.sum())
    if wsum <= 0.0:
        return {
            "error": 0.0,
            "col": None,
            "floor": floor,
            "n_obj": 0,
            "n_cols": int(np.count_nonzero(usable)),
        }
    centroid = float((xs * weight).sum() / wsum)
    error = (centroid - 0.5 * (w - 1)) / (0.5 * (w - 1))
    return {
        "error": float(np.clip(error, -1.0, 1.0)),
        "col": centroid,
        "floor": floor,
        "n_obj": int(np.count_nonzero(closer)),
        "n_cols": int(np.count_nonzero(usable)),
    }
