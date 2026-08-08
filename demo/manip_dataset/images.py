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


def depth_roi_median(depth: np.ndarray, frac: float = 0.4) -> Optional[float]:
    """Median of valid depths in the central ROI (compat wrapper)."""
    return depth_roi_stat(depth, frac=frac, percentile=50.0)
