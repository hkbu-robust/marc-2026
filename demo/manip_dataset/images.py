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


def depth_roi_median(depth: np.ndarray, frac: float = 0.4) -> Optional[float]:
    """Median of valid positive finite depths in the central ROI (frac of width/height)."""
    if depth is None or depth.size == 0:
        return None
    h, w = depth.shape[:2]
    fh = max(1, int(h * frac))
    fw = max(1, int(w * frac))
    y0 = (h - fh) // 2
    x0 = (w - fw) // 2
    roi = depth[y0:y0 + fh, x0:x0 + fw]
    valid = roi[np.isfinite(roi) & (roi > 0.05) & (roi < 20.0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))
