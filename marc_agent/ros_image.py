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
    step = int(getattr(msg, "step", 0) or 0)

    def _hwc(nchan: int, dtype=np.uint8):
        raw = np.frombuffer(msg.data, dtype=dtype)
        row = step if step else (w * nchan * np.dtype(dtype).itemsize)
        item = np.dtype(dtype).itemsize
        row_elems = max(nchan * w, row // item)
        need = h * row_elems
        if raw.size < h * nchan * w:
            return None
        if row_elems == nchan * w and raw.size >= h * nchan * w:
            return raw[: h * nchan * w].reshape(h, w, nchan)
        if raw.size < need:
            return raw[: h * nchan * w].reshape(h, w, nchan)
        return raw[:need].reshape(h, row_elems)[:, : nchan * w].reshape(h, w, nchan)

    if enc in ("rgb8", "rgb"):
        arr = _hwc(3)
        return None if arr is None else np.ascontiguousarray(arr)
    if enc in ("bgr8", "bgr", "8uc3"):
        arr = _hwc(3)
        return None if arr is None else np.ascontiguousarray(arr[:, :, ::-1])
    if enc in ("rgba8", "bgra8"):
        arr = _hwc(4)
        if arr is None:
            return None
        rgb = arr[:, :, :3] if enc == "rgba8" else arr[:, :, 2::-1]
        return np.ascontiguousarray(rgb)
    if enc in ("mono8", "8uc1"):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        if arr.size < h * w:
            return None
        plane = arr[: h * w].reshape(h, w)
        return np.ascontiguousarray(np.stack([plane, plane, plane], axis=-1))
    return None
