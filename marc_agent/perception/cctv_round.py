"""Fetch and cache CCTV frames for one Stage 1 mission round."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Optional

from marc_agent.ros_image import image_to_rgb
from marc_agent.types import CctvFrame

if TYPE_CHECKING:
    from marc_sdk import MARCClient


log = logging.getLogger("marc_agent.cctv_round")


def _timeout_s(discovery_timeout_s: Optional[float]) -> float:
    if discovery_timeout_s is not None:
        return max(0.0, float(discovery_timeout_s))
    return max(0.0, float(os.environ.get("MARC_CCTV_DISCOVERY_TIMEOUT_S") or 10.0))


def discovery_timeout_from_limit(time_limit: float | None) -> Optional[float]:
    """CCTV startup wait while reserving at least 15s for detection and selection."""
    limit = float(time_limit or 0.0)
    if limit <= 0.0:
        return None
    if limit <= 15.0:
        return min(8.0, max(1.0, limit * 0.4))
    return min(15.0, max(8.0, limit - 15.0))


def _kick_discover(client) -> None:
    discover = getattr(client, "_discover_cctv", None)
    if not callable(discover):
        return
    try:
        discover()
    except Exception as exc:
        log.debug("[cctv] discover failed: %s", exc)


def _decode_frames(client, camera_ids: list[str]) -> dict[str, CctvFrame]:
    frames: dict[str, CctvFrame] = {}
    for camera_id in camera_ids:
        rgb = image_to_rgb(client.get_cctv_image(camera_id))
        if rgb is None:
            continue
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        frames[camera_id] = CctvFrame(
            camera_id=camera_id,
            width=width,
            height=height,
            rgb=rgb,
        )
    return frames


def _intrinsics_ready(client, camera_ids: list[str]) -> bool:
    """True when each id has CameraInfo and ground height, or the client has no such getters."""
    info_fn = getattr(client, "get_cctv_info", None)
    height_fn = getattr(client, "get_cctv_ground_height", None)
    if info_fn is None and height_fn is None:
        return True
    for camera_id in camera_ids:
        if info_fn is not None and info_fn(camera_id) is None:
            return False
        if height_fn is not None and height_fn(camera_id) is None:
            return False
    return True


def fetch_cctv_round(
    client: "MARCClient",
    *,
    discovery_timeout_s: float | None = None,
    discovery_settle_s: float = 0.25,
    poll_interval_s: float = 0.05,
) -> tuple[dict[str, CctvFrame], list[str]]:
    """Discover cameras and decode each frame at most once for this round.

    Invokes ``client._discover_cctv`` on each poll when present so the wait does
    not depend on the SDK's 2s timer. Empty ``list_cctv()`` is never treated as
    ready. After ids appear, waits for decoded frames and CameraInfo / ground
    height when the client exposes them.

    Returns:
        frames: camera_id -> decoded frame (only cameras with a usable image)
        camera_ids: sorted ids from ``client.list_cctv()``
    """
    timeout_s = _timeout_s(discovery_timeout_s)
    poll = max(0.001, poll_interval_s)
    settle = max(0.0, discovery_settle_s)
    wait_started = time.monotonic()

    if timeout_s <= 0.0:
        _kick_discover(client)
        camera_ids = sorted(client.list_cctv())
        frames = _decode_frames(client, camera_ids)
        _log_fetch(camera_ids, frames, wait_started)
        return frames, camera_ids

    deadline = time.monotonic() + timeout_s
    _kick_discover(client)
    camera_ids = sorted(client.list_cctv())
    if not camera_ids:
        log.info(
            "[cctv] no cameras discovered; waiting up to %.1fs for subscriptions",
            timeout_s,
        )

    stable_since: float | None = time.monotonic() if camera_ids else None
    while time.monotonic() < deadline:
        _kick_discover(client)
        discovered = sorted(client.list_cctv())
        now = time.monotonic()
        if discovered != camera_ids:
            camera_ids = discovered
            stable_since = now if camera_ids else None
        elif camera_ids and stable_since is not None:
            if now - stable_since >= settle:
                break
        time.sleep(poll)

    frames = _decode_frames(client, camera_ids)
    while time.monotonic() < deadline and camera_ids:
        _kick_discover(client)
        discovered = sorted(client.list_cctv())
        if discovered != camera_ids:
            camera_ids = discovered
            frames = _decode_frames(client, camera_ids)
            continue
        ready_ids = [cid for cid in camera_ids if cid in frames]
        if ready_ids and len(ready_ids) == len(camera_ids) and _intrinsics_ready(
            client, ready_ids
        ):
            break
        frames = _decode_frames(client, camera_ids)
        time.sleep(poll)

    _log_fetch(camera_ids, frames, wait_started)
    return frames, camera_ids


def _log_fetch(
    camera_ids: list[str],
    frames: dict[str, CctvFrame],
    wait_started: float,
) -> None:
    elapsed_ms = (time.monotonic() - wait_started) * 1000.0
    have = sorted(frames)
    missing = [cid for cid in camera_ids if cid not in frames]
    if camera_ids:
        log.info(
            "[cctv] cameras=%d frames=%d missing=%d elapsed=%.0fms have=%s miss=%s",
            len(camera_ids),
            len(frames),
            len(missing),
            elapsed_ms,
            ",".join(have) or "-",
            ",".join(missing) or "-",
        )
        return
    log.warning(
        "[cctv] discovery timed out after %.0fms with no cameras",
        elapsed_ms,
    )
