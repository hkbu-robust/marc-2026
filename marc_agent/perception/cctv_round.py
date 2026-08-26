"""Fetch and cache CCTV frames for one Stage 1 mission round."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

from marc_agent.ros_image import image_to_rgb
from marc_agent.types import CctvFrame

if TYPE_CHECKING:
    from marc_sdk import MARCClient


log = logging.getLogger("marc_agent.cctv_round")


def fetch_cctv_round(
    client: MARCClient,
    *,
    discovery_timeout_s: float | None = None,
    discovery_settle_s: float = 0.25,
    poll_interval_s: float = 0.05,
) -> tuple[dict[str, CctvFrame], list[str]]:
    """Discover cameras and decode each frame at most once for this round.

    Returns:
        frames: camera_id -> decoded frame (only cameras with a usable image)
        camera_ids: sorted ids from ``client.list_cctv()``
    """
    camera_ids = sorted(client.list_cctv())
    if not camera_ids:
        timeout_s = (
            float(os.environ.get("MARC_CCTV_DISCOVERY_TIMEOUT_S") or 10.0)
            if discovery_timeout_s is None
            else discovery_timeout_s
        )
        wait_started = time.monotonic()
        log.info(
            "[cctv] no cameras discovered; waiting up to %.1fs for subscriptions",
            timeout_s,
        )
        deadline = time.monotonic() + max(0.0, timeout_s)
        stable_since: float | None = None
        while time.monotonic() < deadline:
            discovered = sorted(client.list_cctv())
            now = time.monotonic()
            if discovered != camera_ids:
                camera_ids = discovered
                stable_since = now if camera_ids else None
            elif camera_ids and stable_since is not None:
                if now - stable_since >= max(0.0, discovery_settle_s):
                    break
            time.sleep(max(0.001, poll_interval_s))
        elapsed_ms = (time.monotonic() - wait_started) * 1000.0
        if camera_ids:
            log.info(
                "[cctv] discovered %d cameras after %.0fms: %s",
                len(camera_ids),
                elapsed_ms,
                ",".join(camera_ids),
            )
        else:
            log.warning(
                "[cctv] discovery timed out after %.0fms with no cameras",
                elapsed_ms,
            )
    frames: dict[str, CctvFrame] = {}
    for camera_id in camera_ids:
        msg = client.get_cctv_image(camera_id)
        rgb = image_to_rgb(msg)
        if rgb is None:
            continue
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        frames[camera_id] = CctvFrame(
            camera_id=camera_id,
            width=width,
            height=height,
            rgb=rgb,
        )
    return frames, camera_ids
