"""Fetch and cache CCTV frames for one Stage 1 mission round."""

from __future__ import annotations

from typing import TYPE_CHECKING

from marc_agent.ros_image import image_to_rgb
from marc_agent.types import CctvFrame

if TYPE_CHECKING:
    from marc_sdk import MARCClient


def fetch_cctv_round(client: MARCClient) -> tuple[dict[str, CctvFrame], list[str]]:
    """Discover cameras and decode each frame at most once for this round.

    Returns:
        frames: camera_id -> decoded frame (only cameras with a usable image)
        camera_ids: sorted ids from ``client.list_cctv()``
    """
    camera_ids = list(client.list_cctv())
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
