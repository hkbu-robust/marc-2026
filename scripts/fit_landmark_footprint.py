"""Fit the landmark footprint prior (#112) on held-out generator GT, even scenes only.

    python -m scripts.fit_landmark_footprint \
        --gt '~/marc-2026-work/experiments/ablations/05-frozen-eval-sets/heldout-*/gt/marc2026_chungmu/scene_*/rig_*.json'

For every fully visible landmark box, the prim origin (the point the scorer uses
as the anchor) is projected at its ground height into the image and expressed as
fractions (fu, fv) of the GT box; dz is the origin's height above the ground.
The prior is the per-class median. Only even-numbered scenes are used, so the
odd scenes stay a held-out test (#89/#96 use these cases). Training-side GT is not
used: an older generator pinned its landmarks to the flat plane.

Prints the table to paste into ``cctv_project.LANDMARK_FOOTPRINT``.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from pathlib import Path

import numpy as np

from marc_agent.command_parser import LANDMARK_TYPES
from marc_agent.perception.ground_model import GroundModel
from scripts.stage1_replay import load_geometry

MAX_OCCLUSION = 0.3
MIN_BOX_HEIGHT_FRAC = 0.01  # of the image height; smaller boxes give noisy fractions
EDGE_PX = 1.0


def scene_number(path: str) -> int:
    return int(Path(path).parent.name.split("_")[-1])


def to_pixel(geometry, xyz) -> tuple[float, float] | None:
    rotation = np.asarray(geometry.rotation, dtype=float)
    p_cam = rotation.T @ (np.asarray(xyz, dtype=float) - np.asarray(geometry.translation))
    if p_cam[2] <= 1e-6:
        return None
    return (geometry.cx + geometry.fx * p_cam[0] / p_cam[2],
            geometry.cy + geometry.fy * p_cam[1] / p_cam[2])


def samples(files: list[str], ground: GroundModel, geometry: dict) -> dict[str, list[tuple]]:
    out = collections.defaultdict(list)
    for path in files:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        camera = payload["frame_id"]
        width, height = payload["image"]["width"], payload["image"]["height"]
        surface, _ = ground.for_camera(camera, geometry[camera])
        if surface is None:
            continue
        for det in payload["detections"]:
            if det["class"] not in LANDMARK_TYPES or "world_xyz" not in det:
                continue
            if det.get("occlusion", -1.0) > MAX_OCCLUSION:
                continue
            x1, y1, x2, y2 = det["bbox"]
            if (x1 <= EDGE_PX or y1 <= EDGE_PX or x2 >= width - 1 - EDGE_PX
                    or y2 >= height - 1 - EDGE_PX or (y2 - y1) < MIN_BOX_HEIGHT_FRAC * height):
                continue
            x, y, z = det["world_xyz"]
            ground_z = surface.z(x, y)
            if ground_z is None:
                continue
            pixel = to_pixel(geometry[camera], (x, y, ground_z))
            if pixel is None:
                continue
            out[det["class"]].append(
                ((pixel[0] - x1) / (x2 - x1), (pixel[1] - y1) / (y2 - y1), z - ground_z)
            )
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gt", required=True)
    parser.add_argument("--parity", choices=["even", "odd"], default="even")
    args = parser.parse_args(argv)
    files = [
        path for path in sorted(glob.glob(os.path.expanduser(args.gt)))
        if scene_number(path) % 2 == (0 if args.parity == "even" else 1)
    ]
    rows = samples(files, GroundModel.load(), load_geometry())
    print(f"# {len(files)} GT frames, {args.parity} scenes")
    print("LANDMARK_FOOTPRINT = {")
    for name in sorted(rows):
        values = np.asarray(rows[name])
        fu, fv, dz = np.median(values, axis=0)
        q1, q3 = np.percentile(values, [25, 75], axis=0)
        print(f'    "{name}": ({fu:.3f}, {fv:.3f}, {dz:.3f}),  # n={len(values)} '
              f'IQR fv {q1[1]:.2f}-{q3[1]:.2f} dz {q1[2]:+.2f}-{q3[2]:+.2f}')
    print("}")


if __name__ == "__main__":
    main()
