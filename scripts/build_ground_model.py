"""Build the calibrated ground model shipped with the agent (#112).

    python -m scripts.build_ground_model \
        --gt './work/datasets/raw/stage1-tiny-*-20260831/marc2026_chungmu/scene_*/rig_*.json' \
        [--check './work/experiments/ablations/05-frozen-eval-sets/heldout-*/gt/marc2026_chungmu/scene_*/rig_*.json'] \
        [--out marc_agent/perception/data/ground_model.npz]

Ground points are the GT world positions of every mission object (small items lie
on the ground or on furniture; the runtime's lower envelope drops the latter) in
training-side generator scenes. Never pass an evaluation set as ``--gt``.

Each camera's fingerprint is the runtime's own geometry (``scripts/data/
sdk_cameras.json``, captured through ``Tf2ClientLookup``); the build stops if a GT
camera block disagrees with it. ``--check`` reports, per camera, the ground-height
error on another set's ground objects for the flat plane and for this surface.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from marc_agent.command_parser import MISSION_OBJECT_TYPES
from marc_agent.perception.ground_model import DEFAULT_PATH, CameraFingerprint, GroundSurface
from scripts.stage1_replay import DEFAULT_GEOMETRY

GT_POSITION_TOLERANCE_M = 1e-3


def _files(pattern: str) -> list[str]:
    return sorted(glob.glob(os.path.expanduser(pattern)))


def ground_points(files: list[str]) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    points, cameras = collections.defaultdict(list), {}
    for path in files:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        camera = payload["frame_id"]
        cameras.setdefault(camera, payload["camera"])
        for det in payload["detections"]:
            if det["class"] in MISSION_OBJECT_TYPES and "world_xyz" in det:
                points[camera].append(det["world_xyz"])
    return {camera: np.asarray(xyz, dtype=float) for camera, xyz in points.items()}, cameras


def _gt_geometry(camera: dict):
    """A GT camera block in the runtime's convention: world-from-camera rotation
    Rx(roll) Ry(pitch) Rz(yaw) in USD axes, then USD -> optical (diag(1, -1, -1))."""
    import math
    from types import SimpleNamespace

    def rot(axis, degrees):
        c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
        return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
                "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
                "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]

    euler = camera["orientation_euler_deg"]
    rotation = rot("x", euler["roll"]) @ rot("y", euler["pitch"]) @ rot("z", euler["yaw"])
    return SimpleNamespace(rotation=rotation @ np.diag([1.0, -1.0, -1.0]),
                           translation=camera["position"], ground_height=camera["ground_height"])


def build(gt_files: list[str], geometry_path: Path, out: Path) -> dict:
    points, gt_cameras = ground_points(gt_files)
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))["geometry"]
    arrays, counts = {}, {}
    for camera in sorted(points):
        live = geometry[camera]
        gt = gt_cameras[camera]
        gap = np.linalg.norm(np.asarray(gt["position"]) - np.asarray(live["translation"]))
        if gap > GT_POSITION_TOLERANCE_M or abs(gt["ground_height"] - live["ground_height"]) > GT_POSITION_TOLERANCE_M:
            raise SystemExit(f"{camera}: GT camera block differs from {geometry_path}")
        fingerprint = CameraFingerprint(np.asarray(live["rotation"]), np.asarray(live["translation"]),
                                        float(live["ground_height"]))
        if fingerprint.mismatch(_gt_geometry(gt)) is not None:
            raise SystemExit(f"{camera}: GT camera orientation differs from {geometry_path}")
        arrays[f"{camera}/points"] = points[camera].astype(np.float32)
        arrays[f"{camera}/rotation"] = np.asarray(live["rotation"], dtype=float)
        arrays[f"{camera}/translation"] = np.asarray(live["translation"], dtype=float)
        arrays[f"{camera}/ground_height"] = np.asarray(live["ground_height"], dtype=float)
        counts[camera] = len(points[camera])
    source = {
        "cameras": sorted(points),
        "points": counts,
        "gt_files": len(gt_files),
        "gt_dirs": sorted({str(Path(f).parents[2]) for f in gt_files}),
        "fingerprint_geometry": str(geometry_path),
        "classes": sorted(MISSION_OBJECT_TYPES),
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, source=np.asarray(json.dumps(source)), **arrays)
    return source


def check(model_points: dict[str, np.ndarray], files: list[str]) -> list[str]:
    """Ground-height error on another set's objects: flat plane vs surface, per camera."""
    other, cameras = ground_points(files)
    lines = ["| camera | n | flat median abs dz | surface median abs dz | flat p90 | surface p90 |",
             "|---|---|---|---|---|---|"]
    for camera in sorted(other):
        if camera not in model_points:
            continue
        surface = GroundSurface(model_points[camera])
        pts = other[camera]
        # Ground objects only: the flat plane is the reference the check is about,
        # so keep points within 1 m of it in either direction.
        flat = float(cameras[camera]["ground_height"])
        pts = pts[np.abs(pts[:, 2] - flat) < 1.0]
        flat_err = np.abs(pts[:, 2] - flat)
        heights = np.array([surface.z(x, y) for x, y, _ in pts], dtype=object)
        inside = np.array([h is not None for h in heights])
        pts, heights = pts[inside], heights[inside].astype(float)
        flat_err = np.abs(pts[:, 2] - flat)
        surf_err = np.abs(pts[:, 2] - heights)
        lines.append(
            f"| {camera} | {len(pts)} | {np.median(flat_err):.3f} | {np.median(surf_err):.3f} "
            f"| {np.percentile(flat_err, 90):.3f} | {np.percentile(surf_err, 90):.3f} |"
        )
    return lines


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gt", required=True, help="glob of training-side generator GT JSONs")
    parser.add_argument("--check", help="glob of another GT set to report the height error on")
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--out", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args(argv)
    files = _files(args.gt)
    if not files:
        raise SystemExit(f"no GT files match {args.gt}")
    source = build(files, args.geometry, args.out)
    print(json.dumps(source, indent=1))
    if args.check:
        points, _ = ground_points(files)
        print("\n".join(check(points, _files(args.check))))


if __name__ == "__main__":
    main()
