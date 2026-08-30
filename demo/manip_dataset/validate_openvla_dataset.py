#!/usr/bin/env python3
"""Validate MARC OpenVLA source data and optionally its materialized loaders."""

from __future__ import annotations

import argparse

from .openvla_dataset import (
    DEFAULT_CAMERA,
    validate_export,
    validate_openvla_loader,
    validate_tfds,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True, help="to_rlds.py output directory")
    parser.add_argument(
        "--camera",
        default=DEFAULT_CAMERA,
        help="required primary camera, or 'any' for a deliberate mixed export",
    )
    parser.add_argument("--tfds-root", help="materialized TFDS root to smoke test")
    parser.add_argument("--openvla-dir", help="official OpenVLA checkout to smoke test")
    args = parser.parse_args()
    if args.openvla_dir and not args.tfds_root:
        parser.error("--openvla-dir requires --tfds-root")

    camera = None if args.camera == "any" else args.camera
    report = validate_export(args.export, expected_camera=camera)
    print(
        f"[valid] source: {report.episodes} episodes, "
        f"{report.transitions} transitions, camera={report.camera}"
    )
    if args.tfds_root:
        tfds_report = validate_tfds(args.tfds_root)
        print(
            f"[valid] TFDS: sampled {tfds_report.transitions} transitions "
            f"from one episode"
        )
    if args.openvla_dir:
        validate_openvla_loader(args.openvla_dir, args.tfds_root)
        print("[valid] OpenVLA: official RLDSDataset loaded image/action/language")


if __name__ == "__main__":
    main()
