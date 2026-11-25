"""Capture every CCTV camera's geometry through the SDK, once (#87).

Run in the frozen agent image against a live platform, with the agent's ROS
environment (``MARC_TEAM_ID``/``MARC_TOKEN``, host network, ``ROS_DOMAIN_ID``)::

    docker compose -p eval-agent -f docker-compose.yml \\
      -f scripts/frozen-logging.compose.yml run --rm --no-deps \\
      -v $REPO/scripts/frozen_eval:/work/frozen_eval:ro -v $OUT_DIR:/out \\
      --entrypoint python3 agent /work/frozen_eval/capture_sdk_geometry.py /out/sdk_cameras.json

Uses the runtime's own lookup (``Tf2ClientLookup``: CameraInfo K, ground height,
tf2 ``world <- camera``), so the result is exactly what the projector uses in a
round. Standalone: only the image's ``marc_agent``/``marc_sdk``.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone

from marc_agent.perception.cctv_project import Tf2ClientLookup
from marc_sdk import MARCClient

DISCOVERY_TIMEOUT_S = 120.0
SETTLE_S = 5.0


def main(out_path: str) -> int:
    client = MARCClient(node_name="marc_geometry_capture")
    if not client.connect(timeout=60.0):
        print("[sdk-geometry] connect failed", flush=True)
        return 1
    try:
        deadline = time.monotonic() + DISCOVERY_TIMEOUT_S
        cameras: list[str] = []
        while time.monotonic() < deadline:
            discover = getattr(client, "_discover_cctv", None)
            if callable(discover):
                discover()
            found = sorted(client.list_cctv())
            if found and found == cameras:
                break
            cameras = found
            time.sleep(SETTLE_S if found else 1.0)
        lookup = Tf2ClientLookup(client, timeout_s=10.0)
        geometry = {}
        for camera_id in cameras:
            for _ in range(10):
                result = lookup(camera_id)
                if result is not None:
                    geometry[camera_id] = asdict(result)
                    break
                time.sleep(1.0)
        payload = {
            "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "Tf2ClientLookup (CameraInfo.k, get_cctv_ground_height, tf2 world<-camera)",
            "cameras_listed": cameras,
            "geometry": geometry,
        }
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1)
        print(f"[sdk-geometry] {len(geometry)}/{len(cameras)} cameras -> {out_path}", flush=True)
        return 0 if cameras and len(geometry) == len(cameras) else 2
    finally:
        client.shutdown()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
