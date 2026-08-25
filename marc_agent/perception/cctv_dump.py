"""Optional dump of fetched CCTV frames to PNG for local inspection."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from marc_agent.types import CctvDetectionRound

log = logging.getLogger("marc_agent.cctv_dump")

_DEFAULT_DIR = "artifacts/cctv_dump"


def _enabled() -> bool:
    raw = (os.environ.get("MARC_DUMP_CCTV") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def maybe_dump_cctv_round(round_: CctvDetectionRound, *, round_num: int) -> Path | None:
    """Save decoded CCTV frames when ``MARC_DUMP_CCTV=1``.

    Writes ``{MARC_DUMP_CCTV_DIR}/round_NNN/{camera_id}.png`` (default dir:
    ``artifacts/cctv_dump``). Returns the round directory, or None if disabled.
    """
    if not _enabled() or not round_.frames:
        return None

    root = Path(os.environ.get("MARC_DUMP_CCTV_DIR") or _DEFAULT_DIR)
    out_dir = root / f"round_{int(round_num):03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image
    except ImportError:
        log.warning("[cctv-dump] PIL not installed; cannot save frames")
        return None

    saved = 0
    for camera_id, frame in round_.frames.items():
        path = out_dir / f"{camera_id}.png"
        Image.fromarray(frame.rgb, mode="RGB").save(path)
        saved += 1

    log.info("[cctv-dump] saved %d frames -> %s", saved, out_dir.resolve())
    return out_dir
