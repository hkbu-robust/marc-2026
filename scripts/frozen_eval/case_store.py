"""On-disk frozen evaluation cases, the checksummed manifest, and the loader (#87).

Layout under the sets root (``05-frozen-eval-sets/``)::

    heldout/cases/<case_id>/case.json, frames/<camera>.png
    public/cases/<case_id>/case.json, frames/<camera>.png
    manifest.json   # SHA-256 of every case file + counts; written once by ``freeze``

``case.json`` (schema 1) is the same for both sets:

- ``command``, ``parsed`` (runtime ``ParsedCommand`` as a dict), ``kind`` (object|sar);
- ``frames``: camera -> PNG path relative to the case folder (all cameras of the round);
- ``candidates``: the deployed detector's full list, runtime record format plus
  ``source`` (detector|refine|proxy);
- ``gold``: ``target_ids`` / ``anchor_ids`` (candidate ID sets), ``camera_ids``,
  ``target_xyz`` / ``anchor_xyz``, ``target_class``, ``landmark_class``,
  ``relation``, ``recipe_id``;
- ``tags``: strata and the per-case flags counted in the manifest.

Held-out cases may also carry a second candidate version from the final system's
detector code (``refresh_final.py``): ``candidates_final``, ``gold_final``
(target/anchor ID sets), ``tags_final`` and the final reader's ``parsed_final``.
``load_set(..., version="final")`` returns cases with those candidates, gold IDs and
miss flags in place of version 1; ``Case.parsed_final`` is set either way.

Freeze before scoring: ``freeze`` refuses to overwrite a manifest, and
``load_set`` fails on any missing, extra or changed file.

    python -m scripts.frozen_eval.case_store freeze ROOT
    python -m scripts.frozen_eval.case_store verify ROOT
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
SETS = ("heldout", "public")
MANIFEST = "manifest.json"


@dataclass(frozen=True)
class Case:
    """One frozen case. ``frame(camera)`` loads the RGB frame on demand."""

    case_id: str
    set: str
    kind: str
    command: str
    parsed: dict
    frames: dict[str, Path]
    candidates: list[dict]
    gold: dict
    tags: dict
    source: dict = field(default_factory=dict)
    parsed_final: dict | None = None
    version: str = "v1"

    def frame(self, camera_id: str) -> np.ndarray:
        from PIL import Image

        return np.asarray(Image.open(self.frames[camera_id]).convert("RGB"))


def write_case(root: Path, set_name: str, case: dict, frame_sources: dict[str, Path]) -> Path:
    """Write one case folder; frames are copied so the set never depends on its sources."""
    case_dir = root / set_name / "cases" / case["case_id"]
    (case_dir / "frames").mkdir(parents=True, exist_ok=False)
    frames = {}
    for camera_id, src in sorted(frame_sources.items()):
        rel = Path("frames") / f"{camera_id}.png"
        shutil.copyfile(src, case_dir / rel)
        frames[camera_id] = str(rel)
    payload = {"schema_version": SCHEMA_VERSION, "set": set_name, **case, "frames": frames}
    (case_dir / "case.json").write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    return case_dir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _case_files(root: Path, set_name: str) -> list[Path]:
    return sorted(p for p in (root / set_name / "cases").rglob("*") if p.is_file())


VERSIONS = ("v1", "final")


def _read_case(case_dir: Path, version: str = "v1") -> Case:
    raw = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{case_dir}: schema {raw.get('schema_version')} != {SCHEMA_VERSION}")
    if version not in VERSIONS:
        raise ValueError(f"version {version!r} not in {VERSIONS}")
    candidates, gold, tags = raw["candidates"], raw["gold"], raw["tags"]
    if version == "final":
        if "candidates_final" not in raw:
            raise ValueError(f"{case_dir}: no final candidate version (refresh_final.py)")
        candidates = raw["candidates_final"]
        gold = {**gold, **raw["gold_final"]}
        tags = {**tags, **raw["tags_final"]}
    return Case(
        case_id=raw["case_id"], set=raw["set"], kind=raw["kind"], command=raw["command"],
        parsed=raw["parsed"], frames={cam: case_dir / rel for cam, rel in raw["frames"].items()},
        candidates=candidates, gold=gold, tags=tags,
        source=raw.get("source") or {}, parsed_final=raw.get("parsed_final"), version=version,
    )


def read_cases(root: Path, set_name: str, version: str = "v1") -> list[Case]:
    """All cases of one set, without checksum verification (for building the manifest)."""
    cases_dir = root / set_name / "cases"
    if not cases_dir.is_dir():
        return []
    return [_read_case(d, version) for d in sorted(cases_dir.iterdir()) if d.is_dir()]


def counts(cases: list[Case]) -> dict:
    """Manifest counts: per relation x distractor, recipe, stratum, camera, and flags."""
    def tally(key) -> dict:
        return dict(sorted(collections.Counter(key(c) for c in cases).items(), key=str))

    return {
        "cases": len(cases),
        "kind": tally(lambda c: c.kind),
        "relation_x_distractor": tally(
            lambda c: f"{c.gold.get('relation')}|"
                      f"{'distractor_present' if c.tags.get('distractor_present') else 'distractor_absent'}"),
        "recipe": tally(lambda c: c.gold.get("recipe_id")),
        "stratum": tally(lambda c: c.tags.get("stratum")),
        "absent_from_original_sft": tally(lambda c: c.tags.get("absent_from_original_sft")),
        "gold_camera": tally(lambda c: ",".join(c.gold.get("camera_ids") or []) or None),
        "target_class": tally(lambda c: c.gold.get("target_class")),
        "detector_miss": tally(lambda c: c.tags.get("detector_miss")),
        "anchor_miss": tally(lambda c: c.tags.get("anchor_miss")),
        "with_proxy_candidates": tally(lambda c: c.tags.get("proxy_count", 0) > 0),
    }


def case_tags(case: Case) -> dict:
    """Tags #89 filters on: on/above relations (a known label artefact) and, for
    held-out cases, the generator scene's parity (the footprint prior was fitted on even scenes)."""
    tags = {"on_above": case.gold.get("relation") in ("on", "above")}
    gt_json = (case.source or {}).get("gt_json")
    match = re.search(r"scene_(\d+)", gt_json or "")
    if match:
        tags["scene_parity"] = "even" if int(match.group(1)) % 2 == 0 else "odd"
    return tags


def freeze(root: Path, notes: dict | None = None, version: str = "final") -> Path:
    """Write ``manifest.json`` once: per-file SHA-256, counts and per-case tags for every set.

    Counts use ``version`` (the frozen candidate version); v1 counts are kept for reference."""
    path = root / MANIFEST
    if path.exists():
        raise FileExistsError(f"{path} exists; frozen sets must not be re-frozen")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": notes or {},
        "sets": {},
    }
    for set_name in SETS:
        files = _case_files(root, set_name)
        if not files:
            continue
        cases = read_cases(root, set_name, version)
        unconfirmed = [c.case_id for c in cases
                       if set_name == "public" and not c.gold.get("confirmed_by")]
        if unconfirmed:
            raise ValueError(f"{len(unconfirmed)} public gold IDs unconfirmed, "
                             f"e.g. {unconfirmed[:3]} (public.py auto-confirm / confirm)")
        tags = {c.case_id: case_tags(c) for c in cases}
        manifest["sets"][set_name] = {
            "version": version,
            "counts": {**counts(cases),
                       "on_above": dict(collections.Counter(t["on_above"] for t in tags.values())),
                       "scene_parity": dict(collections.Counter(
                           t.get("scene_parity") for t in tags.values()))},
            "counts_v1": counts(read_cases(root, set_name, "v1")) if version != "v1" else None,
            "case_tags": tags,
            "files": {str(p.relative_to(root)): _sha256(p) for p in files},
        }
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return path


def verify(root: Path, set_name: str) -> None:
    """Raise unless every file of the set matches the manifest, with nothing extra."""
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    expected = manifest["sets"][set_name]["files"]
    actual = {str(p.relative_to(root)) for p in _case_files(root, set_name)}
    missing, extra = set(expected) - actual, actual - set(expected)
    if missing or extra:
        raise ValueError(f"{set_name}: {len(missing)} missing, {len(extra)} extra files "
                         f"(e.g. {sorted(missing | extra)[:3]})")
    changed = [rel for rel, sha in expected.items() if _sha256(root / rel) != sha]
    if changed:
        raise ValueError(f"{set_name}: {len(changed)} changed files (e.g. {changed[:3]})")


def load_set(root: Path, set_name: str, *, verify_checksums: bool = True,
             version: str = "v1") -> list[Case]:
    """The frozen cases of one set. Every selector arm loads its inputs through this.

    ``version="final"`` selects the final system's candidates and gold IDs (#87 refresh)."""
    root = Path(root)
    if verify_checksums:
        verify(root, set_name)
    return read_cases(root, set_name, version)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Freeze or verify the frozen evaluation sets.")
    parser.add_argument("action", choices=("freeze", "verify"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--notes", type=Path, help="JSON file merged into manifest.notes")
    parser.add_argument("--version", choices=VERSIONS, default="final",
                        help="candidate version the manifest counts describe (default final)")
    args = parser.parse_args(argv)
    if args.action == "freeze":
        notes = json.loads(args.notes.read_text()) if args.notes else None
        print(freeze(args.root, notes, args.version))
    for set_name in SETS:
        if (args.root / set_name / "cases").is_dir():
            cases = load_set(args.root, set_name)
            print(f"{set_name}: {len(cases)} cases verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
