"""What the deployed Qwen adapter was trained on, from the text-only Arrow caches (#87).

    python -m scripts.frozen_eval.training_record --out training_record.json

Needs ``pyarrow`` (not in the ``marc`` env; use any venv with pyarrow + pyyaml).

The deployed adapter ``qwen3vl-balanced-v3-final`` has a three-stage chain (#84):
original 5,535 -> corrective 280 -> balanced-v3 595 training rows. The renders and
JSONL files are gone; only the LLaMA-Factory Arrow caches remain. Each row keeps
its image paths and prompt, so we recover:

- the stage, scene, camera, row kind (``sar``/``obj``) and target box from the
  image name ``s{scene}_{camera}_{kind}_{x0}_{y0}_{x1}_{y1}_{sha}_{i}.png``;
- the recipe (SAR) from the ``Original request:`` line, matched word for word to
  ``sar_recipes.yaml`` commands; paraphrases stay ``recipe=None``;
- the target class, landmark and relation (objects) by parsing the request
  with the runtime command parser.

World positions are not stored, so the leakage check back-projects target boxes
with the rig geometry instead (see ``heldout.py``). Validation splits are listed
too: they were used for checkpoint selection, so they count for leakage but not
for training counts.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import yaml

from marc_agent.command_parser import CommandParseError, parse_command

ROOT = Path(__file__).resolve().parents[2]
RECIPES = ROOT / "simulation-platform" / "patches" / "sar_recipes.yaml"
CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "datasets" / "json"

# Arrow cache hash -> (stage, split). The deployed chain is original -> corrective -> v3;
# the intermediate ``balanced`` mix (0504e960/16c491e6) is not in the chain.
CACHES = {
    "fed7123b58687d4f": ("original", "train"),
    "95cde05a543cbcff": ("original", "val"),
    "a574f9a4b2af09d8": ("corrective", "train"),
    "ede538b839304852": ("corrective", "val"),
    "2e76f7f1bc631b3b": ("balanced-v3", "train"),
    "76a9f87fcd4eb2ab": ("balanced-v3", "val"),
}

IMAGE_RE = re.compile(
    r"(?:^|/)(?:s|sbundle_)(\d+)_(rig_\d+_[a-z])_(obj|sar)_"
    r"(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)_[0-9a-f]{10}_\d+\.png$"
)
REQUEST_RE = re.compile(r"Original request: (.*)")


def _arrow_rows(cache_hash: str) -> list[dict]:
    import pyarrow as pa

    paths = [
        p for p in CACHE_ROOT.glob(f"default-{cache_hash}/**/json-*.arrow")
        if not p.name.startswith("cache-")
    ]
    if len(paths) != 1:
        raise FileNotFoundError(f"expected one base Arrow file for {cache_hash}, got {paths}")
    return pa.ipc.open_stream(pa.memory_map(str(paths[0]))).read_all().to_pylist()


def row_record(images: list[str], prompt: str, recipe_by_command: dict[str, str]) -> dict:
    """One training row -> {scene, camera, kind, bbox, request, recipe, target_type, landmark, relation}."""
    match = IMAGE_RE.search(images[0])
    if match is None:
        raise ValueError(f"unrecognised training image name: {images[0]}")
    scene, camera, kind = int(match.group(1)), match.group(2), match.group(3)
    bbox = [int(match.group(i)) for i in range(4, 8)]
    req_match = REQUEST_RE.search(prompt)
    request = req_match.group(1).strip() if req_match else None
    record = {"scene": scene, "camera": camera, "kind": kind, "bbox": bbox,
              "request": request, "recipe": None, "target_type": None, "landmark": None,
              "relation": None}
    if kind == "sar":
        record["recipe"] = recipe_by_command.get(request or "")
    elif request:
        try:
            parsed = parse_command(request)
        except CommandParseError:
            return record
        record["target_type"] = parsed.target_type
        record["landmark"] = parsed.landmark
        record["relation"] = parsed.relation.value if parsed.relation else None
    return record


def build_record() -> dict:
    recipes = yaml.safe_load(RECIPES.read_text(encoding="utf-8"))["recipes"]
    recipe_by_command = {r["command"].strip(): r["id"] for r in recipes}
    rows = []
    for cache_hash, (stage, split) in CACHES.items():
        for row in _arrow_rows(cache_hash):
            record = row_record(row["images"], row["messages"][0]["content"], recipe_by_command)
            rows.append({"stage": stage, "split": split, **record})

    train = [r for r in rows if r["split"] == "train"]
    original_train = [r for r in train if r["stage"] == "original"]
    return {
        "source": "Arrow caches of the deployed adapter chain (#84)",
        "caches": {h: {"stage": s, "split": p} for h, (s, p) in CACHES.items()},
        "train_counts": {
            "recipe": collections.Counter(r["recipe"] for r in train if r["kind"] == "sar"),
            "relation": collections.Counter(r["relation"] for r in train if r["kind"] == "obj"),
        },
        "original_train_counts": {
            "recipe": collections.Counter(r["recipe"] for r in original_train if r["kind"] == "sar"),
            "relation": collections.Counter(
                r["relation"] for r in original_train if r["kind"] == "obj"),
        },
        "unmatched_sar_requests": collections.Counter(
            r["request"] for r in train if r["kind"] == "sar" and r["recipe"] is None),
        "rows": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    record = build_record()
    args.out.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("train_counts", "original_train_counts")},
                     indent=1, default=str))
    print(f"{len(record['rows'])} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
