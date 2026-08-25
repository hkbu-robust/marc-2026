#!/usr/bin/env python3
"""Convert CCTV dataset-gen GT (PNG + JSON) into Ultralytics YOLO detect layout.

Does not run Isaac. Same scene dump as cctv_gt_to_qwen.py; split by scene_id so
the two exporters can share --val-frac / --seed.

  python3 demo/vlm_dataset/cctv_gt_to_yolo.py \\
    --in trainer_output/marc2026_chungmu \\
    --out trainer_output/yolo_cctv
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

from collections import Counter
from typing import List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from cctv_gt_to_qwen import _iter_scenes, _load_pair

# Stable class ids. Extra names found in GT are appended (sorted) after these.
CANONICAL_NAMES = [
    "tumbler", "cracker_box", "cola_can", "disposable_cup",
    "umbrella", "long_umbrella", "juice", "tissue", "mug", "sunblock", "pencilcase",
    "bench", "bicycle", "hydrant", "kick_scooter", "picnic_table",
    "postbox", "vending_machine", "trash_can",
    "police_car", "normal_car_1", "normal_car_2",
    "sports_car", "sports_car_2", "suv", "taxi",
    "person_walking", "person_reaching", "person_crouching", "person_prone",
    "person_sitting", "person_walking_away", "person_reclining",
    "person_bending", "person_lying_down",
]

MIN_BOX_PX = 4


def _collect_scenes(in_dir: str) -> List[Tuple[int, str]]:
    scenes = list(_iter_scenes(in_dir))
    if scenes:
        return scenes
    nested = []
    try:
        for name in sorted(os.listdir(in_dir)):
            path = os.path.join(in_dir, name)
            if os.path.isdir(path):
                nested.extend(_iter_scenes(path))
    except OSError:
        pass
    return nested


def _link_or_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        shutil.copy2(src, dst)


def _write_yolo_line(det: dict, names: List[str], width: float, height: float) -> Optional[str]:
    cls = det.get("class")
    if cls not in names:
        return None
    bbox = det.get("bbox")
    if not bbox or len(bbox) < 4:
        return None
    x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    x0 = max(0.0, min(width, x0))
    x1 = max(0.0, min(width, x1))
    y0 = max(0.0, min(height, y0))
    y1 = max(0.0, min(height, y1))
    bw, bh = x1 - x0, y1 - y0
    if bw < MIN_BOX_PX or bh < MIN_BOX_PX:
        return None
    cx = min(1.0, max(0.0, (x0 + x1) * 0.5 / width))
    cy = min(1.0, max(0.0, (y0 + y1) * 0.5 / height))
    nw = min(1.0, max(1e-6, bw / width))
    nh = min(1.0, max(1e-6, bh / height))
    return f"{names.index(cls)} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def convert(in_dir: str, out_dir: str, val_frac: float) -> None:
    scenes = _collect_scenes(in_dir)
    if not scenes:
        raise SystemExit(f"no scene_* directories under {in_dir}")

    names = list(CANONICAL_NAMES)
    extra = set()
    for _, scene_dir in scenes:
        for name in os.listdir(scene_dir):
            if not name.endswith(".json") or name.endswith("_overlay.json"):
                continue
            pair = _load_pair(scene_dir, os.path.splitext(name)[0])
            if pair is None:
                continue
            _, payload = pair
            for det in payload.get("detections") or []:
                cls = det.get("class")
                if cls and cls not in names:
                    extra.add(str(cls))
    names.extend(sorted(extra))

    n_cut = max(1, int(round(len(scenes) * (1.0 - val_frac)))) if len(scenes) > 1 else len(scenes)
    counts: Counter = Counter()
    n_img = {"train": 0, "val": 0}
    n_box = {"train": 0, "val": 0}

    for idx, (scene_id, scene_dir) in enumerate(scenes):
        split = "train" if idx < n_cut else "val"
        stems = [
            os.path.splitext(n)[0]
            for n in os.listdir(scene_dir)
            if n.endswith(".json") and not n.endswith("_overlay.json")
        ]
        for stem in sorted(stems):
            pair = _load_pair(scene_dir, stem)
            if pair is None:
                continue
            png_path, payload = pair
            img = payload.get("image") or {}
            width = float(img.get("width") or 1280)
            height = float(img.get("height") or 720)
            if width <= 1 or height <= 1:
                continue
            lines = []
            for det in payload.get("detections") or []:
                line = _write_yolo_line(det, names, width, height)
                if line is None:
                    continue
                lines.append(line)
                counts[str(det.get("class"))] += 1
            img_name = f"scene_{scene_id:05d}_{stem}.png"
            dst_img = os.path.join(out_dir, "images", split, img_name)
            dst_lbl = os.path.join(out_dir, "labels", split, img_name.replace(".png", ".txt"))
            _link_or_copy(png_path, dst_img)
            os.makedirs(os.path.dirname(dst_lbl), exist_ok=True)
            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            n_img[split] += 1
            n_box[split] += len(lines)

    yaml_path = os.path.join(out_dir, "data.yaml")
    os.makedirs(out_dir, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        for i, n in enumerate(names):
            f.write(f"  {i}: {n}\n")

    info = {
        "source": os.path.abspath(in_dir),
        "num_train_images": n_img["train"],
        "num_val_images": n_img["val"],
        "num_train_boxes": n_box["train"],
        "num_val_boxes": n_box["val"],
        "class_counts": dict(counts.most_common()),
        "names": names,
        "notes": (
            "Ultralytics detect layout. Canonical pose/object classes; no situation "
            "labels. Scene-grouped split; use the same --val-frac as "
            "cctv_gt_to_qwen.py. Do not mix ego-cam frames."
        ),
    }
    info_path = os.path.join(out_dir, "dataset_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"[yolo] train_img={n_img['train']} val_img={n_img['val']} "
          f"train_box={n_box['train']} val_box={n_box['val']}")
    if counts:
        print("[yolo] classes:", ", ".join(f"{k}={v}" for k, v in counts.most_common()))
    print(f"[yolo] {yaml_path}")
    print(f"[yolo] {info_path}")
    print("[yolo] train with: yolo detect train data=%s model=yolo11x.pt imgsz=1280" % yaml_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", dest="out_dir", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="match cctv_gt_to_qwen.py so splits share scenes")
    args = ap.parse_args()
    if not os.path.isdir(args.in_dir):
        raise SystemExit(f"input dir not found: {args.in_dir}")
    convert(args.in_dir, args.out_dir, args.val_frac)


if __name__ == "__main__":
    sys.exit(main())
