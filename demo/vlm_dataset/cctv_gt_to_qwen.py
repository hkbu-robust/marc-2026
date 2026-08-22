#!/usr/bin/env python3
"""Convert CCTV dataset-gen GT (PNG + JSON) into Qwen2-VL instruction rows.

Does not run Isaac. Reads trainer_output/<scenario>/scene_XXXXX/{camera}.png|.json
and writes LLaMA-Factory / official Qwen2-VL JSONL.

Each row is one referring-expression grounded in a GroundingResult payload
(camera_id, target_type, landmark, relation, target_coord, anchor_coord).
SAR (person) rows use target_type "person" and interpretation.situation
(accident / emergency / abnormal / normal). person_sitting is always normal.

When several instances of the same class are in frame, the landmark is what
disambiguates — that is the VLM signal YOLO does not need.

Usage (from repo root or demo/):

  python3 demo/vlm_dataset/cctv_gt_to_qwen.py \\
    --in trainer_output/marc2026_chungmu \\
    --out trainer_output/qwen2vl_cctv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

SCENE_RE = re.compile(r"^scene_(\d+)$")

# Spoken names in MARC voice_command text.
DISPLAY = {
    "cracker_box": "cracker box",
    "cola_can": "cola can",
    "disposable_cup": "disposable cup",
    "pencilcase": "pencil case",
    "picnic_table": "picnic table",
    "vending_machine": "vending machine",
    "trash_can": "trash can",
    "kick_scooter": "kick scooter",
    "police_car": "police car",
    "normal_car_1": "car",
    "normal_car_2": "car",
    "sports_car": "sports car",
    "sports_car_2": "sports car",
}

TEMPLATES = {
    "beside": (
        "Find the {obj} beside the {lm}.",
        "Find the {obj} next to the {lm}.",
        "Locate the {obj} beside the {lm}.",
    ),
    "near": (
        "Find the {obj} near the {lm}.",
        "Find the {obj} close to the {lm}.",
        "Locate the {obj} near the {lm}.",
    ),
}

# Skip people / pose types in the lost-object family — SAR is a second row type.
_PERSON_PREFIXES = ("person_", "male_", "female_", "professor_", "student_")

# Spoken pose fragments for SAR prompts (GT class is the pose, not identity).
_SAR_POSE_SPEAK = {
    "person_prone": "lying face-down",
    "person_lying_down": "lying down",
    "person_bending": "bent over",
    "person_crouching": "crouched",
    "person_reclining": "reclining",
    "person_sitting": "sitting",
    "person_reaching": "reaching",
    "person_walking": "walking",
    "person_walking_away": "walking away",
}

# Heuristic pose -> situation. person_sitting is always normal.
_SAR_SITUATION = {
    "person_prone": "accident",
    "person_lying_down": "accident",
    "person_bending": "emergency",
    "person_crouching": "emergency",
    "person_reclining": "abnormal",
    "person_sitting": "normal",
    "person_reaching": "normal",
    "person_walking": "normal",
    "person_walking_away": "normal",
}

SAR_TEMPLATES = (
    "Find the person {pose} near the {lm}.",
    "Find the person {pose} beside the {lm}.",
    "Locate the person {pose} near the {lm}.",
)


def _display(name: str) -> str:
    if name in DISPLAY:
        return DISPLAY[name]
    return str(name).replace("_", " ")


def _is_person(cls: str) -> bool:
    c = (cls or "").lower()
    return c == "person" or c.startswith("person_") or any(c.startswith(p) for p in _PERSON_PREFIXES)


def _is_umbrella(cls: str) -> bool:
    return "umbrella" in (cls or "").lower()


def _sar_pose_speak(cls: str) -> str:
    c = (cls or "").lower()
    if c in _SAR_POSE_SPEAK:
        return _SAR_POSE_SPEAK[c]
    if c.startswith("person_"):
        return c.replace("person_", "").replace("_", " ")
    return "in distress"


def _sar_situation(cls: str) -> str:
    c = (cls or "").lower()
    if c == "person_sitting" or c.startswith("person_sitting"):
        return "normal"
    return _SAR_SITUATION.get(c, "abnormal")


def _bbox_center(bbox) -> Optional[Tuple[float, float]]:
    if not bbox or len(bbox) < 4:
        return None
    x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)


def _matmul(a: List[List[float]], v: List[float]) -> List[float]:
    return [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]


def _euler_zyx_matrix(roll: float, pitch: float, yaw: float) -> List[List[float]]:
    """Match camera_projection: R.from_euler('zyx', [yaw, pitch, roll])."""
    r, p, y = (math.radians(roll), math.radians(pitch), math.radians(yaw))
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rz = [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]]
    ry = [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]]
    rx = [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]]
    # Rz @ Ry
    zy = [
        [rz[i][0] * ry[0][j] + rz[i][1] * ry[1][j] + rz[i][2] * ry[2][j] for j in range(3)]
        for i in range(3)
    ]
    return [
        [zy[i][0] * rx[0][j] + zy[i][1] * rx[1][j] + zy[i][2] * rx[2][j] for j in range(3)]
        for i in range(3)
    ]


def _ground_xyz(payload: dict, u: float, v: float) -> Optional[List[float]]:
    """Back-project a pixel onto the ground plane using GT camera extrinsics."""
    cam = payload.get("camera") or {}
    img = payload.get("image") or {}
    pos = cam.get("position")
    euler = cam.get("orientation_euler_deg") or {}
    if not pos or len(pos) < 3:
        return None
    width = float(img.get("width") or 0.0)
    height = float(img.get("height") or 0.0)
    if width <= 1 or height <= 1:
        return None
    f = float(cam.get("focal_length_mm") or 0.0)
    ha = float(cam.get("horizontal_aperture_mm") or 0.0)
    va = float(cam.get("vertical_aperture_mm") or 0.0)
    if f <= 0 or ha <= 0:
        return None
    if va <= 0:
        va = ha * height / width
    gz = cam.get("ground_height")
    ground_z = float(gz if gz is not None else pos[2] - 8.0)
    # USD cam: +X right, +Y up, -Z forward. Image y is down.
    x = (u / width - 0.5) * ha
    y = (0.5 - v / height) * va
    z = -f
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    dir_cam = [x / n, y / n, z / n]
    rot = _euler_zyx_matrix(
        float(euler.get("roll") or 0.0),
        float(euler.get("pitch") or 0.0),
        float(euler.get("yaw") or 0.0),
    )
    d = _matmul(rot, dir_cam)
    if abs(d[2]) < 1e-6:
        return None
    t = (ground_z - float(pos[2])) / d[2]
    if t <= 0.05 or t > 80.0:
        return None
    return [
        round(float(pos[0]) + t * d[0], 4),
        round(float(pos[1]) + t * d[1], 4),
        round(ground_z, 4),
    ]


def _det_xyz(det: dict, payload: dict) -> Optional[List[float]]:
    xyz = det.get("world_xyz")
    if isinstance(xyz, (list, tuple)) and len(xyz) >= 3:
        return [round(float(xyz[0]), 4), round(float(xyz[1]), 4), round(float(xyz[2]), 4)]
    c = _bbox_center(det.get("bbox"))
    if c is None:
        return None
    return _ground_xyz(payload, c[0], c[1])


def _relation(obj_c: Tuple[float, float], lm_c: Tuple[float, float], width: float) -> str:
    dist = math.hypot(obj_c[0] - lm_c[0], obj_c[1] - lm_c[1])
    return "beside" if dist < 0.22 * max(width, 1.0) else "near"


def _coord3(c: Optional[List[float]]) -> List[float]:
    vals = [float(v) for v in (c or [])]
    while len(vals) < 3:
        vals.append(0.0)
    return [round(v, 4) for v in vals[:3]]


def _iter_scenes(root: str) -> Iterable[Tuple[int, str]]:
    try:
        names = os.listdir(root)
    except OSError:
        return
    found = []
    for name in names:
        m = SCENE_RE.match(name)
        path = os.path.join(root, name)
        if m and os.path.isdir(path):
            found.append((int(m.group(1)), path))
    for item in sorted(found):
        yield item


def _load_pair(scene_dir: str, stem: str) -> Optional[Tuple[str, dict]]:
    json_path = os.path.join(scene_dir, f"{stem}.json")
    png_path = os.path.join(scene_dir, f"{stem}.png")
    if not os.path.isfile(json_path) or not os.path.isfile(png_path):
        return None
    if stem.endswith("_overlay"):
        return None
    try:
        with open(json_path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return png_path, payload


def _pairs_for_payload(payload: dict, rng: random.Random, max_per_image: int):
    dets = list(payload.get("detections") or [])
    objects = [
        d for d in dets
        if (d.get("kind") or "object") != "landmark"
        and not _is_person(str(d.get("class") or ""))
        and not _is_umbrella(str(d.get("class") or ""))
        and d.get("class")
        and _bbox_center(d.get("bbox"))
    ]
    landmarks = [
        d for d in dets
        if (d.get("kind") == "landmark" or str(d.get("class") or "") in DISPLAY)
        and d.get("class")
        and not _is_umbrella(str(d.get("class") or ""))
        and _bbox_center(d.get("bbox"))
        and (d.get("kind") == "landmark" or str(d.get("class")) not in {
            o.get("class") for o in objects
        })
    ]
    # Prefer explicit kind=landmark; fall back to leftover non-object dets.
    if not landmarks:
        landmarks = [
            d for d in dets
            if d not in objects and d.get("class") and _bbox_center(d.get("bbox"))
            and not _is_person(str(d.get("class") or ""))
            and not _is_umbrella(str(d.get("class") or ""))
        ]
    if not objects or not landmarks:
        return []

    counts: Dict[str, int] = {}
    for o in objects:
        counts[o["class"]] = counts.get(o["class"], 0) + 1

    width = float((payload.get("image") or {}).get("width") or 1280)
    scored = []
    for obj in objects:
        oc = _bbox_center(obj["bbox"])
        best = None
        best_d = None
        for lm in landmarks:
            if lm is obj:
                continue
            if lm.get("class") == obj.get("class"):
                continue
            lc = _bbox_center(lm["bbox"])
            d = math.hypot(oc[0] - lc[0], oc[1] - lc[1])
            if best_d is None or d < best_d:
                best_d, best = d, lm
        if best is None:
            continue
        rel = _relation(oc, _bbox_center(best["bbox"]), width)
        hard = counts.get(obj["class"], 1) > 1
        scored.append((hard, rng.random(), obj, best, rel))

    scored.sort(key=lambda t: (not t[0], t[1]))  # hard (same-class) first
    picked = scored[: max(1, max_per_image)]
    rows = []
    for hard, _, obj, lm, rel in picked:
        tmpl = rng.choice(TEMPLATES[rel])
        instruction = tmpl.format(obj=_display(obj["class"]), lm=_display(lm["class"]))
        rows.append((instruction, obj, lm, rel, hard))
    return rows


def _payload_landmarks(dets, objects):
    object_classes = {o.get("class") for o in objects}
    landmarks = [
        d for d in dets
        if (d.get("kind") == "landmark" or str(d.get("class") or "") in DISPLAY)
        and d.get("class")
        and not _is_umbrella(str(d.get("class") or ""))
        and not _is_person(str(d.get("class") or ""))
        and _bbox_center(d.get("bbox"))
        and (d.get("kind") == "landmark" or str(d.get("class")) not in object_classes)
    ]
    if not landmarks:
        landmarks = [
            d for d in dets
            if d not in objects and d.get("class") and _bbox_center(d.get("bbox"))
            and not _is_person(str(d.get("class") or ""))
            and not _is_umbrella(str(d.get("class") or ""))
        ]
    return landmarks


def _sar_pairs_for_payload(payload: dict, rng: random.Random, max_per_image: int):
    """Person rows: target_type=person, situation from pose (sitting always normal)."""
    dets = list(payload.get("detections") or [])
    people = [
        d for d in dets
        if _is_person(str(d.get("class") or ""))
        and d.get("class")
        and _bbox_center(d.get("bbox"))
    ]
    if not people:
        return []
    objects = [
        d for d in dets
        if (d.get("kind") or "object") != "landmark"
        and not _is_person(str(d.get("class") or ""))
        and not _is_umbrella(str(d.get("class") or ""))
        and d.get("class")
    ]
    landmarks = _payload_landmarks(dets, objects)
    if not landmarks:
        return []

    n_people = len(people)
    scored = []
    for person in people:
        oc = _bbox_center(person["bbox"])
        best = None
        best_d = None
        for lm in landmarks:
            lc = _bbox_center(lm["bbox"])
            d = math.hypot(oc[0] - lc[0], oc[1] - lc[1])
            if best_d is None or d < best_d:
                best_d, best = d, lm
        if best is None:
            continue
        sit = _sar_situation(str(person.get("class") or ""))
        distress = sit != "normal"
        hard = n_people > 1
        scored.append((not distress, not hard, rng.random(), person, best, sit, hard))

    scored.sort()
    picked = scored[: max(1, max_per_image)]
    rows = []
    for _, _, _, person, lm, sit, hard in picked:
        tmpl = rng.choice(SAR_TEMPLATES)
        instruction = tmpl.format(
            pose=_sar_pose_speak(str(person.get("class") or "")),
            lm=_display(lm["class"]),
        )
        rows.append((instruction, person, lm, sit, hard))
    return rows


def _sample(png_path: str, payload: dict, camera_id: str, instruction: str,
            obj: dict, lm: dict, fmt: str, rel: Optional[str] = None,
            situation: Optional[str] = None, target_type: Optional[str] = None) -> dict:
    interp = {
        "target_type": target_type or obj["class"],
        "landmark": lm["class"] if lm else "",
    }
    if rel is not None:
        interp["relation"] = rel
    if situation is not None:
        interp["situation"] = situation
    answer = {
        "camera_id": camera_id,
        "interpretation": interp,
        "grounding": {
            "anchor_coord": _coord3(_det_xyz(lm, payload)) if lm else [0.0, 0.0, 0.0],
            "target_coord": _coord3(_det_xyz(obj, payload)),
        },
    }
    text = json.dumps(answer, ensure_ascii=False)
    image_rel = os.path.abspath(png_path)
    prompt = f"<image>\n{instruction}"
    if fmt == "sharegpt":
        return {
            "images": [image_rel],
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": text},
            ],
        }
    return {
        "images": [image_rel],
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": text},
        ],
    }


def convert(in_dir: str, out_dir: str, val_frac: float, max_per_image: int,
            seed: int, fmt: str, max_sar_per_image: int = 2) -> None:
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    scenes = list(_iter_scenes(in_dir))
    if not scenes:
        # Allow passing trainer_output/ (scenario subdirs) or a single scene root.
        nested = []
        try:
            for name in sorted(os.listdir(in_dir)):
                path = os.path.join(in_dir, name)
                if os.path.isdir(path):
                    nested.extend(_iter_scenes(path))
        except OSError:
            pass
        scenes = nested
    if not scenes:
        raise SystemExit(f"no scene_* directories under {in_dir}")

    n_cut = max(1, int(round(len(scenes) * (1.0 - val_frac)))) if len(scenes) > 1 else len(scenes)
    train_rows: List[dict] = []
    val_rows: List[dict] = []
    n_hard = 0
    n_skip = 0
    n_sar = 0
    sit_counts: Counter = Counter()
    lm_counts: Counter = Counter()
    obj_counts: Counter = Counter()

    for idx, (scene_id, scene_dir) in enumerate(scenes):
        stems = [
            os.path.splitext(n)[0]
            for n in os.listdir(scene_dir)
            if n.endswith(".json") and not n.endswith("_overlay.json")
        ]
        bucket = train_rows if idx < n_cut else val_rows
        for stem in sorted(stems):
            pair = _load_pair(scene_dir, stem)
            if pair is None:
                n_skip += 1
                continue
            png_path, payload = pair
            camera_id = str(payload.get("frame_id") or stem)
            obj_rows = _pairs_for_payload(payload, rng, max_per_image)
            sar_rows = _sar_pairs_for_payload(payload, rng, max_sar_per_image)
            if not obj_rows and not sar_rows:
                n_skip += 1
                continue
            for instruction, obj, lm, rel, hard in obj_rows:
                bucket.append(_sample(
                    png_path, payload, camera_id, instruction, obj, lm, fmt, rel=rel))
                lm_counts[str(lm.get("class") or "?")] += 1
                obj_counts[str(obj.get("class") or "?")] += 1
                if hard:
                    n_hard += 1
            for instruction, person, lm, sit, hard in sar_rows:
                bucket.append(_sample(
                    png_path, payload, camera_id, instruction, person, lm, fmt,
                    situation=sit, target_type="person"))
                lm_counts[str(lm.get("class") or "?")] += 1
                sit_counts[sit] += 1
                n_sar += 1
                if hard:
                    n_hard += 1

    def _write(name: str, rows: List[dict]) -> str:
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    train_path = _write("train.jsonl", train_rows)
    val_path = _write("val.jsonl", val_rows)
    n_total = len(train_rows) + len(val_rows)
    info = {
        "format": fmt,
        "source": os.path.abspath(in_dir),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "num_hard_same_class": n_hard,
        "num_sar": n_sar,
        "hard_frac": round(n_hard / n_total, 4) if n_total else 0.0,
        "skipped_cameras": n_skip,
        "max_per_image": max_per_image,
        "max_sar_per_image": max_sar_per_image,
        "landmark_class_counts": dict(lm_counts.most_common()),
        "object_class_counts": dict(obj_counts.most_common()),
        "sar_situation_counts": dict(sit_counts.most_common()),
        "assistant": "GroundingResult JSON (camera_id, interpretation, grounding)",
        "notes": (
            "Fine-tune Qwen2-VL-Instruct with these JSONL files. "
            "Object rows use relation; SAR rows use target_type=person and situation. "
            "person_sitting is always situation=normal. "
            "Do not mix ego-cam or OpenVLA pick frames into this set."
        ),
    }
    info_path = os.path.join(out_dir, "dataset_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"[qwen] train={len(train_rows)} val={len(val_rows)} "
          f"hard={n_hard} hard_frac={info['hard_frac']} sar={n_sar} skipped_cam={n_skip}")
    if lm_counts:
        lm_txt = ", ".join(f"{k}={v}" for k, v in lm_counts.most_common())
        print(f"[qwen] landmarks: {lm_txt}")
    if obj_counts:
        obj_txt = ", ".join(f"{k}={v}" for k, v in obj_counts.most_common())
        print(f"[qwen] objects: {obj_txt}")
    if sit_counts:
        sit_txt = ", ".join(f"{k}={v}" for k, v in sit_counts.most_common())
        print(f"[qwen] sar situations: {sit_txt}")
    print(f"[qwen] {train_path}")
    print(f"[qwen] {val_path}")
    print(f"[qwen] {info_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_dir", required=True,
                    help="dataset-gen output (scenario dir or trainer_output/)")
    ap.add_argument("--out", dest="out_dir", required=True,
                    help="JSONL output directory")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-per-image", type=int, default=4)
    ap.add_argument("--max-sar-per-image", type=int, default=2,
                    help="max SAR (person) rows per camera image")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--format", choices=("llama-factory", "sharegpt"),
                    default="llama-factory",
                    help="llama-factory = messages[]; sharegpt = conversations[]")
    args = ap.parse_args()
    if not os.path.isdir(args.in_dir):
        raise SystemExit(f"input dir not found: {args.in_dir}")
    convert(args.in_dir, args.out_dir, args.val_frac, args.max_per_image,
            args.seed, args.format, max_sar_per_image=args.max_sar_per_image)


if __name__ == "__main__":
    sys.exit(main())
