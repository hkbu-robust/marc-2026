#!/usr/bin/env python3
# Copyright (c) 2026, IoT Convergence & Open Sharing System (IoTCOSS)
#
# Persist each detection's world xyz in dataset-gen GT JSON so a Qwen2-VL
# converter can supervise GroundingResult.target_coord / anchor_coord.
# ---------------------------------------------------------------------------
# Stock _read_detections keeps prim path only as an internal `_prim_path` that
# _verify_detections pops before save_offline. This patch snapshots the prim
# world translation onto `world_xyz` at parse time.
#
# The helper MUST be inserted at module level, before `class GroundTruthGenerator`.
# Inserting it before `_read_detections` (inside the class) ends the class in
# Python, so `_read_detections` is no longer a method and capture() crashes.
import os
import sys

TARGET = os.environ.get(
    "MARC_GT_XYZ_TARGET",
    "/metacom2026/simulation_app/marc2026/trainer/ground_truth.py",
)
MARKER = "_marc_world_xyz"
CLASS_ANCHOR = "\nclass GroundTruthGenerator"
READ_ANCHOR = "\n    def _read_detections("

HELPER = '''
def _marc_world_xyz(prim_path):
    """[marc-patch] World translation of a detection prim, or None."""
    if not prim_path:
        return None
    try:
        import omni.usd
        from pxr import Usd, UsdGeom
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None
        t = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()).ExtractTranslation()
        return [round(float(t[0]), 4), round(float(t[1]), 4), round(float(t[2]), 4)]
    except Exception:
        return None

'''

OLD_APPEND = '''            detections.append({
                "class": cls,
                "instance_id": self._instance_id(prim_path),
                "kind": self._kind_from_path(prim_path),
                "bbox": [x_min, y_min, x_max, y_max],
                "occlusion": round(occlusion, 4),
                # 재투영 검증용 내부 필드 — _verify_detections 에서 제거된다.
                "_prim_path": prim_path,
            })
'''

XYZ_ONLY_APPEND = '''            det = {
                "class": cls,
                "instance_id": self._instance_id(prim_path),
                "kind": self._kind_from_path(prim_path),
                "bbox": [x_min, y_min, x_max, y_max],
                "occlusion": round(occlusion, 4),
                # 재투영 검증용 내부 필드 — _verify_detections 에서 제거된다.
                "_prim_path": prim_path,
            }
            xyz = _marc_world_xyz(prim_path)
            if xyz is not None:
                det["world_xyz"] = xyz
            detections.append(det)
'''


NEW_APPEND = '''            det = {
                "class": cls,
                "instance_id": self._instance_id(prim_path),
                "kind": self._kind_from_path(prim_path),
                "bbox": [x_min, y_min, x_max, y_max],
                "occlusion": round(occlusion, 4),
                # 재투영 검증용 내부 필드 — _verify_detections 에서 제거된다.
                "_prim_path": prim_path,
            }
            xyz = _marc_world_xyz(prim_path)
            if xyz is not None:
                det["world_xyz"] = xyz
            if str(cls or "").startswith("person_"):
                try:
                    import json as _json_rec
                    _rec = _json_rec.load(open("/tmp/marc_sar_recipe.json"))
                    if _rec.get("target_pose") == cls:
                        for _k in ("recipe_id", "support_class", "surface_class"):
                            if _rec.get(_k):
                                det[_k] = _rec[_k]
                except Exception:
                    pass
            detections.append(det)
'''


def _is_correct(src):
    """Helper is module-level (before the class) and _read_detections is still a method."""
    i_helper = src.find("def _marc_world_xyz(")
    i_class = src.find(CLASS_ANCHOR)
    i_read = src.find(READ_ANCHOR)
    if i_helper < 0 or i_class < 0 or i_read < 0:
        return False
    return i_helper < i_class < i_read and NEW_APPEND in src


def main():
    if not os.path.exists(TARGET):
        print(f"[gt-xyz-patch] source not found, skipping: {TARGET}")
        return 0

    with open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()

    if _is_correct(src):
        print("[gt-xyz-patch] already applied, skipping")
        return 0

    original = src

    # A previous version stuffed HELPER inside the class and ended it. Strip
    # that copy so _read_detections becomes a method again, then re-insert
    # at module level.
    if HELPER in src:
        src = src.replace(HELPER, "")
        print("[gt-xyz-patch] removed in-class helper (restores _read_detections)")

    if "def _marc_world_xyz(" not in src:
        insert_at = src.find(CLASS_ANCHOR)
        if insert_at < 0:
            print("[gt-xyz-patch] class GroundTruthGenerator not found, skipping")
            return 0
        src = src[:insert_at] + HELPER + src[insert_at:]

    if OLD_APPEND in src:
        src = src.replace(OLD_APPEND, NEW_APPEND, 1)
    elif XYZ_ONLY_APPEND in src:
        src = src.replace(XYZ_ONLY_APPEND, NEW_APPEND, 1)
    elif NEW_APPEND not in src:
        print("[gt-xyz-patch] detection-append anchor not found, skipping")
        return 0

    if src == original:
        print("[gt-xyz-patch] no change")
        return 0

    if not _is_correct(src):
        print("[gt-xyz-patch] patched source failed sanity check, skipping write")
        return 0

    try:
        with open(TARGET, "w", encoding="utf-8") as f:
            f.write(src)
    except OSError as e:
        print(f"[gt-xyz-patch] write failed, skipping: {e}")
        return 0

    print("[gt-xyz-patch] detections now include world_xyz and SAR recipe fields")
    return 0


if __name__ == "__main__":
    sys.exit(main())
