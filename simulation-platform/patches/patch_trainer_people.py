#!/usr/bin/env python3
# Copyright (c) 2026, IoT Convergence & Open Sharing System (IoTCOSS)
#
# MARC 2026 - source patch: all four person identities per dataset-gen scene
# ---------------------------------------------------------------------------
# Stock _build_deploy_lists person-mode is one radio character x every pose
# (defaults to male_student_01). Female / professor / student_weight never
# appear. This replaces the combo loop with one of each catalog identity and
# a random pose (4 people/scene). TRAINER_PEOPLE_ALL_CHARS=0 restores stock.
#
# Also stamps placement.on_landmark_type onto a fraction of people so the
# on-landmark deployer patch can seat them on benches / vending machines /
# trash cans (SAR "climbed on the drink machine"). TRAINER_PEOPLE_ON_LANDMARK_PROB
# (default 0.45) is per person; TRAINER_PEOPLE_WRONG_PLACE_PROB (default 0.55)
# of those prefer a non-seat landmark. TRAINER_FORCE_CLIMB_LANDMARK_PROB
# (default 0.35) injects vending_machine or trash_can when the scene has none.
#
# TRAINER_PEOPLE_ABOVE=1 constructs every scene with one random-pose climber
# on a guaranteed climb landmark. Extra climb types are
# dropped so the in-frame vending/trash/hydrant/postbox is the stacked one.
# Benches stay as clutter. Seats/vehicles are never climb targets.
# TRAINER_PEOPLE_CHARS and TRAINER_PEOPLE_POSES optionally restrict meshes
# and poses (comma lists; empty = catalog). Short names work (weight, sitting).
#
# boot.py runs this before dataset-gen (trainer.py). Idempotent via MARKER.
import os
import re
import sys

TARGET = "/metacom2026/simulation_app/marc2026/trainer.py"
MARKER = "_marc_people_above"

OLD_COMBOS = '''    combos = []
    if not people_state.get("enabled", True):
        # "Add people" 해제 → 사람 미배치.
        return landmarks, objects
    if people_state.get("mode") == "pose":
        pose = pose_by_key.get(people_state.get("pose"))
        if pose:
            combos = [(pose, ch) for ch in all_chars]
    else:  # person
        ch = char_by_key.get(people_state.get("person"))
        if ch:
            combos = [(pose, ch) for pose in all_poses]
    for pose, ch in combos:
        objects.append({"type": pose, "character": ch, "count": 1})
'''

V1_COMBOS = '''    combos = []
    if not people_state.get("enabled", True):
        # "Add people" 해제 → 사람 미배치.
        return landmarks, objects
    def _marc_people_all_chars():
        """[marc-patch] One of each identity, random pose (covers female/professor/weight)."""
        raw = os.environ.get("TRAINER_PEOPLE_ALL_CHARS", "1")
        return str(raw).strip().lower() not in ("0", "false", "no", "off")
    if _marc_people_all_chars() and all_chars and all_poses:
        for ch in all_chars:
            combos.append((random.choice(all_poses), ch))
    elif people_state.get("mode") == "pose":
        pose = pose_by_key.get(people_state.get("pose"))
        if pose:
            combos = [(pose, ch) for ch in all_chars]
    else:
        ch = char_by_key.get(people_state.get("person"))
        if ch:
            combos = [(pose, ch) for pose in all_poses]
    for pose, ch in combos:
        objects.append({"type": pose, "character": ch, "count": 1})
'''

V2_COMBOS = '''    combos = []
    if not people_state.get("enabled", True):
        # "Add people" 해제 → 사람 미배치.
        return landmarks, objects
    def _marc_people_all_chars():
        """[marc-patch] One of each identity, random pose (covers female/professor/weight)."""
        raw = os.environ.get("TRAINER_PEOPLE_ALL_CHARS", "1")
        return str(raw).strip().lower() not in ("0", "false", "no", "off")
    def _marc_frac(name, default):
        """[marc-patch] Clamp an env probability into [0, 1]."""
        try:
            v = float(os.environ.get(name, str(default)) or str(default))
        except (TypeError, ValueError):
            v = float(default)
        return max(0.0, min(1.0, v))
    def _marc_people_on_landmark():
        """[marc-patch] Marker: people may sit/climb on landmark tops."""
        return True
    def _marc_lm_typ(spec):
        import re as _re_lm
        return _re_lm.sub(r"_[0-9]+$", "", str((spec or {}).get("name") or ""))
    if _marc_people_all_chars() and all_chars and all_poses:
        for ch in all_chars:
            combos.append((random.choice(all_poses), ch))
    elif people_state.get("mode") == "pose":
        pose = pose_by_key.get(people_state.get("pose"))
        if pose:
            combos = [(pose, ch) for ch in all_chars]
    else:
        ch = char_by_key.get(people_state.get("person"))
        if ch:
            combos = [(pose, ch) for pose in all_poses]
    _marc_people_on_landmark()
    _SEAT = ("bench", "picnic_table")
    _CLIMB = ("vending_machine", "trash_can", "hydrant", "postbox")
    _VEH = ("police_car", "normal_car_1", "normal_car_2",
            "sports_car", "sports_car_2", "suv", "taxi")
    _WRONG = _CLIMB + _VEH
    _ON_WRONG_POSES = ("person_sitting", "person_crouching", "person_reaching")
    _ON_SEAT_POSES = ("person_sitting", "person_reclining", "person_lying_down",
                      "person_prone")
    if random.random() < _marc_frac("TRAINER_FORCE_CLIMB_LANDMARK_PROB", 0.35):
        _have = {_marc_lm_typ(s) for s in landmarks}
        if not (_have & set(_CLIMB)):
            landmarks.append({"name": "%s_0" % random.choice(_CLIMB)})
    _lm_types = []
    for _s in landmarks:
        _t = _marc_lm_typ(_s)
        if _t and _t not in _lm_types:
            _lm_types.append(_t)
    _seats = [t for t in _lm_types if t in _SEAT]
    _wrongs = [t for t in _lm_types if t in _WRONG]
    _pose_set = set(all_poses)
    for pose, ch in combos:
        spec = {"type": pose, "character": ch, "count": 1}
        if random.random() < _marc_frac("TRAINER_PEOPLE_ON_LANDMARK_PROB", 0.45):
            want_wrong = random.random() < _marc_frac(
                "TRAINER_PEOPLE_WRONG_PLACE_PROB", 0.55)
            lm_typ = None
            if want_wrong and _wrongs:
                lm_typ = random.choice(_wrongs)
            elif _seats:
                lm_typ = random.choice(_seats)
            elif _wrongs:
                lm_typ = random.choice(_wrongs)
            if lm_typ:
                spec["placement"] = {
                    "on_landmark_type": lm_typ,
                    "z_offset_from_top": 0.02,
                }
                pool = _ON_WRONG_POSES if lm_typ in _WRONG else _ON_SEAT_POSES
                fit = [p for p in pool if p in _pose_set]
                if fit:
                    spec["type"] = random.choice(fit)
        objects.append(spec)
'''

NEW_COMBOS = '''    combos = []
    if not people_state.get("enabled", True):
        # "Add people" 해제 → 사람 미배치.
        return landmarks, objects
    def _marc_people_all_chars():
        """[marc-patch] One of each identity, random pose (covers female/professor/weight)."""
        raw = os.environ.get("TRAINER_PEOPLE_ALL_CHARS", "1")
        return str(raw).strip().lower() not in ("0", "false", "no", "off")
    def _marc_frac(name, default):
        """[marc-patch] Clamp an env probability into [0, 1]."""
        try:
            v = float(os.environ.get(name, str(default)) or str(default))
        except (TypeError, ValueError):
            v = float(default)
        return max(0.0, min(1.0, v))
    def _marc_flag(name, default="0"):
        raw = os.environ.get(name, default)
        return str(raw).strip().lower() not in ("0", "false", "no", "off", "")
    def _marc_yolo_mode():
        """[marc-patch] _marc_yolo_mode: ordinary CCTV, no SAR climb seating."""
        return _marc_flag("TRAINER_YOLO_MODE", "0")
    def _marc_csv_names(name):
        return [x.strip() for x in (os.environ.get(name, "") or "").split(",")
                if x.strip()]
    def _marc_filter_names(want, available):
        avail = list(available or [])
        if not want:
            return avail
        out = []
        lower = [(str(x).lower(), x) for x in avail]
        for w in want:
            wl = str(w or "").strip().lower()
            if not wl:
                continue
            aliases = [wl]
            if not wl.startswith("person_"):
                aliases.append("person_" + wl)
            chosen = [x for lo, x in lower if lo in aliases]
            if not chosen:
                chosen = [
                    x for lo, x in lower
                    if any(
                        lo.startswith(a + "_")
                        or ("_" + a + "_") in ("_" + lo + "_")
                        for a in aliases
                    )
                ]
            for x in chosen:
                if x not in out:
                    out.append(x)
        return out
    def _marc_people_on_landmark():
        """[marc-patch] Marker: people may sit/climb on landmark tops."""
        return True
    def _marc_people_above():
        """[marc-patch] Marker: abnormal climbed-above (landmark XY + Z lift)."""
        return _marc_flag("TRAINER_PEOPLE_ABOVE", "0")
    def _marc_above_type_weight(typ):
        parsed = {}
        raw = os.environ.get("TRAINER_PEOPLE_ABOVE_WEIGHTS", "") or ""
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                key, val = part.split(":", 1)
            elif "=" in part:
                key, val = part.split("=", 1)
            else:
                continue
            try:
                parsed[key.strip()] = max(0.0, float(val))
            except (TypeError, ValueError):
                continue
        if typ in parsed:
            return parsed[typ]
        return 1.0
    def _marc_above_clearance():
        try:
            return float(os.environ.get(
                "TRAINER_PEOPLE_ABOVE_CLEARANCE", "0.03") or "0.03")
        except (TypeError, ValueError):
            return 0.03
    def _marc_sit_sink(typ, pose):
        if str(pose or "") == "person_crouching":
            if str(typ or "") != "hydrant":
                return 0.0
            try:
                return float(os.environ.get(
                    "TRAINER_PEOPLE_CROUCH_SINK_HYDRANT", "0.18") or "0.18")
            except (TypeError, ValueError):
                return 0.18
        if str(pose or "") != "person_sitting":
            return 0.0
        if str(typ or "") == "vending_machine":
            try:
                return float(os.environ.get(
                    "TRAINER_PEOPLE_VEND_BUTT_SINK", "0.12") or "0.12")
            except (TypeError, ValueError):
                return 0.12
        if str(typ or "") == "trash_can":
            # _marc_trash_sit_rim: sit on the lid, not inside the can.
            try:
                return float(os.environ.get(
                    "TRAINER_PEOPLE_SIT_SINK_TRASH", "0.42") or "0.42")
            except (TypeError, ValueError):
                return 0.42
        try:
            sink = float(os.environ.get(
                "TRAINER_PEOPLE_SIT_SINK", "0.70") or "0.70")
        except (TypeError, ValueError):
            sink = 0.70
        if str(typ or "") in ("postbox", "hydrant"):
            try:
                sink += float(os.environ.get(
                    "TRAINER_PEOPLE_SIT_SINK_SHORT", "0.35") or "0.35")
            except (TypeError, ValueError):
                sink += 0.35
        return sink
    def _marc_above_climber_pose(typ, pose_set):
        # TRAINER_PEOPLE_POSES pins the climber to that set. Otherwise
        # vending prefers sit/reach; other climb types stay even.
        if _marc_csv_names("TRAINER_PEOPLE_POSES"):
            pool = [p for p in pose_set if p]
            if pool:
                return random.choice(pool)
            return "person_sitting"
        if str(typ or "") == "vending_machine":
            weighted = (
                ("person_sitting", 4),
                ("person_reaching", 4),
                ("person_crouching", 1),
                ("person_walking", 1),
            )
        else:
            weighted = (
                ("person_crouching", 1),
                ("person_sitting", 1),
                ("person_walking", 1),
            )
        pool = []
        for p, w in weighted:
            if p in pose_set:
                pool.extend([p] * int(w))
        if not pool:
            return "person_sitting"
        return random.choice(pool)
    def _marc_lm_typ(spec):
        import re as _re_lm
        return _re_lm.sub(r"_[0-9]+$", "", str((spec or {}).get("name") or ""))
    def _marc_is_seat(typ):
        return (typ or "").lower() in ("bench", "picnic_table")
    def _marc_people_is_vehicle(typ):
        t = (typ or "").lower()
        return ("car" in t) or t in ("suv", "taxi")
    def _marc_is_above_ok(typ):
        t = (typ or "").lower()
        if _marc_is_seat(t):
            return False
        if _marc_people_is_vehicle(t):
            return False
        return t in (
            "vending_machine", "trash_can", "hydrant", "postbox",
            "bicycle", "kick_scooter",
        )
    _char_filter = _marc_csv_names("TRAINER_PEOPLE_CHARS")
    _pose_filter = _marc_csv_names("TRAINER_PEOPLE_POSES")
    all_chars = _marc_filter_names(_char_filter, all_chars)
    all_poses = _marc_filter_names(_pose_filter, all_poses)
    if all_chars and all_poses and (
            _marc_people_all_chars() or _char_filter):
        for ch in all_chars:
            combos.append((random.choice(all_poses), ch))
    elif people_state.get("mode") == "pose":
        pose = pose_by_key.get(people_state.get("pose"))
        if pose:
            combos = [(pose, ch) for ch in all_chars]
    else:
        ch = char_by_key.get(people_state.get("person"))
        if ch:
            combos = [(pose, ch) for pose in all_poses]
    _marc_people_on_landmark()
    _SEAT = ("bench", "picnic_table")
    _CLIMB = ("vending_machine", "trash_can", "hydrant", "postbox")
    _VEH = ("police_car", "normal_car_1", "normal_car_2",
            "sports_car", "sports_car_2", "suv", "taxi")
    _RIDE = ("bicycle", "kick_scooter")
    _WRONG = _CLIMB + _VEH
    _ABOVE_DEFAULT = _CLIMB + _VEH + _RIDE
    _ON_WRONG_POSES = ("person_sitting", "person_crouching", "person_reaching")
    _ON_SEAT_POSES = ("person_sitting", "person_reclining", "person_lying_down",
                      "person_prone")
    # Climb poses on non-sittable landmarks. person_walking is the catalog
    # standing/upright clip (there is no person_standing).
    _ON_ABOVE_POSES = (
        "person_crouching", "person_sitting", "person_walking",
        "person_reaching")
    _yolo_mode = _marc_yolo_mode()
    if _yolo_mode and (
            _marc_flag("TRAINER_SAR_RECIPES", "0")
            or _marc_flag("TRAINER_PEOPLE_ABOVE", "0")):
        if not globals().get("_marc_yolo_mode_logged"):
            globals()["_marc_yolo_mode_logged"] = True
            print("[people-patch] TRAINER_YOLO_MODE=1: ignoring SAR recipes and PEOPLE_ABOVE")
    _above_mode = False if _yolo_mode else _marc_people_above()
    _target = None
    if (not _above_mode and
            random.random() < _marc_frac("TRAINER_FORCE_CLIMB_LANDMARK_PROB", 0.35)):
        _have = {_marc_lm_typ(s) for s in landmarks}
        if not (_have & set(_CLIMB)):
            landmarks.append({"name": "%s_0" % random.choice(
                ("vending_machine", "trash_can", "hydrant", "postbox"))})
    _lm_types = []
    for _s in landmarks:
        _t = _marc_lm_typ(_s)
        if _t and _t not in _lm_types:
            _lm_types.append(_t)
    _raw_above = os.environ.get("TRAINER_PEOPLE_ABOVE_TYPES", "") or ""
    _above_filter = [x.strip() for x in _raw_above.split(",") if x.strip()]
    _aboves = [t for t in _lm_types if _marc_is_above_ok(t)]
    if _above_mode:
        # Cycle climb targets evenly (optional TRAINER_PEOPLE_ABOVE_WEIGHTS),
        # injecting a type only when missing.
        _base = [
            t for t in _above_filter if _marc_is_above_ok(t)
        ] or ["vending_machine", "trash_can", "hydrant",
              "postbox", "bicycle", "kick_scooter"]
        _pool = []
        for _t in _base:
            _rep = int(round(_marc_above_type_weight(_t)))
            if _rep > 0:
                _pool.extend([_t] * _rep)
        if not _pool:
            _pool = list(_base)
        global _marc_above_type_cursor, _marc_above_char_cursor
        try:
            _cursor = int(_marc_above_type_cursor)
        except (NameError, TypeError, ValueError):
            _cursor = 0
        _target = _pool[_cursor % len(_pool)]
        _marc_above_type_cursor = _cursor + 1
        if not any(_marc_lm_typ(s) == _target for s in landmarks):
            landmarks.append({"name": "%s_0" % _target})
        if _target not in _lm_types:
            _lm_types.append(_target)
        _aboves = [_target]
        # One climb target plus seats. Extra vendings/hydrants/trash sit off
        # camera and look like "vending in frame with nobody on it."
        landmarks[:] = (
            [s for s in landmarks if _marc_lm_typ(s) == _target][:1]
            + [s for s in landmarks if _marc_is_seat(_marc_lm_typ(s))]
        )
        _lm_types = []
        for _s in landmarks:
            _t = _marc_lm_typ(_s)
            if _t and _t not in _lm_types:
                _lm_types.append(_t)
    elif _above_filter:
        _want = set(t for t in _above_filter if _marc_is_above_ok(t))
        _aboves = [t for t in _lm_types if t in _want]
    _seats = [t for t in _lm_types if _marc_is_seat(t)]
    _wrongs = [t for t in _lm_types if t in _WRONG or _marc_people_is_vehicle(t)]
    _pose_set = set(all_poses)
    def _marc_sar_recipe_cursor_start():
        raw = os.environ.get("TRAINER_SAR_RECIPE_CURSOR", "")
        if str(raw).strip() != "":
            try:
                return max(0, int(raw))
            except (TypeError, ValueError):
                pass
        scenario = str(os.environ.get("ENV_MARC_SCENARIO") or "").strip() or (
            "marc2026_chungmu")
        root = os.path.join(
            os.environ.get("MARC_TRAINER_DIR") or "/metacom2026/trainer_output",
            scenario,
        )
        best = -1
        try:
            names = os.listdir(root)
        except OSError:
            return 0
        for name in names:
            if not str(name).startswith("scene_"):
                continue
            try:
                n = int(str(name).split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if n > best:
                best = n
        return best + 1
    def _marc_sar_recipe_list():
        paths = [
            "/marc-patches/sar_recipes.yaml",
            os.path.join(os.getcwd(), "simulation-platform", "patches",
                         "sar_recipes.yaml"),
        ]
        for path in paths:
            try:
                import yaml as _yaml_rec
                with open(path, encoding="utf-8") as _rf:
                    _data = _yaml_rec.safe_load(_rf) or {}
                _recs = list(_data.get("recipes") or [])
                _wanted = {
                    x.strip() for x in os.environ.get(
                        "TRAINER_SAR_RECIPE_IDS", "").split(",") if x.strip()
                }
                if _wanted:
                    _recs = [r for r in _recs if r.get("id") in _wanted]
                if _recs:
                    return _recs
                print("[people-patch] ERROR: %s has no recipes" % path)
            except FileNotFoundError:
                continue
            except Exception as _e:
                print("[people-patch] ERROR: could not load %s: %s" % (path, _e))
        print("[people-patch] ERROR: sar_recipes.yaml not found; recipes disabled")
        return []
    def _marc_recipe_rel(rec):
        # PyYAML 1.1 parses a bare `on` as True. Seat-lie recipes must
        # still get on_landmark_type.
        r = rec.get("relation") if rec else None
        if r is True:
            return "on"
        return str(r or "").strip().lower()
    def _marc_dump_recipe(rec, pose=None, character=None):
        if not rec:
            return
        try:
            import json as _json_rec
            with open("/tmp/marc_sar_recipe.json", "w", encoding="utf-8") as _rf:
                _json_rec.dump({
                    "recipe_id": rec.get("id"),
                    "target_pose": pose or rec.get("pose"),
                    "target_character": character,
                    "support_class": rec.get("support_class") or rec.get(
                        "support_landmark"),
                    "surface_class": rec.get("surface_class"),
                    "support_landmark": rec.get("support_landmark"),
                    "surface_anchor": rec.get("surface_anchor"),
                    "extra_landmarks": list(rec.get("extra_landmarks") or []),
                    "tip_landmark": rec.get("tip_landmark"),
                    "relation": _marc_recipe_rel(rec),
                    "above": bool(rec.get("above")),
                    "command": rec.get("command"),
                    "situation": rec.get("situation"),
                    "scene_hint": rec.get("scene_hint"),
                }, _rf)
        except Exception:
            pass
    _recipe = None
    if (not _yolo_mode) and _marc_flag("TRAINER_SAR_RECIPES", "0"):
        _recs = _marc_sar_recipe_list()
        if _recs:
            global _marc_sar_recipe_cursor
            try:
                _rc = int(_marc_sar_recipe_cursor)
            except (NameError, TypeError, ValueError):
                _rc = _marc_sar_recipe_cursor_start()
                print("[people-patch] SAR recipe cursor start=%d (%s)" % (
                    _rc, _recs[_rc % len(_recs)].get("id")))
            _recipe = _recs[_rc % len(_recs)]
            _marc_sar_recipe_cursor = _rc + 1
            _need = _recipe.get("support_landmark")
            if _need and not any(_marc_lm_typ(s) == _need for s in landmarks):
                landmarks.append({"name": "%s_0" % _need})
            _extras = list(_recipe.get("extra_landmarks") or [])
            _anchor = _recipe.get("surface_anchor")
            if _anchor:
                _extras.append(_anchor)
            _tip = _recipe.get("tip_landmark")
            if _tip:
                _extras.append(_tip)
            for _extra in _extras:
                if _extra and not any(_marc_lm_typ(s) == _extra for s in landmarks):
                    landmarks.append({"name": "%s_0" % _extra})
            if _recipe.get("above"):
                # Extra climb cabinets sit on the FOV rim and show as a
                # cropped "vending with nobody on it." Keep the recipe
                # support plus seats / non-climb clutter.
                _keep = set()
                for _k in (
                        _recipe.get("support_landmark"),
                        _recipe.get("surface_anchor"),
                        _recipe.get("tip_landmark"),
                ):
                    if _k:
                        _keep.add(_k)
                for _extra in list(_recipe.get("extra_landmarks") or []):
                    if _extra:
                        _keep.add(_extra)
                landmarks[:] = [
                    s for s in landmarks
                    if (_marc_lm_typ(s) in _keep
                        or _marc_is_seat(_marc_lm_typ(s))
                        or not _marc_is_above_ok(_marc_lm_typ(s)))
                ]
            _marc_dump_recipe(_recipe)
    if (_above_mode or _recipe is not None) and combos:
        # Climber identity is random per scene. Recipe mode used catalog
        # order, so every stacked sitter was male_student_01.
        random.shuffle(combos)
    _climber_placed = False
    _recipe_target_placed = False
    for pose, ch in combos:
        spec = {"type": pose, "character": ch, "count": 1}
        if _recipe is not None and not _recipe_target_placed:
            _want = str(_recipe.get("pose") or pose)
            if _want not in _pose_set and _want == "person_prone" and (
                    "person_lying_down" in _pose_set):
                _want = "person_lying_down"
            elif _want not in _pose_set:
                _want = pose if pose in _pose_set else (
                    list(_pose_set)[0] if _pose_set else pose)
            spec["type"] = _want
            _pl = {
                "recipe_id": _recipe.get("id"),
                "support_class": _recipe.get("support_class") or _recipe.get(
                    "support_landmark"),
                "surface_class": _recipe.get("surface_class"),
            }
            _lm = _recipe.get("support_landmark")
            _rel = _marc_recipe_rel(_recipe)
            if _lm and (_recipe.get("above") or _rel in ("on", "above")):
                _pl["on_landmark_type"] = _lm
                _pl["above"] = bool(_recipe.get("above"))
                if _pl["above"]:
                    _pl["contact_clearance"] = _marc_above_clearance()
                    _pl["z_offset_from_top"] = (
                        _marc_above_clearance()
                        - _marc_sit_sink(_lm, spec["type"])
                    )
                else:
                    _pl["z_offset_from_top"] = 0.02
            _beside_lm = _lm if (
                _lm and (_recipe.get("beside") or _rel == "beside")
            ) else None
            if not _beside_lm and (
                    _recipe.get("beside") or _rel == "beside"):
                _beside_lm = _recipe.get("surface_anchor")
            if _beside_lm:
                _pl["beside_landmark_type"] = _beside_lm
            spec["placement"] = _pl
            _marc_dump_recipe(_recipe, pose=spec["type"], character=ch)
            _recipe_target_placed = True
        elif _recipe is not None:
            _dposes = list(_recipe.get("distractor_poses") or [])
            _fit = [p for p in _dposes if p in _pose_set] or [
                p for p in _pose_set if p]
            spec["type"] = random.choice(_fit) if _fit else pose
            spec["placement"] = {
                "on_landmark_type": None,
                "above": False,
            }
        elif _above_mode and _target and not _climber_placed:
            spec["type"] = _marc_above_climber_pose(_target, _pose_set)
            spec["placement"] = {
                "on_landmark_type": _target,
                "above": True,
                "contact_clearance": _marc_above_clearance(),
                "z_offset_from_top": (
                    _marc_above_clearance()
                    - _marc_sit_sink(_target, spec["type"])
                ),
            }
            _climber_placed = True
        elif _above_mode:
            # Pose YAML may stamp on_landmark_type (prone on the vending).
            # Extras must stay on the ground, not share the climb target.
            spec["placement"] = {
                "on_landmark_type": None,
                "above": False,
            }
        elif (not _above_mode) and random.random() < _marc_frac(
                "TRAINER_PEOPLE_ON_LANDMARK_PROB", 0.45):
            want_wrong = (not _yolo_mode) and random.random() < _marc_frac(
                "TRAINER_PEOPLE_WRONG_PLACE_PROB", 0.55)
            lm_typ = None
            if want_wrong and _wrongs:
                lm_typ = random.choice(_wrongs)
            elif _seats:
                lm_typ = random.choice(_seats)
            elif (not _yolo_mode) and _wrongs:
                lm_typ = random.choice(_wrongs)
            if lm_typ and lm_typ != _target:
                spec["placement"] = {
                    "on_landmark_type": lm_typ,
                    "z_offset_from_top": 0.02,
                }
                pool = _ON_WRONG_POSES if (
                    lm_typ in _WRONG or _marc_people_is_vehicle(lm_typ)
                ) else _ON_SEAT_POSES
                fit = [p for p in pool if p in _pose_set]
                if fit:
                    spec["type"] = random.choice(fit)
        objects.append(spec)
    if _above_mode and _target and not _climber_placed and all_chars and all_poses:
        _fb_type = _marc_above_climber_pose(_target, _pose_set)
        objects.append({
            "type": _fb_type,
            "character": random.choice(all_chars),
            "count": 1,
            "placement": {
                "on_landmark_type": _target,
                "above": True,
                "contact_clearance": _marc_above_clearance(),
                "z_offset_from_top": (
                    _marc_above_clearance()
                    - _marc_sit_sink(_target, _fb_type)
                ),
            },
        })
'''


def _write(src):
    try:
        with open(TARGET, "w", encoding="utf-8") as f:
            f.write(src)
        return True
    except OSError as e:
        print(f"[people-patch] write failed, skipping: {e}")
        return False


def main():
    if not os.path.exists(TARGET):
        print(f"[people-patch] source not found, skipping: {TARGET}")
        return 0

    with open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        changed = False
        if "_marc_yolo_mode" not in src:
            _old_flag_fn = (
                '    def _marc_flag(name, default="0"):\n'
                '        raw = os.environ.get(name, default)\n'
                '        return str(raw).strip().lower() not in '
                '("0", "false", "no", "off", "")\n'
                '    def _marc_csv_names(name):\n'
            )
            _new_flag_fn = (
                '    def _marc_flag(name, default="0"):\n'
                '        raw = os.environ.get(name, default)\n'
                '        return str(raw).strip().lower() not in '
                '("0", "false", "no", "off", "")\n'
                '    def _marc_yolo_mode():\n'
                '        """[marc-patch] _marc_yolo_mode: ordinary CCTV, '
                'no SAR climb seating."""\n'
                '        return _marc_flag("TRAINER_YOLO_MODE", "0")\n'
                '    def _marc_csv_names(name):\n'
            )
            if _old_flag_fn in src:
                src = src.replace(_old_flag_fn, _new_flag_fn, 1)
                changed = True
            _old_above_assign = (
                '    _above_mode = _marc_people_above()\n'
            )
            _new_above_assign = (
                '    _yolo_mode = _marc_yolo_mode()\n'
                '    if _yolo_mode and (\n'
                '            _marc_flag("TRAINER_SAR_RECIPES", "0")\n'
                '            or _marc_flag("TRAINER_PEOPLE_ABOVE", "0")):\n'
                '        if not globals().get("_marc_yolo_mode_logged"):\n'
                '            globals()["_marc_yolo_mode_logged"] = True\n'
                '            print("[people-patch] TRAINER_YOLO_MODE=1: '
                'ignoring SAR recipes and PEOPLE_ABOVE")\n'
                '    _above_mode = False if _yolo_mode else '
                '_marc_people_above()\n'
            )
            if _old_above_assign in src:
                src = src.replace(_old_above_assign, _new_above_assign, 1)
                changed = True
            _old_sar_if = (
                '    _recipe = None\n'
                '    if _marc_flag("TRAINER_SAR_RECIPES", "0"):\n'
            )
            _new_sar_if = (
                '    _recipe = None\n'
                '    if (not _yolo_mode) and _marc_flag('
                '"TRAINER_SAR_RECIPES", "0"):\n'
            )
            if _old_sar_if in src:
                src = src.replace(_old_sar_if, _new_sar_if, 1)
                changed = True
            _old_wrong = (
                '            want_wrong = random.random() < _marc_frac(\n'
                '                "TRAINER_PEOPLE_WRONG_PLACE_PROB", 0.55)\n'
                '            lm_typ = None\n'
                '            if want_wrong and _wrongs:\n'
                '                lm_typ = random.choice(_wrongs)\n'
                '            elif _seats:\n'
                '                lm_typ = random.choice(_seats)\n'
                '            elif _wrongs:\n'
                '                lm_typ = random.choice(_wrongs)\n'
            )
            _new_wrong = (
                '            want_wrong = (not _yolo_mode) and '
                'random.random() < _marc_frac(\n'
                '                "TRAINER_PEOPLE_WRONG_PLACE_PROB", 0.55)\n'
                '            lm_typ = None\n'
                '            if want_wrong and _wrongs:\n'
                '                lm_typ = random.choice(_wrongs)\n'
                '            elif _seats:\n'
                '                lm_typ = random.choice(_seats)\n'
                '            elif (not _yolo_mode) and _wrongs:\n'
                '                lm_typ = random.choice(_wrongs)\n'
            )
            if _old_wrong in src:
                src = src.replace(_old_wrong, _new_wrong, 1)
                changed = True
        _old_vend_w = (
            '        if typ == "vending_machine":\n'
            '            return 3.0\n'
            '        return 1.0\n'
        )
        _new_vend_w = '        return 1.0\n'
        if _old_vend_w in src:
            src = src.replace(_old_vend_w, _new_vend_w, 1)
            changed = True
        if "TRAINER_PEOPLE_CHARS" not in src:
            _old_flag = (
                '        return str(raw).strip().lower() not in '
                '("0", "false", "no", "off", "")\n'
                '    def _marc_people_on_landmark():\n'
            )
            _new_flag = (
                '        return str(raw).strip().lower() not in '
                '("0", "false", "no", "off", "")\n'
                '    def _marc_csv_names(name):\n'
                '        return [x.strip() for x in '
                '(os.environ.get(name, "") or "").split(",")\n'
                '                if x.strip()]\n'
                '    def _marc_filter_names(want, available):\n'
                '        avail = list(available or [])\n'
                '        if not want:\n'
                '            return avail\n'
                '        out = []\n'
                '        lower = [(str(x).lower(), x) for x in avail]\n'
                '        for w in want:\n'
                '            wl = str(w or "").strip().lower()\n'
                '            if not wl:\n'
                '                continue\n'
                '            aliases = [wl]\n'
                '            if not wl.startswith("person_"):\n'
                '                aliases.append("person_" + wl)\n'
                '            chosen = [x for lo, x in lower if lo in aliases]\n'
                '            if not chosen:\n'
                '                chosen = [\n'
                '                    x for lo, x in lower\n'
                '                    if any(\n'
                '                        lo.startswith(a + "_")\n'
                '                        or ("_" + a + "_") in ("_" + lo + "_")\n'
                '                        for a in aliases\n'
                '                    )\n'
                '                ]\n'
                '            for x in chosen:\n'
                '                if x not in out:\n'
                '                    out.append(x)\n'
                '        return out\n'
                '    def _marc_people_on_landmark():\n'
            )
            if _old_flag in src:
                src = src.replace(_old_flag, _new_flag, 1)
                changed = True
            _old_combos_if = (
                '    if _marc_people_all_chars() and all_chars and all_poses:\n'
            )
            _new_combos_if = (
                '    _char_filter = _marc_csv_names("TRAINER_PEOPLE_CHARS")\n'
                '    _pose_filter = _marc_csv_names("TRAINER_PEOPLE_POSES")\n'
                '    all_chars = _marc_filter_names(_char_filter, all_chars)\n'
                '    all_poses = _marc_filter_names(_pose_filter, all_poses)\n'
                '    if all_chars and all_poses and (\n'
                '            _marc_people_all_chars() or _char_filter):\n'
            )
            if _old_combos_if in src:
                src = src.replace(_old_combos_if, _new_combos_if, 1)
                changed = True
            _old_climber_pose = (
                '    def _marc_above_climber_pose(typ, pose_set):\n'
                '        if str(typ or "") == "vending_machine":\n'
            )
            _new_climber_pose = (
                '    def _marc_above_climber_pose(typ, pose_set):\n'
                '        if _marc_csv_names("TRAINER_PEOPLE_POSES"):\n'
                '            pool = [p for p in pose_set if p]\n'
                '            if pool:\n'
                '                return random.choice(pool)\n'
                '            return "person_sitting"\n'
                '        if str(typ or "") == "vending_machine":\n'
            )
            if _old_climber_pose in src:
                src = src.replace(_old_climber_pose, _new_climber_pose, 1)
                changed = True
            _old_climber_pose_cmt = (
                '    def _marc_above_climber_pose(typ, pose_set):\n'
                '        # Vending stack: sit and reach most of the time. '
                'Other climb\n'
                '        # types stay an even crouch/sit/walk mix.\n'
                '        if str(typ or "") == "vending_machine":\n'
            )
            if _old_climber_pose_cmt in src:
                src = src.replace(_old_climber_pose_cmt, _new_climber_pose, 1)
                changed = True
        if "_marc_above_climber_pose" not in src:
            _old_lm = (
                '        return sink\n'
                '    def _marc_lm_typ(spec):\n'
            )
            _new_lm = (
                '        return sink\n'
                '    def _marc_above_climber_pose(typ, pose_set):\n'
                '        if _marc_csv_names("TRAINER_PEOPLE_POSES"):\n'
                '            pool = [p for p in pose_set if p]\n'
                '            if pool:\n'
                '                return random.choice(pool)\n'
                '            return "person_sitting"\n'
                '        if str(typ or "") == "vending_machine":\n'
                '            weighted = (\n'
                '                ("person_sitting", 4),\n'
                '                ("person_reaching", 4),\n'
                '                ("person_crouching", 1),\n'
                '                ("person_walking", 1),\n'
                '            )\n'
                '        else:\n'
                '            weighted = (\n'
                '                ("person_crouching", 1),\n'
                '                ("person_sitting", 1),\n'
                '                ("person_walking", 1),\n'
                '            )\n'
                '        pool = []\n'
                '        for p, w in weighted:\n'
                '            if p in pose_set:\n'
                '                pool.extend([p] * int(w))\n'
                '        if not pool:\n'
                '            return "person_sitting"\n'
                '        return random.choice(pool)\n'
                '    def _marc_lm_typ(spec):\n'
            )
            if _old_lm in src:
                src = src.replace(_old_lm, _new_lm, 1)
                changed = True
        _old_pick3 = (
            '            spec["type"] = random.choice(\n'
            '                [p for p in ("person_crouching", "person_sitting",\n'
            '                             "person_walking")\n'
            '                 if p in _pose_set] or ["person_sitting"])\n'
        )
        _new_pick = (
            '            spec["type"] = _marc_above_climber_pose('
            '_target, _pose_set)\n'
        )
        if _old_pick3 in src:
            src = src.replace(_old_pick3, _new_pick, 1)
            changed = True
        _old_fb3 = (
            '        _fb_type = random.choice(\n'
            '            [p for p in ("person_crouching", "person_sitting",\n'
            '                         "person_walking")\n'
            '             if p in _pose_set] or ["person_sitting"])\n'
        )
        _new_fb = (
            '        _fb_type = _marc_above_climber_pose(_target, _pose_set)\n'
        )
        if _old_fb3 in src:
            src = src.replace(_old_fb3, _new_fb, 1)
            changed = True
        if "_marc_sit_sink" not in src:
            _clear_fn = (
                '    def _marc_above_clearance():\n'
                '        try:\n'
                '            return float(os.environ.get(\n'
                '                "TRAINER_PEOPLE_ABOVE_CLEARANCE", "0.03") '
                'or "0.03")\n'
                '        except (TypeError, ValueError):\n'
                '            return 0.03\n'
            )
            _sink_fn = (
                _clear_fn
                + '    def _marc_sit_sink(typ, pose):\n'
                + '        if str(pose or "") == "person_crouching":\n'
                + '            if str(typ or "") != "hydrant":\n'
                + '                return 0.0\n'
                + '            try:\n'
                + '                return float(os.environ.get(\n'
                + '                    "TRAINER_PEOPLE_CROUCH_SINK_HYDRANT", '
                '"0.18") or "0.18")\n'
                + '            except (TypeError, ValueError):\n'
                + '                return 0.18\n'
                + '        if str(pose or "") != "person_sitting":\n'
                + '            return 0.0\n'
                + '        if str(typ or "") == "vending_machine":\n'
                + '            try:\n'
                + '                return float(os.environ.get(\n'
                + '                    "TRAINER_PEOPLE_VEND_BUTT_SINK", '
                '"0.12") or "0.12")\n'
                + '            except (TypeError, ValueError):\n'
                + '                return 0.12\n'
                + '        try:\n'
                + '            sink = float(os.environ.get(\n'
                + '                "TRAINER_PEOPLE_SIT_SINK", "0.70") '
                'or "0.70")\n'
                + '        except (TypeError, ValueError):\n'
                + '            sink = 0.70\n'
                + '        if str(typ or "") == "trash_can":\n'
                + '            try:\n'
                + '                sink += float(os.environ.get(\n'
                + '                    "TRAINER_PEOPLE_SIT_SINK_TRASH", '
                '"0.15") or "0.15")\n'
                + '            except (TypeError, ValueError):\n'
                + '                sink += 0.15\n'
                + '        elif str(typ or "") in ("postbox", "hydrant"):\n'
                + '            try:\n'
                + '                sink += float(os.environ.get(\n'
                + '                    "TRAINER_PEOPLE_SIT_SINK_SHORT", '
                '"0.35") or "0.35")\n'
                + '            except (TypeError, ValueError):\n'
                + '                sink += 0.35\n'
                + '        return sink\n'
            )
            if _clear_fn in src:
                src = src.replace(_clear_fn, _sink_fn, 1)
                changed = True
            for _old_off, _new_off in (
                (
                    '- (0.70 if spec["type"] == "person_sitting" else 0.0)',
                    '- _marc_sit_sink(_target, spec["type"])',
                ),
                (
                    '- (0.70 if _fb_type == "person_sitting" else 0.0)',
                    '- _marc_sit_sink(_target, _fb_type)',
                ),
                (
                    '- (0.35 if spec["type"] == "person_sitting" else 0.0)',
                    '- _marc_sit_sink(_target, spec["type"])',
                ),
                (
                    '- (0.35 if _fb_type == "person_sitting" else 0.0)',
                    '- _marc_sit_sink(_target, _fb_type)',
                ),
            ):
                if _old_off in src:
                    src = src.replace(_old_off, _new_off)
                    changed = True
        if ("_marc_sit_sink" in src
                and "TRAINER_PEOPLE_VEND_BUTT_SINK" not in src):
            _old_sit_ret = (
                '        if str(pose or "") != "person_sitting":\n'
                '            return 0.0\n'
                '        try:\n'
            )
            _new_sit_ret = (
                '        if str(pose or "") != "person_sitting":\n'
                '            return 0.0\n'
                '        if str(typ or "") == "vending_machine":\n'
                '            try:\n'
                '                return float(os.environ.get(\n'
                '                    "TRAINER_PEOPLE_VEND_BUTT_SINK", '
                '"0.12") or "0.12")\n'
                '            except (TypeError, ValueError):\n'
                '                return 0.12\n'
                '        try:\n'
            )
            if _old_sit_ret in src:
                src = src.replace(_old_sit_ret, _new_sit_ret, 1)
                changed = True
            _old_vend_extra = (
                '        elif str(typ or "") == "vending_machine":\n'
                '            try:\n'
                '                sink += float(os.environ.get(\n'
                '                    "TRAINER_PEOPLE_SIT_SINK_VENDING", '
                '"0.20") or "0.20")\n'
                '            except (TypeError, ValueError):\n'
                '                sink += 0.20\n'
                '        return sink\n'
            )
            _new_vend_extra = '        return sink\n'
            if _old_vend_extra in src:
                src = src.replace(_old_vend_extra, _new_vend_extra, 1)
                changed = True
        if "_marc_sit_sink" in src:
            _old_butt = (
                '"TRAINER_PEOPLE_VEND_BUTT_SINK", "0.15") or "0.15")\n'
                '            except (TypeError, ValueError):\n'
                '                return 0.15\n'
            )
            _new_butt = (
                '"TRAINER_PEOPLE_VEND_BUTT_SINK", "0.12") or "0.12")\n'
                '            except (TypeError, ValueError):\n'
                '                return 0.12\n'
            )
            if _old_butt in src:
                src = src.replace(_old_butt, _new_butt, 1)
                changed = True
            _old_butt04 = (
                '"TRAINER_PEOPLE_VEND_BUTT_SINK", "0.04") or "0.04")\n'
                '            except (TypeError, ValueError):\n'
                '                return 0.04\n'
            )
            if _old_butt04 in src:
                src = src.replace(_old_butt04, _new_butt, 1)
                changed = True
            _old_butt09 = (
                '"TRAINER_PEOPLE_VEND_BUTT_SINK", "0.09") or "0.09")\n'
                '            except (TypeError, ValueError):\n'
                '                return 0.09\n'
            )
            if _old_butt09 in src:
                src = src.replace(_old_butt09, _new_butt, 1)
                changed = True
        if ("_marc_sit_sink" in src
                and "TRAINER_PEOPLE_SIT_SINK_TRASH" not in src):
            _old_grouped = (
                '        if str(typ or "") in ("trash_can", "postbox", '
                '"hydrant"):\n'
                '            try:\n'
                '                sink += float(os.environ.get(\n'
                '                    "TRAINER_PEOPLE_SIT_SINK_SHORT", '
                '"0.35") or "0.35")\n'
                '            except (TypeError, ValueError):\n'
                '                sink += 0.35\n'
                '        elif str(typ or "") == "vending_machine":\n'
            )
            _new_split = (
                '        if str(typ or "") == "trash_can":\n'
                '            try:\n'
                '                sink += float(os.environ.get(\n'
                '                    "TRAINER_PEOPLE_SIT_SINK_TRASH", '
                '"0.15") or "0.15")\n'
                '            except (TypeError, ValueError):\n'
                '                sink += 0.15\n'
                '        elif str(typ or "") in ("postbox", "hydrant"):\n'
                '            try:\n'
                '                sink += float(os.environ.get(\n'
                '                    "TRAINER_PEOPLE_SIT_SINK_SHORT", '
                '"0.35") or "0.35")\n'
                '            except (TypeError, ValueError):\n'
                '                sink += 0.35\n'
                '        elif str(typ or "") == "vending_machine":\n'
            )
            if _old_grouped in src:
                src = src.replace(_old_grouped, _new_split, 1)
                changed = True
        if ("_marc_sit_sink" in src
                and "_marc_trash_sit_rim" not in src):
            _old_trash_add = (
                '        if str(typ or "") == "trash_can":\n'
                '            try:\n'
                '                sink += float(os.environ.get(\n'
                '                    "TRAINER_PEOPLE_SIT_SINK_TRASH", '
                '"0.15") or "0.15")\n'
                '            except (TypeError, ValueError):\n'
                '                sink += 0.15\n'
                '        elif str(typ or "") in ("postbox", "hydrant"):\n'
            )
            _new_trash_rim = (
                '        if str(typ or "") == "trash_can":\n'
                '            # _marc_trash_sit_rim: sit on the lid, not '
                'inside the can.\n'
                '            try:\n'
                '                return float(os.environ.get(\n'
                '                    "TRAINER_PEOPLE_SIT_SINK_TRASH", '
                '"0.42") or "0.42")\n'
                '            except (TypeError, ValueError):\n'
                '                return 0.42\n'
                '        if str(typ or "") in ("postbox", "hydrant"):\n'
            )
            if _old_trash_add in src:
                src = src.replace(_old_trash_add, _new_trash_rim, 1)
                changed = True
        if ("_marc_sit_sink" in src
                and '"TRAINER_PEOPLE_SIT_SINK_TRASH", "0.18"' in src):
            src = src.replace(
                '"TRAINER_PEOPLE_SIT_SINK_TRASH", "0.18") or "0.18")\n'
                '            except (TypeError, ValueError):\n'
                '                return 0.18\n',
                '"TRAINER_PEOPLE_SIT_SINK_TRASH", "0.42") or "0.42")\n'
                '            except (TypeError, ValueError):\n'
                '                return 0.42\n',
                1,
            )
            changed = True
        if ("_marc_sit_sink" in src
                and "TRAINER_PEOPLE_CROUCH_SINK_HYDRANT" not in src):
            _old_sit_only = (
                '    def _marc_sit_sink(typ, pose):\n'
                '        if str(pose or "") != "person_sitting":\n'
                '            return 0.0\n'
            )
            _new_crouch = (
                '    def _marc_sit_sink(typ, pose):\n'
                '        if str(pose or "") == "person_crouching":\n'
                '            if str(typ or "") != "hydrant":\n'
                '                return 0.0\n'
                '            try:\n'
                '                return float(os.environ.get(\n'
                '                    "TRAINER_PEOPLE_CROUCH_SINK_HYDRANT", '
                '"0.18") or "0.18")\n'
                '            except (TypeError, ValueError):\n'
                '                return 0.18\n'
                '        if str(pose or "") != "person_sitting":\n'
                '            return 0.0\n'
            )
            if _old_sit_only in src:
                src = src.replace(_old_sit_only, _new_crouch, 1)
                changed = True
        if "random.shuffle(combos)" not in src and "_climber_placed = False" in src:
            _old_rot = (
                "    if _above_mode and combos:\n"
                "        # First combo is the climber. Catalog order is male-first, so rotate\n"
                "        # identities or every stacked sitter is male_student_01.\n"
                "        try:\n"
                "            _ch_i = int(_marc_above_char_cursor)\n"
                "        except (NameError, TypeError, ValueError):\n"
                "            _ch_i = 0\n"
                "        _rot = _ch_i % len(combos)\n"
                "        combos = combos[_rot:] + combos[:_rot]\n"
                "        _marc_above_char_cursor = _ch_i + 1\n"
                "    _climber_placed = False\n"
            )
            _new_rot = (
                "    if _above_mode and combos:\n"
                "        random.shuffle(combos)\n"
                "    _climber_placed = False\n"
            )
            if _old_rot in src:
                src = src.replace(_old_rot, _new_rot, 1)
                changed = True
            _old_fb_ch = (
                '            "character": all_chars[_ch_i % len(all_chars)],\n'
            )
            _new_fb_ch = (
                '            "character": random.choice(all_chars),\n'
            )
            if _old_fb_ch in src:
                src = src.replace(_old_fb_ch, _new_fb_ch, 1)
                changed = True
        if "_marc_above_char_cursor" not in src and "_climber_placed = False" in src:
            src = src.replace(
                "        global _marc_above_type_cursor\n",
                "        global _marc_above_type_cursor, "
                "_marc_above_char_cursor\n",
                1,
            )
            _old_first = (
                "    _pose_set = set(all_poses)\n"
                "    _climber_placed = False\n"
            )
            _new_first = (
                "    _pose_set = set(all_poses)\n"
                "    if _above_mode and combos:\n"
                "        random.shuffle(combos)\n"
                "    _climber_placed = False\n"
            )
            if _old_first in src:
                src = src.replace(_old_first, _new_first, 1)
                changed = True
        _old_above_poses = (
            '    _ON_ABOVE_POSES = ("person_crouching", "person_sitting")\n'
        )
        _new_above_poses = (
            '    _ON_ABOVE_POSES = ("person_crouching", "person_sitting", '
            '"person_walking")\n'
        )
        if _old_above_poses in src:
            src = src.replace(_old_above_poses, _new_above_poses, 1)
            changed = True
        _old_above_poses3 = (
            '    _ON_ABOVE_POSES = ("person_crouching", "person_sitting", '
            '"person_walking")\n'
        )
        _new_above_poses4 = (
            '    _ON_ABOVE_POSES = (\n'
            '        "person_crouching", "person_sitting", "person_walking",\n'
            '        "person_reaching")\n'
        )
        if _old_above_poses3 in src:
            src = src.replace(_old_above_poses3, _new_above_poses4, 1)
            changed = True
        _old_fit2 = (
            '[p for p in ("person_crouching", "person_sitting")\n'
            '                 if p in _pose_set] or ["person_sitting"])\n'
        )
        _new_fit3 = (
            '[p for p in ("person_crouching", "person_sitting", '
            '"person_walking")\n'
            '                 if p in _pose_set] or ["person_sitting"])\n'
        )
        if _old_fit2 in src:
            src = src.replace(_old_fit2, _new_fit3)
            changed = True
        _old_pin_sit = (
            '            if _target in ("vending_machine", "postbox"):\n'
            '                _tall = [p for p in ("person_sitting", '
            '"person_crouching")\n'
            '                         if p in _pose_set]\n'
            '                if "person_sitting" in _tall:\n'
            '                    spec["type"] = "person_sitting"\n'
            '                elif _tall:\n'
            '                    spec["type"] = random.choice(_tall)\n'
        )
        if _old_pin_sit in src:
            src = src.replace(_old_pin_sit, "", 1)
            changed = True
        _helper_pick = (
            '            spec["type"] = _marc_above_climber_pose('
            '_target, _pose_set)\n'
        )
        _helper_fb = (
            '        _fb_type = _marc_above_climber_pose(_target, _pose_set)\n'
        )
        # Repeat after 2-pose -> 3-pose upgrades so an already-patched
        # trainer still gets the vending sit/reach bias.
        _old_pick3b = (
            '            spec["type"] = random.choice(\n'
            '                [p for p in ("person_crouching", "person_sitting",\n'
            '                             "person_walking")\n'
            '                 if p in _pose_set] or ["person_sitting"])\n'
        )
        _old_pick3_inline = (
            '            spec["type"] = random.choice(\n'
            '                [p for p in ("person_crouching", "person_sitting", '
            '"person_walking")\n'
            '                 if p in _pose_set] or ["person_sitting"])\n'
        )
        for _old_even in (_old_pick3b, _old_pick3_inline):
            if _old_even in src:
                src = src.replace(_old_even, _helper_pick, 1)
                changed = True
        _old_fb3b = (
            '        _fb_type = random.choice(\n'
            '            [p for p in ("person_crouching", "person_sitting",\n'
            '                         "person_walking")\n'
            '             if p in _pose_set] or ["person_sitting"])\n'
        )
        if _old_fb3b in src:
            src = src.replace(_old_fb3b, _helper_fb, 1)
            changed = True
        _old_fit_above = (
            '        if _above_mode and _target and not _climber_placed:\n'
            '            _fit_above = [p for p in '
            '("person_crouching", "person_sitting", "person_walking") '
            'if p in _pose_set]\n'
            '            if _fit_above:\n'
            '                spec["type"] = random.choice(_fit_above)\n'
            '            spec["placement"] = {\n'
        )
        _new_fit_above = (
            '        if _above_mode and _target and not _climber_placed:\n'
            + _helper_pick
            + '            spec["placement"] = {\n'
        )
        if _old_fit_above in src:
            src = src.replace(_old_fit_above, _new_fit_above, 1)
            changed = True
        _old_fb_all = (
            "    if _above_mode and _target and not _climber_placed "
            "and all_chars and all_poses:\n"
            "        objects.append({\n"
            "            \"type\": random.choice(all_poses),\n"
        )
        _new_fb_all = (
            "    if _above_mode and _target and not _climber_placed "
            "and all_chars and all_poses:\n"
            + _helper_fb
            + "        objects.append({\n"
            "            \"type\": _fb_type,\n"
        )
        if _old_fb_all in src:
            src = src.replace(_old_fb_all, _new_fb_all, 1)
            changed = True
        if _helper_pick not in src and "_climber_placed" in src:
            src2, nsub = re.subn(
                r"(        if _above_mode and _target and not _climber_placed:\n)"
                r"(?:            .*\n)*?"
                r"(            spec\[\"placement\"\] = \{\n)",
                r"\1" + _helper_pick + r"\2",
                src,
                count=1,
            )
            if nsub:
                src = src2
                changed = True
        old_unconstrained_climber = (
            '        if _above_mode and _target and not _climber_placed:\n'
            '            spec["placement"] = {\n'
        )
        new_constrained_climber = (
            '        if _above_mode and _target and not _climber_placed:\n'
            + _helper_pick
            + '            spec["placement"] = {\n'
        )
        if (old_unconstrained_climber in src
                and new_constrained_climber not in src):
            src = src.replace(
                old_unconstrained_climber, new_constrained_climber, 1)
            changed = True
        old_pin = (
            '            fit = [p for p in _ON_ABOVE_POSES if p in _pose_set]\n'
            '            if _target in ("vending_machine", "postbox"):\n'
            '                _tall = [p for p in ("person_sitting", "person_crouching")\n'
            '                         if p in _pose_set]\n'
            '                if _tall:\n'
            '                    fit = _tall\n'
            '            if fit:\n'
            '                spec["type"] = random.choice(fit)\n'
            '            _climber_placed = True\n'
        )
        if old_pin in src:
            src = src.replace(old_pin, "            _climber_placed = True\n", 1)
            changed = True
        fallback = (
            "    if _above_mode and _target and not _climber_placed "
            "and all_chars and all_poses:"
        )
        if fallback not in src and "_climber_placed = False" in src:
            needle = (
                "                if fit:\n"
                "                    spec[\"type\"] = random.choice(fit)\n"
                "        objects.append(spec)\n"
            )
            insert = needle + (
                "    if _above_mode and _target and not _climber_placed "
                "and all_chars and all_poses:\n"
                "        _fb_type = _marc_above_climber_pose(_target, _pose_set)\n"
                "        objects.append({\n"
                "            \"type\": _fb_type,\n"
                "            \"character\": all_chars[0],\n"
                "            \"count\": 1,\n"
                "            \"placement\": {\n"
                "                \"on_landmark_type\": _target,\n"
                "                \"above\": True,\n"
                "                \"contact_clearance\": _marc_above_clearance(),\n"
                "                \"z_offset_from_top\": _marc_above_clearance(),\n"
                "            },\n"
                "        })\n"
            )
            if needle in src:
                src = src.replace(needle, insert, 1)
                changed = True
        _old_seat = (
            '_ON_SEAT_POSES = ("person_sitting", "person_reclining", '
            '"person_lying_down")'
        )
        _new_seat = (
            '_ON_SEAT_POSES = ("person_sitting", "person_reclining", '
            '"person_lying_down",\n'
            '                      "person_prone")'
        )
        if _old_seat in src:
            src = src.replace(_old_seat, _new_seat)
            changed = True
        if "def _marc_sar_recipe_list():" in src and "target_character" not in src:
            _a = src.find("    def _marc_sar_recipe_list():")
            _b = src.find("    if _above_mode and combos:")
            if _b < 0:
                _b = src.find(
                    "    if (_above_mode or _recipe is not None) and combos:")
            _shuf_if = "    if (_above_mode or _recipe is not None) and combos:"
            _new_rec = NEW_COMBOS[
                NEW_COMBOS.find("    def _marc_sar_recipe_list():"):
                NEW_COMBOS.find(_shuf_if)
            ]
            if _a >= 0 and _b > _a and _new_rec:
                src = src[:_a] + _new_rec + src[_b:]
                changed = True
            _old_tgt = (
                "            if _lm and (_recipe.get(\"beside\") or "
                "_recipe.get(\"relation\") == \"beside\"):\n"
                "                _pl[\"beside_landmark_type\"] = _lm\n"
                "            spec[\"placement\"] = _pl\n"
                "            _recipe_target_placed = True\n"
            )
            _nt_a = NEW_COMBOS.find(
                "            _beside_lm = _lm if (\n")
            _nt_b = NEW_COMBOS.find(
                "        elif _recipe is not None:\n")
            if _old_tgt in src and _nt_a >= 0 and _nt_b > _nt_a:
                src = src.replace(
                    _old_tgt,
                    NEW_COMBOS[_nt_a:_nt_b],
                    1,
                )
                changed = True
            src = src.replace(
                '                "recipe_id": _recipe.get("id"),\n'
                "            }\n"
                "        elif _above_mode and _target and not _climber_placed:\n",
                "            }\n"
                "        elif _above_mode and _target and not _climber_placed:\n",
                1,
            )
        if ('"surface_anchor": rec.get("surface_anchor"),' in src
                and '"extra_landmarks": list(rec.get("extra_landmarks")'
                not in src):
            src = src.replace(
                '"surface_anchor": rec.get("surface_anchor"),\n',
                '"surface_anchor": rec.get("surface_anchor"),\n'
                '                    "extra_landmarks": list('
                'rec.get("extra_landmarks") or []),\n',
                1,
            )
            changed = True
        if "TRAINER_SAR_RECIPES" not in src and "_pose_set = set(all_poses)" in src:
            _shuf_if = "    if (_above_mode or _recipe is not None) and combos:\n"
            _old_pose_set = "    _pose_set = set(all_poses)\n    if _above_mode and combos:\n"
            _new_pose_set = NEW_COMBOS[
                NEW_COMBOS.find("    _pose_set = set(all_poses)\n"):
                NEW_COMBOS.find(_shuf_if) + len(_shuf_if)
            ]
            if _old_pose_set in src and _new_pose_set:
                src = src.replace(_old_pose_set, _new_pose_set, 1)
                changed = True
            _old_loop = (
                "    _climber_placed = False\n"
                "    for pose, ch in combos:\n"
                "        spec = {\"type\": pose, \"character\": ch, \"count\": 1}\n"
                "        if _above_mode and _target and not _climber_placed:\n"
            )
            _new_loop = NEW_COMBOS[
                NEW_COMBOS.find("    _climber_placed = False\n"):
                NEW_COMBOS.find(
                    "        elif _above_mode and _target and not _climber_placed:\n"
                ) + len(
                    "        elif _above_mode and _target and not _climber_placed:\n"
                )
            ]
            if _old_loop in src and _new_loop:
                src = src.replace(_old_loop, _new_loop, 1)
                changed = True
        if "def _marc_sar_recipe_cursor_start():" not in src and (
                "def _marc_sar_recipe_list():" in src):
            helper = NEW_COMBOS[
                NEW_COMBOS.find("    def _marc_sar_recipe_cursor_start():"):
                NEW_COMBOS.find("    def _marc_sar_recipe_list():")
            ]
            if helper:
                src = src.replace(
                    "    def _marc_sar_recipe_list():",
                    helper + "    def _marc_sar_recipe_list():",
                    1,
                )
                changed = True
        _old_rc = (
            "            except (NameError, TypeError, ValueError):\n"
            "                _rc = 0\n"
        )
        _new_rc = (
            "            except (NameError, TypeError, ValueError):\n"
            "                _rc = _marc_sar_recipe_cursor_start()\n"
            "                print(\"[people-patch] SAR recipe cursor start=%d (%s)\" % (\n"
            "                    _rc, _recs[_rc % len(_recs)].get(\"id\")))\n"
        )
        if _old_rc in src:
            src = src.replace(_old_rc, _new_rc, 1)
            changed = True
        _old_dump_rel = (
            '                    "relation": rec.get("relation"),\n'
            '                    "command": rec.get("command"),\n'
        )
        _new_dump_rel = (
            '                    "relation": rec.get("relation"),\n'
            '                    "above": bool(rec.get("above")),\n'
            '                    "command": rec.get("command"),\n'
        )
        if _old_dump_rel in src and '"above": bool(rec.get("above"))' not in src:
            src = src.replace(_old_dump_rel, _new_dump_rel, 1)
            changed = True
        _old_shuf = (
            "    if _above_mode and combos:\n"
            "        # Climber identity is random per scene. Round-robin locked the\n"
            "        # same mesh to the same CCTV as cameras cycled.\n"
            "        random.shuffle(combos)\n"
        )
        _new_shuf = (
            "    if (_above_mode or _recipe is not None) and combos:\n"
            "        # Climber identity is random per scene. Recipe mode used catalog\n"
            "        # order, so every stacked sitter was male_student_01.\n"
            "        random.shuffle(combos)\n"
        )
        if _old_shuf in src:
            src = src.replace(_old_shuf, _new_shuf, 1)
            changed = True
        _old_shuf_plain = (
            "    if _above_mode and combos:\n"
            "        random.shuffle(combos)\n"
        )
        _new_shuf_plain = (
            "    if (_above_mode or _recipe is not None) and combos:\n"
            "        random.shuffle(combos)\n"
        )
        if _old_shuf_plain in src:
            src = src.replace(_old_shuf_plain, _new_shuf_plain, 1)
            changed = True
        _old_extra_dump = (
            "            for _extra in _extras:\n"
            "                if _extra and not any(_marc_lm_typ(s) == _extra for s in landmarks):\n"
            "                    landmarks.append({\"name\": \"%s_0\" % _extra})\n"
            "            _marc_dump_recipe(_recipe)\n"
        )
        _new_extra_dump = NEW_COMBOS[
            NEW_COMBOS.find("            for _extra in _extras:\n"):
            NEW_COMBOS.find("            _marc_dump_recipe(_recipe)\n")
            + len("            _marc_dump_recipe(_recipe)\n")
        ]
        if (_old_extra_dump in src
                and "Extra climb cabinets sit on the FOV rim" not in src
                and _new_extra_dump):
            src = src.replace(_old_extra_dump, _new_extra_dump, 1)
            changed = True
        if "def _marc_recipe_rel(" not in src and "def _marc_dump_recipe(" in src:
            helper = NEW_COMBOS[
                NEW_COMBOS.find("    def _marc_recipe_rel("):
                NEW_COMBOS.find("    def _marc_dump_recipe(")
            ]
            if helper:
                src = src.replace(
                    "    def _marc_dump_recipe(",
                    helper + "    def _marc_dump_recipe(",
                    1,
                )
                changed = True
        _old_on_gate = (
            '            if _lm and (_recipe.get("above") or '
            '_recipe.get("relation") in (\n'
            '                    "on", "above")):\n'
        )
        _new_on_gate = (
            '            _rel = _marc_recipe_rel(_recipe)\n'
            '            if _lm and (_recipe.get("above") or '
            '_rel in ("on", "above")):\n'
        )
        if _old_on_gate in src:
            src = src.replace(_old_on_gate, _new_on_gate, 1)
            changed = True
        if ('_rel = _marc_recipe_rel(_recipe)' in src
                and '_recipe.get("relation") == "beside"' in src):
            src = src.replace(
                '_recipe.get("relation") == "beside"',
                '_rel == "beside"',
            )
            changed = True
        _old_dump_rel_raw = '"relation": rec.get("relation"),'
        _new_dump_rel_norm = '"relation": _marc_recipe_rel(rec),'
        if _old_dump_rel_raw in src:
            src = src.replace(_old_dump_rel_raw, _new_dump_rel_norm, 1)
            changed = True
        if changed:
            if not _write(src):
                return 0
            print("[people-patch] upgraded: compact climber on guaranteed target")
            _log_people()
            return 0
        print("[people-patch] already applied, skipping")
        return 0

    if V2_COMBOS in src:
        src = src.replace(V2_COMBOS, NEW_COMBOS, 1)
        if not _write(src):
            return 0
        print("[people-patch] upgraded: identities + on-landmark + above-mode")
        _log_people()
        return 0

    if V1_COMBOS in src:
        src = src.replace(V1_COMBOS, NEW_COMBOS, 1)
        if not _write(src):
            return 0
        print("[people-patch] upgraded: identities + on-landmark seating")
        _log_people()
        return 0

    if OLD_COMBOS not in src:
        print("[people-patch] people-combo anchor not found, skipping - "
              "trainer.py may have changed upstream")
        return 0

    src = src.replace(OLD_COMBOS, NEW_COMBOS, 1)
    if not _write(src):
        return 0

    _log_people()
    return 0


def _log_people():
    mode = os.environ.get("TRAINER_PEOPLE_ALL_CHARS", "1")
    on_p = os.environ.get("TRAINER_PEOPLE_ON_LANDMARK_PROB", "0.45")
    above = os.environ.get("TRAINER_PEOPLE_ABOVE", "0")
    clearance = os.environ.get("TRAINER_PEOPLE_ABOVE_CLEARANCE", "0.03")
    weights = os.environ.get("TRAINER_PEOPLE_ABOVE_WEIGHTS", "")
    chars = os.environ.get("TRAINER_PEOPLE_CHARS", "") or "*"
    poses = os.environ.get("TRAINER_PEOPLE_POSES", "") or "*"
    on_off = "ON" if str(above).strip().lower() not in (
        "0", "false", "no", "off", "") else "OFF (ground scenes, no stacking)"
    recipes = os.environ.get("TRAINER_SAR_RECIPES", "0")
    cursor = os.environ.get("TRAINER_SAR_RECIPE_CURSOR", "") or "auto"
    yolo = os.environ.get("TRAINER_YOLO_MODE", "0")
    print(f"[people-patch] all identities per scene "
          f"(TRAINER_PEOPLE_ALL_CHARS={mode}); chars={chars} poses={poses}; "
          f"on-landmark p={on_p}; "
          f"above={above} [{on_off}] clearance={clearance} weights={weights}; "
          f"sar_recipes={recipes} recipe_cursor={cursor}; "
          f"yolo_mode={yolo}")


if __name__ == "__main__":
    sys.exit(main())
