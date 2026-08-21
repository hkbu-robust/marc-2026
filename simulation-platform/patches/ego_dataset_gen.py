#!/usr/bin/env python3
# Copyright (c) 2026, IoT Convergence & Open Sharing System (IoTCOSS)
#
# MARC 2026 - ego-cam YOLO dataset generator (robot Base_Camera_Left)
# ---------------------------------------------------------------------------
# Sibling of dataset-gen (CCTV / aerial) and manip-trainer. Spawns lost objects
# in front of the robot, attaches Isaac Replicator bbox+rgb annotators to the
# chassis camera, and writes an Ultralytics YOLO dataset.
#
# Do NOT mix this with the CCTV set when training the approach detector — this
# is the view auto-approach actually sees (near-ground, ~640x480).
#
# Env:
#   EGO_SCENES          new scenes this run (appends after existing stems; default 200)
#   EGO_SEED            RNG seed (default 0; offset by resume index so reruns aren't clones)
#   EGO_OBJECTS_MIN/MAX objects per scene (default 1-3)
#   EGO_REACH_MIN/MAX   forward distance m (default 0.5-3.5)
#   EGO_LAT             lateral clamp m (default 2.5; FOV placement uses bearing)
#   EGO_BEARING_DEG     object azimuth half-FOV deg (default 50)
#   EGO_ROBOT_XY        robot wander radius around start_point m (default 4)
#   EGO_YAW_SPAN        robot yaw randomisation deg (default 360 = full turn)
#   EGO_SKIP_CLASSES    comma names to drop (default umbrella,long_umbrella)
#   EGO_EMPTY_FRAC      fraction of empty (negative) scenes (default 0.05)
#   EGO_VAL_FRAC        validation fraction (default 0.2)
#   EGO_OUT             output root (default trainer_output/ego_yolo)
#   MARC_MANIP_OBJECT   lock to one class (optional; bypasses skip list)
#   HEADLESS            default true for this service
#
import os
import sys
import math
import random
import logging
import traceback
import re

sys.path.insert(0, "/metacom2026/simulation_app/marc2026")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from isaacsim import SimulationApp

_headless = os.environ.get("HEADLESS", "true").lower() in ("1", "true", "yes")

simulation_app = SimulationApp({
    "renderer": "RayTracedLighting",
    "headless": _headless,
    "physics_dt": 1.0 / 20.0,
    "physics_substeps": 1,
    "physics_auto_apply_apis": False,
    "experience": os.path.join(
        os.environ.get("ISAAC_SIM_PATH", ""), "apps", "isaacsim.exp.full.kit"
    ),
    "extra_args": ["--/isaac/startup/create_new_stage=false"],
})
simulation_app.update()

import numpy as np
from isaacsim.core.api import World

from marc_config.configurations import Configurations
from marc_config.constants import PRIM_ROBOT, PRIM_TRASH
from world.world_setup import (
    setup_world_stage, init_world_stage, setup_friction_material, _resume_rendering,
)
from world.object_deployer import deploy_mission_objects
from robot.robot_loader import deploy_robot
from runtime.robot_arm_driver import ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES
from trainer.ground_truth import GroundTruthGenerator

PHYSICS_DT = 1.0 / 20.0


def _apply_initial_arm_pose(stage):
    from pxr import Usd
    root = stage.GetPrimAtPath(PRIM_ROBOT)
    if not root or not root.IsValid():
        return
    names = set(ARM_JOINT_NAMES) | set(GRIPPER_JOINT_NAMES)
    for prim in Usd.PrimRange(root):
        if prim.GetName() not in names:
            continue
        tgt = prim.GetAttribute("drive:angular:physics:targetPosition")
        st = prim.GetAttribute("state:angular:physics:position")
        if tgt and tgt.HasValue() and st:
            st.Set(float(tgt.Get()))


def _resolve_robot_camera(stage, cam_name):
    from pxr import Usd
    root = stage.GetPrimAtPath(PRIM_ROBOT)
    if not root or not root.IsValid():
        return None
    for prim in Usd.PrimRange(root):
        if prim.GetName() == cam_name:
            return str(prim.GetPath())
    return None


def _world_pos(stage, prim_path):
    from pxr import Usd, UsdGeom
    if not prim_path:
        return None
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    t = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()).ExtractTranslation()
    return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=float)


class EgoScene:
    """Robot frame + spawn helpers (subset of ManipulationEnv, no trainer import)."""

    def __init__(self, cfg, sim_ctx, stage, start_point, robot_yaw):
        self.cfg = cfg
        self.sim = sim_ctx
        self.stage = stage
        self.start_xy = (float(start_point[0]), float(start_point[1]))
        self.start_yaw_deg = float(robot_yaw)
        self.spawn_height = float(cfg.get_robot_spawn_height())
        self.base_xy = self.start_xy
        self.base_yaw_deg = self.start_yaw_deg
        self.ground_z = float(start_point[2])
        self.arm_paths = self._find_joints(ARM_JOINT_NAMES)
        self.grip_paths = self._find_joints(GRIPPER_JOINT_NAMES)
        self.home_deg = [
            self._read_deg(self.arm_paths.get(n)) or 0.0 for n in ARM_JOINT_NAMES
        ]
        self.grip_open_deg = 0.0
        self._fdir_cache = None
        self._link_cache = None
        self._art = None
        skip = _skip_classes()
        all_types = [
            t for t in cfg.get_supported_object_types()
            if not cfg.is_person_type(t) and not _is_skipped(t, skip)
        ]
        fixed = os.environ.get("MARC_MANIP_OBJECT")
        if fixed:
            self.object_types = [fixed]
        else:
            self.object_types = all_types
        if not self.object_types:
            raise RuntimeError("no graspable object types in config")

    def _find_joints(self, names):
        from pxr import Usd
        found = {}
        root = self.stage.GetPrimAtPath(PRIM_ROBOT)
        if root and root.IsValid():
            nameset = set(names)
            for prim in Usd.PrimRange(root):
                n = prim.GetName()
                if n in nameset and n not in found:
                    found[n] = str(prim.GetPath())
        return found

    def _set_drive_deg(self, path, deg):
        from pxr import UsdPhysics
        prim = self.stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if drive and drive.GetTargetPositionAttr():
            drive.GetTargetPositionAttr().Set(float(deg))

    def _read_deg(self, path):
        if not path:
            return None
        prim = self.stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return None
        attr = prim.GetAttribute("state:angular:physics:position")
        if attr and attr.HasValue():
            return float(attr.Get())
        return None

    def home_arm(self):
        for i, n in enumerate(ARM_JOINT_NAMES):
            p = self.arm_paths.get(n)
            if p is not None:
                self._set_drive_deg(p, self.home_deg[i])
        for n in GRIPPER_JOINT_NAMES:
            p = self.grip_paths.get(n)
            if p is not None:
                self._set_drive_deg(p, self.grip_open_deg)

    def _link_world(self, link_name):
        if self._link_cache is None:
            from pxr import Usd
            self._link_cache = {}
            root = self.stage.GetPrimAtPath(PRIM_ROBOT)
            if root and root.IsValid():
                for prim in Usd.PrimRange(root):
                    if prim.GetName() in ("left_finger_pad", "right_finger_pad",
                                          "arm_base_link"):
                        self._link_cache[prim.GetName()] = str(prim.GetPath())
        p = self._link_cache.get(link_name)
        return _world_pos(self.stage, p) if p else None

    def _ee_world(self):
        lf = self._link_world("left_finger_pad")
        rf = self._link_world("right_finger_pad")
        if lf is not None and rf is not None:
            return (lf + rf) / 2.0
        return lf if lf is not None else self._link_world("arm_base_link")

    def _robot_frame(self):
        ab = self._link_world("arm_base_link")
        origin = (np.array([float(ab[0]), float(ab[1])]) if ab is not None
                  else np.array([self.base_xy[0], self.base_xy[1]]))
        fdir = self._fdir_cache
        if fdir is None:
            ee = self._ee_world()
            if ab is not None and ee is not None:
                d = np.array([float(ee[0] - ab[0]), float(ee[1] - ab[1])])
                if np.linalg.norm(d) > 0.05:
                    fdir = d / np.linalg.norm(d)
            if fdir is None:
                yaw = math.radians(self.base_yaw_deg)
                fdir = np.array([math.cos(yaw), math.sin(yaw)])
            self._fdir_cache = fdir
        return (origin, fdir, np.array([fdir[1], -fdir[0]]))

    def _ground_z(self, x, y):
        import carb
        try:
            from omni.physx import get_physx_scene_query_interface
            sq = get_physx_scene_query_interface()
        except Exception:
            return None
        origin_z = self.ground_z + 5.0
        for _ in range(8):
            hit = sq.raycast_closest(
                carb.Float3(float(x), float(y), float(origin_z)),
                carb.Float3(0.0, 0.0, -1.0), 60.0)
            if not hit or not hit.get("hit"):
                return None
            path = str(hit.get("rigidBody") or hit.get("collision") or "")
            hz = hit["position"][2]
            if path.startswith(PRIM_ROBOT) or path.startswith(PRIM_TRASH):
                origin_z = hz - 0.05
                continue
            return hz
        return None

    def _harden_object(self, prim_path):
        from pxr import Usd, UsdPhysics, PhysxSchema
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return
        for p in Usd.PrimRange(prim):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                rb = PhysxSchema.PhysxRigidBodyAPI.Apply(p)
                rb.CreateMaxDepenetrationVelocityAttr(1.0)
                rb.CreateMaxLinearVelocityAttr(5.0)

    def _articulation(self):
        if self._art is None:
            from isaacsim.core.prims import SingleArticulation
            path = PRIM_ROBOT
            try:
                from runtime.robot_control_graph import find_chassis_articulation_root
                path = find_chassis_articulation_root(PRIM_ROBOT) or PRIM_ROBOT
            except Exception:
                pass
            art = SingleArticulation(prim_path=path)
            art.initialize()
            self._art = art
        return self._art

    def teleport(self, x, y, yaw_deg):
        """Move the chassis in physics (USD xform on the robot root is static)."""
        gz = self._ground_z(x, y)
        z = float((gz if gz is not None else self.ground_z) + self.spawn_height)
        half = math.radians(float(yaw_deg)) * 0.5
        quat = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])  # (w,x,y,z)
        art = self._articulation()
        art.set_world_pose(
            position=np.array([float(x), float(y), z]),
            orientation=quat,
        )
        try:
            art.set_linear_velocity(np.zeros(3))
            art.set_angular_velocity(np.zeros(3))
        except Exception:
            pass
        self.base_xy = (float(x), float(y))
        self.base_yaw_deg = float(yaw_deg) % 360.0
        if gz is not None:
            self.ground_z = float(gz)
        self._fdir_cache = None

_log_handler = logging.StreamHandler(sys.stdout)
_log_handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
log = logging.getLogger("marc.ego_dataset")
log.setLevel(logging.INFO)
log.addHandler(_log_handler)
log.propagate = False

# Keep class ids stable and aligned with the CCTV / approach object list.
# Umbrellas are skipped: long thin silhouettes dominate the frame and poison
# the approach detector. Override via EGO_SKIP_CLASSES.
CANONICAL_NAMES = [
    "tumbler", "cracker_box", "cola_can", "disposable_cup",
    "juice", "tissue", "mug", "sunblock", "pencilcase",
]
DEFAULT_SKIP = ("umbrella", "long_umbrella")

CAM_W, CAM_H = 640, 480
MIN_BOX_PX = 8
# Object azimuth bins as a fraction of EGO_BEARING_DEG, so boxes land left /
# center / right of the image instead of clustering on the optical axis.
_BEARING_BINS = (-0.9, -0.55, -0.25, 0.0, 0.25, 0.55, 0.9)


def _skip_classes():
    raw = os.environ.get("EGO_SKIP_CLASSES")
    names = raw.split(",") if raw else list(DEFAULT_SKIP)
    skip = {n.strip() for n in names if n.strip()}
    # Any type with "umbrella" in the name is also dropped (closedlongumbrella, …).
    return skip


def _is_skipped(name, skip):
    if name in skip:
        return True
    if "umbrella" in name.lower() and any("umbrella" in s.lower() for s in skip):
        return True
    return False


def _env_int(name, default):
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _build_names(object_types):
    names = []
    for n in CANONICAL_NAMES:
        if n in object_types and n not in names:
            names.append(n)
    for n in object_types:
        if n not in names:
            names.append(n)
    return names


def _plan_scene_classes(rng, names, n_scenes, obj_min, obj_max, empty_frac):
    """Per-scene class lists: every name appears at least once; extras are shuffled.

    Empty (negative) scenes are kept only if remaining content scenes can
    still cover the catalogue. obj_max may be exceeded if that is the only
    way to fit every class.
    """
    n_cls = len(names)
    n_empty = int(round(n_scenes * empty_frac)) if names else int(round(n_scenes * empty_frac))
    if n_cls:
        n_empty = min(n_empty, max(0, n_scenes - math.ceil(n_cls / max(1, obj_max))))
    empty_idx = set(rng.sample(range(n_scenes), n_empty)) if n_empty else set()
    content = [i for i in range(n_scenes) if i not in empty_idx]
    if not content:
        return [[] for _ in range(n_scenes)]
    cap = max(1, obj_max)
    counts = [max(0, obj_min)] * len(content)
    need = max(0, n_cls - sum(counts))
    for i in range(len(content)):
        while need > 0 and counts[i] < cap:
            counts[i] += 1
            need -= 1
    i = 0
    while need > 0:
        counts[i % len(content)] += 1
        need -= 1
        i += 1
    for i in range(len(content)):
        room = cap - counts[i]
        if room > 0:
            counts[i] += rng.randint(0, room)
    n_slots = sum(counts)
    bag = []
    if names:
        must = list(names)
        rng.shuffle(must)
        bag.extend(must)
        while len(bag) < n_slots:
            extra = list(names)
            rng.shuffle(extra)
            bag.extend(extra)
        bag = bag[:n_slots]
        rng.shuffle(bag)
    plan = [[] for _ in range(n_scenes)]
    it = iter(bag)
    for scene_i, n_obj in zip(content, counts):
        plan[scene_i] = [next(it) for _ in range(n_obj)]
    return plan


def _sample_yaw(rng, base_yaw, span):
    span = max(0.0, float(span))
    if span >= 359.0:
        return rng.uniform(0.0, 360.0)
    return (float(base_yaw) + rng.uniform(-0.5 * span, 0.5 * span)) % 360.0


def _sample_robot_xy(env, rng, radius, tries=10):
    if radius <= 1e-3:
        return env.start_xy
    ox, oy = env.start_xy
    for _ in range(tries):
        ang = rng.uniform(0.0, 2.0 * math.pi)
        r = radius * math.sqrt(rng.random())
        x = ox + r * math.cos(ang)
        y = oy + r * math.sin(ang)
        if env._ground_z(x, y) is not None:
            return (x, y)
    return env.start_xy


def _sample_object_offset(rng, reach_min, reach_max, lat_max, bearing_deg):
    """fwd/lat in the robot frame so boxes spread across the image, not the axis."""
    span = max(0.05, reach_max - reach_min)
    u = rng.random()
    if u < 0.4:
        fwd = rng.uniform(reach_min, reach_min + 0.35 * span)
    elif u < 0.8:
        fwd = rng.uniform(reach_min + 0.25 * span, reach_min + 0.7 * span)
    else:
        fwd = rng.uniform(reach_min + 0.55 * span, reach_max)
    frac = rng.choice(_BEARING_BINS) + rng.uniform(-0.08, 0.08)
    bearing = math.radians(bearing_deg * max(-1.0, min(1.0, frac)))
    lat = fwd * math.tan(bearing)
    lat = max(-lat_max, min(lat_max, lat))
    return fwd, lat


def _clear_trash(stage):
    trash = stage.GetPrimAtPath(PRIM_TRASH)
    if trash and trash.IsValid():
        stage.RemovePrim(PRIM_TRASH)
        simulation_app.update()


def _spawn_objects(env, rng, class_list, reach_min, reach_max, lat_max, bearing_deg):
    """Place the given classes on the ground in front of the current robot pose."""
    _clear_trash(env.stage)
    if not class_list:
        return []
    origin, fdir, rdir = env._robot_frame()
    objects = []
    placed = []
    xs, ys = [], []
    for i, otype in enumerate(class_list):
        fwd, lat = _sample_object_offset(
            rng, reach_min, reach_max, lat_max, bearing_deg)
        xy = origin + fdir * fwd + rdir * lat
        x, y = float(xy[0]), float(xy[1])
        gz = env._ground_z(x, y)
        z = float((gz if gz is not None else env.ground_z) + 0.02)
        rot = [0.0, 0.0, rng.uniform(-180.0, 180.0)]
        prim_path = f"{PRIM_TRASH}/ego_{i}_{otype}"
        objects.append({
            "type": otype,
            "count": 1,
            "resolved": {"instances": [{
                "prim_path": prim_path,
                "position": [x, y, z],
                "rotation": rot,
            }]},
        })
        placed.append((otype, prim_path, fwd, lat))
        xs.append(x)
        ys.append(y)
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    z0 = float(env.ground_z)
    area = {
        "top_left": [cx - 2, cy + 2, z0],
        "top_right": [cx + 2, cy + 2, z0],
        "bottom_right": [cx + 2, cy - 2, z0],
        "bottom_left": [cx - 2, cy - 2, z0],
    }
    deploy_mission_objects("ego", objects, area, env.cfg)
    for _, prim_path, _, _ in placed:
        env._harden_object(prim_path)
    simulation_app.update()
    return placed


def _save_rgb(arr, path):
    rgb = np.asarray(arr)
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    if rgb.dtype != np.uint8:
        mx = float(np.max(rgb)) if rgb.size else 0.0
        if mx <= 1.5:
            rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(rgb).save(path)
        return True
    except Exception:
        try:
            import imageio.v2 as imageio
            imageio.imwrite(path, rgb)
            return True
        except Exception as e:
            log.warning(f"RGB save failed {path}: {e}")
            return False


_STEM_RE = re.compile(r"^(\d+)")


def _next_image_index(out_dir):
    """One past the highest existing 00000.jpg/.png/.txt stem. Reruns only append."""
    highest = -1
    for split in ("train", "val"):
        for sub in ("images", "labels", "overlays"):
            folder = os.path.join(out_dir, sub, split)
            if not os.path.isdir(folder):
                continue
            try:
                names = os.listdir(folder)
            except OSError:
                continue
            for name in names:
                m = _STEM_RE.match(name)
                if m:
                    highest = max(highest, int(m.group(1)))
    return highest + 1


def _stem_taken(out_dir, stem):
    for split in ("train", "val"):
        for sub, exts in (
            ("images", (".png", ".jpg", ".jpeg")),
            ("labels", (".txt",)),
        ):
            folder = os.path.join(out_dir, sub, split)
            for ext in exts:
                if os.path.isfile(os.path.join(folder, stem + ext)):
                    return True
    return False


def _write_yolo(label_path, detections, names, width, height):
    name_to_id = {n: i for i, n in enumerate(names)}
    lines = []
    kept = 0
    for det in detections:
        cls = det.get("class")
        if cls not in name_to_id:
            continue
        x0, y0, x1, y1 = det["bbox"]
        x0 = max(0.0, min(float(width), float(x0)))
        x1 = max(0.0, min(float(width), float(x1)))
        y0 = max(0.0, min(float(height), float(y0)))
        y1 = max(0.0, min(float(height), float(y1)))
        bw, bh = x1 - x0, y1 - y0
        if bw < MIN_BOX_PX or bh < MIN_BOX_PX:
            continue
        cx = (x0 + x1) * 0.5 / float(width)
        cy = (y0 + y1) * 0.5 / float(height)
        nw = bw / float(width)
        nh = bh / float(height)
        cx = min(1.0, max(0.0, cx))
        cy = min(1.0, max(0.0, cy))
        nw = min(1.0, max(1e-6, nw))
        nh = min(1.0, max(1e-6, nh))
        lines.append(f"{name_to_id[cls]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        kept += 1
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return kept


def _write_data_yaml(out_dir, names):
    path = os.path.join(out_dir, "data.yaml")
    lines = [
        f"path: {out_dir}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for i, n in enumerate(names):
        lines.append(f"  {i}: {n}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main():
    n_scenes = max(1, _env_int("EGO_SCENES", 200))
    seed = _env_int("EGO_SEED", 0)
    obj_min = max(0, _env_int("EGO_OBJECTS_MIN", 1))
    obj_max = max(obj_min, _env_int("EGO_OBJECTS_MAX", 3))
    reach_min = _env_float("EGO_REACH_MIN", 0.5)
    reach_max = max(reach_min + 0.05, _env_float("EGO_REACH_MAX", 3.5))
    lat_max = max(0.1, _env_float("EGO_LAT", 2.5))
    bearing_deg = max(5.0, _env_float("EGO_BEARING_DEG", 50.0))
    robot_xy_r = max(0.0, _env_float("EGO_ROBOT_XY", 4.0))
    yaw_span = max(0.0, _env_float("EGO_YAW_SPAN", 360.0))
    empty_frac = min(1.0, max(0.0, _env_float("EGO_EMPTY_FRAC", 0.05)))
    val_frac = min(0.5, max(0.05, _env_float("EGO_VAL_FRAC", 0.2)))
    out_dir = os.environ.get(
        "EGO_OUT",
        os.path.join(os.environ.get("MARC_TRAINER_DIR") or "/metacom2026/trainer_output",
                     "ego_yolo"),
    )
    start_idx = _next_image_index(out_dir)
    rng = random.Random(seed + start_idx)
    val_period = max(2, int(round(1.0 / val_frac)))

    print("=" * 60)
    print("  MARC 2026 ego-cam YOLO generator (Base_Camera_Left)")
    print(f"  new_scenes={n_scenes}  resume_index={start_idx:05d}  format=png")
    print(f"  objects/scene={obj_min}-{obj_max}  "
          f"reach={reach_min:.2f}-{reach_max:.2f}m  "
          f"robot_xy=±{robot_xy_r:.1f}m  yaw_span={yaw_span:.0f}°")
    print(f"  out={out_dir}")
    print("=" * 60, flush=True)
    if start_idx > 0:
        log.info(f"appending after existing stems (next={start_idx:05d}); "
                 f"prior JPEG/PNG kept. seed offset={seed + start_idx}")

    try:
        cfg = Configurations()
    except Exception:
        traceback.print_exc()
        simulation_app.close()
        return 1

    setup_world_stage(*cfg.get_view_point(), cfg.get_playground_usd_path(),
                      app=simulation_app)
    simulation_app.update()
    init_world_stage()
    simulation_app.update()
    setup_friction_material()
    simulation_app.update()

    start_point = cfg.get_robot_start_point()
    if not start_point:
        log.error("scenario has no robot.start_point")
        simulation_app.close()
        return 1
    robot_yaw = float((cfg.scenario_data.get("robot") or {}).get("yaw", 0.0))
    deploy_robot(cfg.get_robot_usd_path(), start_point, yaw=robot_yaw,
                 configurations=cfg, draw_station_decal=False)
    simulation_app.update()

    import omni.usd
    stage = omni.usd.get_context().get_stage()
    _apply_initial_arm_pose(stage)
    simulation_app.update()
    _resume_rendering()
    simulation_app.update()

    sim_ctx = World(physics_dt=PHYSICS_DT)
    sim_ctx.initialize_physics()
    sim_ctx.play()
    for _ in range(60):
        sim_ctx.step(render=True)

    stage = omni.usd.get_context().get_stage()
    env = EgoScene(cfg, sim_ctx, stage, start_point, robot_yaw)
    env.home_arm()
    env._robot_frame()  # cache forward while arm is at home

    cam_path = _resolve_robot_camera(stage, "Base_Camera_Left")
    if not cam_path:
        log.error("Base_Camera_Left prim not found under the robot")
        simulation_app.close()
        return 1
    log.info(f"GT camera: {cam_path}")

    names = _build_names(env.object_types)
    os.makedirs(out_dir, exist_ok=True)
    yaml_path = _write_data_yaml(out_dir, names)
    log.info(f"classes ({len(names)}): {', '.join(names)}")
    log.info(f"skip: {sorted(_skip_classes())}")
    scene_classes = _plan_scene_classes(
        rng, names, n_scenes, obj_min, obj_max, empty_frac)

    missions = [{"id": "base_left", "_camera_prim_path": cam_path}]
    gt = GroundTruthGenerator(missions, out_dir, resolution=(CAM_W, CAM_H))
    if not gt.setup():
        log.error("GroundTruthGenerator setup failed")
        simulation_app.close()
        return 1

    n_ok = 0
    n_empty_ok = 0
    n_skip = 0
    spawned_counts = {n: 0 for n in names}
    labeled_counts = {n: 0 for n in names}
    idx = start_idx
    for i in range(n_scenes):
        if not simulation_app.is_running():
            break
        class_list = list(scene_classes[i])
        empty = not class_list
        try:
            xy = _sample_robot_xy(env, rng, robot_xy_r)
            yaw = _sample_yaw(rng, env.start_yaw_deg, yaw_span)
            env.teleport(xy[0], xy[1], yaw)
            env.home_arm()
            for _ in range(20):
                sim_ctx.step(render=True)
            placed = _spawn_objects(
                env, rng, class_list,
                reach_min, reach_max, lat_max, bearing_deg,
            )
            env.home_arm()
            for _ in range(40):
                sim_ctx.step(render=True)
            if not sim_ctx.is_playing():
                sim_ctx.play()
            payloads = gt.capture(
                i, simulation_app,
                render_step=lambda: sim_ctx.step(render=True),
            )
        except Exception as e:
            log.error(f"scene {i}: {e}")
            traceback.print_exc()
            n_skip += 1
            continue

        payload = payloads.get("base_left") or {}
        if not gt.last_render_ok:
            log.warning(f"scene {i}: black frame, skip")
            n_skip += 1
            continue

        annot = gt._rgb_annotators.get("base_left")
        rgb = None
        if annot is not None:
            raw = annot.get_data()
            rgb = raw.get("data") if isinstance(raw, dict) else raw
        if rgb is None:
            log.warning(f"scene {i}: no RGB, skip")
            n_skip += 1
            continue

        split = "val" if (idx % val_period == 0) else "train"
        while _stem_taken(out_dir, f"{idx:05d}"):
            idx += 1
        stem = f"{idx:05d}"
        img_path = os.path.join(out_dir, "images", split, f"{stem}.png")
        lbl_path = os.path.join(out_dir, "labels", split, f"{stem}.txt")
        if not _save_rgb(rgb, img_path):
            n_skip += 1
            continue
        kept = _write_yolo(lbl_path, payload.get("detections") or [], names, CAM_W, CAM_H)
        for otype, _, _, _ in placed:
            if otype in spawned_counts:
                spawned_counts[otype] += 1
        for det in payload.get("detections") or []:
            cls = det.get("class")
            if cls in labeled_counts:
                labeled_counts[cls] += 1
        if os.environ.get("EGO_SAVE_OVERLAY", "1") != "0":
            ov = os.path.join(out_dir, "overlays", split, f"{stem}.png")
            try:
                from PIL import Image, ImageDraw
                im = Image.fromarray(np.asarray(rgb)[..., :3])
                dr = ImageDraw.Draw(im)
                for det in payload.get("detections") or []:
                    if det.get("class") not in names:
                        continue
                    x0, y0, x1, y1 = det["bbox"]
                    dr.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=2)
                    dr.text((x0 + 2, max(0, y0 - 11)), str(det.get("class", "")),
                            fill=(255, 255, 0))
                os.makedirs(os.path.dirname(ov), exist_ok=True)
                im.save(ov)
            except Exception:
                pass
        n_ok += 1
        if empty:
            n_empty_ok += 1
        kinds = ",".join(p[0] for p in placed) or "-"
        log.info(f"scene {stem} {split} boxes={kept} types=[{kinds}] "
                 f"xy=({xy[0]:.2f},{xy[1]:.2f}) yaw={yaw:.0f}")
        idx += 1

    missing_spawn = [n for n in names if spawned_counts[n] == 0]
    missing_label = [n for n in names if labeled_counts[n] == 0]
    spawn_s = " ".join(f"{n}={spawned_counts[n]}" for n in names)
    label_s = " ".join(f"{n}={labeled_counts[n]}" for n in names)
    log.info(f"spawned: {spawn_s}")
    log.info(f"labeled: {label_s}")
    if missing_spawn:
        log.error(f"classes never spawned: {missing_spawn}")
    if missing_label:
        log.warning(f"classes never labeled (spawned but no box?): {missing_label}")
    print(f"[ego-gen] done  kept={n_ok}/{n_scenes}  empty={n_empty_ok}  "
          f"skipped={n_skip}  yaml={yaml_path}", flush=True)
    try:
        gt.destroy()
    except Exception:
        pass
    sim_ctx.stop()
    simulation_app.close()
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
