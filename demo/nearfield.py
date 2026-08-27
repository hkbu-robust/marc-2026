"""Near-field RGB-D reacquisition for Stage 2 pick.

After D* standoff, ``NearFieldAcquire.acquire(reveal, deadline)`` searches inside
the reveal disc, closes on robust bbox depth, and returns a validated arm_base
3D target. PathFollower / occupancy / grasp waypoints stay elsewhere.

Callers never pass YOLO or depth knobs. Missing ultralytics -> ``valid=False``
(fail closed; do not pick from hint XY).
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

log = logging.getLogger("participant_app")

ARM_BASE_FWD = 0.18
CAMERA = "base_left"
DEPTH_WHICH = "base"
TICK_S = 0.10
LOCK_FRAMES = 12
CRAWL_MPS = 0.15
MAX_STEP_M = 0.04
STOP_M = 0.55
ALREADY_NEAR_M = 0.45
CENTER_TOL = 0.05
REALIGN = 0.12
ALIGN_HOLD = 2
ALIGN_SETTLE_S = 0.15
BURST_M = 0.15
YAW_GAIN = 1.2
YAW_MAX = 0.35
YAW_MIN = 0.10
SCAN_YAW = 0.30
SCAN_FLIP_S = 3.0
DEPTH_PERCENTILE = 30.0
DEPTH_MIN_M = 0.05
DEPTH_MAX_M = 20.0
DEPTH_MIN_VALID = 4
BBOX_INSET = 0.10
RADIUS_SLACK_M = 0.20
FWD_MIN_M = 0.15
FWD_MAX_M = 1.00
LAT_MAX_M = 0.45
MIN_CONF = 0.12
STALE_STOP_S = 0.8

# MARC class -> COCO names used only while the checkpoint lacks the MARC label.
COCO_ALIASES = {
    "tumbler": ("cup", "bottle", "wine glass"),
    "mug": ("cup",),
    "disposable_cup": ("cup", "bottle"),
    "cola_can": ("bottle", "cup"),
    "juice": ("bottle", "cup"),
    "cracker_box": ("book",),
    "tissue": ("book",),
    "sunblock": ("bottle",),
    "pencilcase": ("book", "cell phone"),
}


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Depth inside an RGB box (copied so tests never import torch / ROS).
# ---------------------------------------------------------------------------

def depth_in_bbox(
    depth: np.ndarray,
    bbox_xyxy,
    rgb_shape,
    percentile: float = DEPTH_PERCENTILE,
    min_m: float = DEPTH_MIN_M,
    max_m: float = DEPTH_MAX_M,
    min_valid: int = DEPTH_MIN_VALID,
    inset: float = BBOX_INSET,
) -> Optional[float]:
    """Near-biased depth (m) inside an RGB bbox, mapped onto the depth image."""
    if depth is None or depth.size == 0 or bbox_xyxy is None or rgb_shape is None:
        return None
    h_d, w_d = depth.shape[:2]
    h_r, w_r = int(rgb_shape[0]), int(rgb_shape[1])
    if h_d < 2 or w_d < 2 or h_r < 2 or w_r < 2:
        return None
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    sx = w_d / float(w_r)
    sy = h_d / float(h_r)
    dx1, dy1, dx2, dy2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
    bw, bh = dx2 - dx1, dy2 - dy1
    if bw < 2.0 or bh < 2.0:
        return None
    pad_x = float(np.clip(inset, 0.0, 0.45)) * bw
    pad_y = float(np.clip(inset, 0.0, 0.45)) * bh
    ix1 = int(np.clip(np.floor(dx1 + pad_x), 0, w_d - 1))
    iy1 = int(np.clip(np.floor(dy1 + pad_y), 0, h_d - 1))
    ix2 = int(np.clip(np.ceil(dx2 - pad_x), ix1 + 1, w_d))
    iy2 = int(np.clip(np.ceil(dy2 - pad_y), iy1 + 1, h_d))
    crop = depth[iy1:iy2, ix1:ix2]
    valid = crop[np.isfinite(crop) & (crop > float(min_m)) & (crop < float(max_m))]
    if valid.size < max(1, int(min_valid)):
        return None
    p = float(np.clip(percentile, 0.0, 100.0))
    return float(np.percentile(valid, p))


def _image_to_rgb(msg) -> Optional[np.ndarray]:
    if msg is None:
        return None
    h, w = int(msg.height), int(msg.width)
    if h <= 0 or w <= 0:
        return None
    enc = (getattr(msg, "encoding", "") or "").lower()
    step = int(getattr(msg, "step", 0) or 0)

    def _hwc(nchan: int, dtype=np.uint8):
        raw = np.frombuffer(msg.data, dtype=dtype)
        row = step if step else (w * nchan * np.dtype(dtype).itemsize)
        item = np.dtype(dtype).itemsize
        row_elems = max(nchan * w, row // item)
        need = h * row_elems
        if raw.size < h * nchan * w:
            return None
        if row_elems == nchan * w and raw.size >= h * nchan * w:
            return raw[: h * nchan * w].reshape(h, w, nchan)
        if raw.size < need:
            return raw[: h * nchan * w].reshape(h, w, nchan)
        return raw[:need].reshape(h, row_elems)[:, : nchan * w].reshape(h, w, nchan)

    if enc in ("rgb8", "rgb"):
        arr = _hwc(3)
        return None if arr is None else np.ascontiguousarray(arr)
    if enc in ("bgr8", "bgr", "8uc3"):
        arr = _hwc(3)
        return None if arr is None else np.ascontiguousarray(arr[:, :, ::-1])
    return None


def _image_to_depth(msg) -> Optional[np.ndarray]:
    if msg is None:
        return None
    h, w = int(msg.height), int(msg.width)
    if h <= 0 or w <= 0:
        return None
    enc = (getattr(msg, "encoding", "") or "").lower()
    if enc in ("32fc1", "32fc"):
        return np.ascontiguousarray(np.frombuffer(msg.data, dtype=np.float32).reshape(h, w))
    if enc in ("16uc1", "mono16"):
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w).astype(np.float32)
        return np.ascontiguousarray(arr * 0.001)
    return None


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def inside_reveal_disc(xy, hint_xy, radius, slack=RADIUS_SLACK_M) -> bool:
    if xy is None or hint_xy is None:
        return True
    r = float(radius or 0.0) + float(slack)
    if r <= slack and (radius is None or float(radius) <= 0.0):
        return True
    return math.hypot(float(xy[0]) - float(hint_xy[0]),
                      float(xy[1]) - float(hint_xy[1])) <= r


def yaw_toward_hint(pose, hint_xy) -> float:
    x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    err = _wrap_pi(math.atan2(float(hint_xy[1]) - y, float(hint_xy[0]) - x) - yaw)
    return _clamp(YAW_GAIN * err, -YAW_MAX, YAW_MAX)


def gated_cmd_vel(pose, linear, angular, hint_xy, hint_radius, dt=TICK_S,
                  slack=RADIUS_SLACK_M):
    """Block motion that would leave the reveal disc; allow crawl *toward* the center."""
    lin = float(linear)
    ang = float(angular)
    if pose is None or hint_xy is None:
        return max(0.0, lin), ang
    radius = float(hint_radius or 0.0)
    if radius <= 0.0:
        return max(0.0, lin), ang
    x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    cx, cy = float(hint_xy[0]), float(hint_xy[1])
    dist = math.hypot(x - cx, y - cy)
    if not inside_reveal_disc((x, y), hint_xy, radius, slack=slack):
        lin = max(0.0, lin)
        nx = x + lin * float(dt) * math.cos(yaw)
        ny = y + lin * float(dt) * math.sin(yaw)
        closer = math.hypot(nx - cx, ny - cy) < dist - 1e-6
        if lin > 1e-6 and closer:
            return lin, ang
        return 0.0, yaw_toward_hint(pose, hint_xy)
    lin = max(0.0, lin)
    if lin <= 1e-6:
        return 0.0, ang
    nx = x + lin * float(dt) * math.cos(yaw)
    ny = y + lin * float(dt) * math.sin(yaw)
    if not inside_reveal_disc((nx, ny), hint_xy, radius, slack=slack):
        if math.hypot(nx - cx, ny - cy) < dist - 1e-6:
            return lin, ang
        return 0.0, ang
    return lin, ang


def optical_xyz(u, v, z, fx, fy, cx, cy):
    """Camera-optical 3D (X right, Y down, Z forward) from pixel + depth."""
    z = float(z)
    return ((float(u) - float(cx)) / float(fx) * z,
            (float(v) - float(cy)) / float(fy) * z,
            z)


def optical_to_car_base(x, y, z, R=None, t=None):
    """Optical point -> car_base (fwd, lat, up).

    Default fallback is the REP-103 mapping ``[Z, -X, -Y]`` (camera looking
    along chassis +X). Optional ``R`` (3x3) and ``t`` (3,) override that.
    """
    p = np.array([float(x), float(y), float(z)], dtype=np.float64)
    if R is not None:
        Rm = np.asarray(R, dtype=np.float64).reshape(3, 3)
        tm = np.zeros(3) if t is None else np.asarray(t, dtype=np.float64).reshape(3)
        q = Rm @ p + tm
        return float(q[0]), float(q[1]), float(q[2])
    return float(z), float(-x), float(-y)


def car_to_arm_base(fwd, lat, up, arm_base_fwd=ARM_BASE_FWD):
    return float(fwd) - float(arm_base_fwd), float(lat), float(up)


def car_to_world_xy(pose, fwd, lat):
    x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    cc, ss = math.cos(yaw), math.sin(yaw)
    return (x + fwd * cc - lat * ss, y + fwd * ss + lat * cc)


def k_from_info(info, rgb_shape):
    """(fx, fy, cx, cy) from CameraInfo, or a pinhole guess from image size."""
    if info is not None:
        k = getattr(info, "k", None)
        if k is not None and len(k) >= 6:
            fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
            if fx > 1.0 and fy > 1.0:
                return fx, fy, cx, cy
    h, w = int(rgb_shape[0]), int(rgb_shape[1])
    fx = fy = 0.8 * float(max(w, 1))
    return fx, fy, 0.5 * float(w), 0.5 * float(h)


@dataclass
class LocalizedTarget:
    fwd: float = 0.0
    lat: float = 0.0
    z: float = 0.0
    conf: float = 0.0
    name: str = ""
    world_xy: Optional[tuple] = None
    valid: bool = False
    reason: str = ""
    bbox_xyxy: Optional[tuple] = None
    depth: object = None
    k: Optional[tuple] = None
    R: object = None
    t: object = None
    rgb_shape: Optional[tuple] = None


def _geom(bbox, depth, k, R, t, rgb_shape) -> dict:
    box = None if bbox is None else tuple(float(v) for v in bbox)
    return dict(
        bbox_xyxy=box,
        depth=depth,
        k=k,
        R=R,
        t=t,
        rgb_shape=None if rgb_shape is None else tuple(int(v) for v in rgb_shape[:2]),
    )


def localize_detection(
    bbox, depth, rgb_shape, k, pose, hint_xy, hint_radius,
    conf=1.0, name="", arm_base_fwd=ARM_BASE_FWD, R=None, t=None,
) -> LocalizedTarget:
    """RGB box + depth image -> arm_base target with radius / range checks."""
    geom = _geom(bbox, depth, k, R, t, rgb_shape)
    d = depth_in_bbox(depth, bbox, rgb_shape)
    if d is None:
        return LocalizedTarget(reason="no_depth", **geom)
    if conf is not None and float(conf) < MIN_CONF:
        return LocalizedTarget(
            conf=float(conf), name=name or "", reason="low_conf", **geom
        )
    x1, y1, x2, y2 = (float(v) for v in bbox)
    u = 0.5 * (x1 + x2)
    v = 0.5 * (y1 + y2)
    fx, fy, cx, cy = k
    ox, oy, oz = optical_xyz(u, v, d, fx, fy, cx, cy)
    fwd_c, lat_c, up_c = optical_to_car_base(ox, oy, oz, R=R, t=t)
    fwd, lat, up = car_to_arm_base(fwd_c, lat_c, up_c, arm_base_fwd=arm_base_fwd)
    world_xy = None if pose is None else car_to_world_xy(pose, fwd_c, lat_c)
    if world_xy is not None and hint_xy is not None:
        if not inside_reveal_disc(world_xy, hint_xy, hint_radius, slack=RADIUS_SLACK_M):
            return LocalizedTarget(
                fwd=fwd, lat=lat, z=up, conf=float(conf or 0.0), name=name or "",
                world_xy=world_xy, valid=False, reason="outside_radius", **geom,
            )
    if not (FWD_MIN_M <= fwd <= FWD_MAX_M) or abs(lat) > LAT_MAX_M:
        return LocalizedTarget(
            fwd=fwd, lat=lat, z=up, conf=float(conf or 0.0), name=name or "",
            world_xy=world_xy, valid=False, reason="out_of_reach", **geom,
        )
    return LocalizedTarget(
        fwd=fwd, lat=lat, z=up, conf=float(conf or 0.0), name=name or "",
        world_xy=world_xy, valid=True, reason="ok", **geom,
    )


# ---------------------------------------------------------------------------
# Detector protocol
# ---------------------------------------------------------------------------

class Detector(Protocol):
    def load(self) -> bool: ...
    def set_target(self, name: Optional[str]) -> None: ...
    def detect(self, rgb: np.ndarray) -> Optional[dict]: ...
    def unload(self) -> None: ...


class MissingDetector:
    """Fail-closed stand-in when ultralytics / weights are absent."""

    def __init__(self, reason: str = "no_detector"):
        self.reason = reason

    def load(self) -> bool:
        return False

    def set_target(self, name: Optional[str]) -> None:
        return None

    def detect(self, rgb: np.ndarray) -> Optional[dict]:
        return None

    def unload(self) -> None:
        return None


class YoloDetector:
    """Adapter over ``manip_dataset.yolo_detect.YoloTracker`` plus COCO aliases."""

    def __init__(self, tracker):
        self._yolo = tracker

    def load(self) -> bool:
        return bool(self._yolo.load())

    def set_target(self, name: Optional[str]) -> None:
        raw = None if not name else str(name).strip()
        names = getattr(self._yolo, "names", None) or {}
        inv = {str(n): int(i) for i, n in names.items()}
        if raw and raw in inv:
            self._yolo.set_target(raw)
            return
        if raw:
            for alias in COCO_ALIASES.get(raw, ()):
                if alias in inv:
                    self._yolo.set_target(alias)
                    return
        self._yolo.set_target(None)

    def detect(self, rgb: np.ndarray) -> Optional[dict]:
        return self._yolo.detect(rgb)

    def unload(self) -> None:
        unload = getattr(self._yolo, "unload", None)
        if callable(unload):
            unload()


def make_detector() -> Detector:
    """Production detector: pretrained YOLO if ultralytics loads, else missing."""
    try:
        from manip_dataset.yolo_detect import YoloTracker
    except ImportError as e:
        log.warning("[nearfield] YOLO import failed (%s) -- acquire will fail closed", e)
        return MissingDetector("import_failed")
    return YoloDetector(YoloTracker())


# ---------------------------------------------------------------------------
# Acquire loop
# ---------------------------------------------------------------------------

def _hint_from_reveal(reveal):
    if reveal is None:
        return None, 0.0, ""
    hc = getattr(reveal, "hint_center", None) or []
    hint_xy = (float(hc[0]), float(hc[1])) if len(hc) >= 2 else None
    radius = float(getattr(reveal, "hint_radius", 0.0) or 0.0)
    ttype = str(getattr(reveal, "target_type", "") or "")
    return hint_xy, radius, ttype


def _yaw_for_err(err: float) -> float:
    ang = -YAW_GAIN * float(err)
    mag = min(abs(ang), YAW_MAX)
    mag = max(mag, YAW_MIN)
    return math.copysign(mag, ang)


class NearFieldAcquire:
    """Search inside the reveal disc, close on bbox depth, emit arm_base 3D."""

    def __init__(self, client, detector: Optional[Detector] = None, dump_dir=None):
        self.client = client
        self.detector = detector if detector is not None else make_detector()
        self._dump_dir = dump_dir
        self._dumped_phases: set[str] = set()
        self._lock_rgb = None
        self._lock_depth = None
        self._phase = "scan"
        self._phase_t0 = 0.0
        self._align_ok = 0
        self._scan_sign = 1.0
        self._tf = {}          # child -> (parent, t(3,), R(3,3))
        self._tf_ready = False

    def acquire(self, reveal, deadline) -> LocalizedTarget:
        c = self.client
        if not self.detector.load():
            reason = getattr(self.detector, "reason", "no_detector")
            log.warning("[nearfield] detector unavailable (%s) -- fail closed", reason)
            return LocalizedTarget(valid=False, reason=str(reason))

        hint_xy, radius, ttype = _hint_from_reveal(reveal)
        self.detector.set_target(ttype or None)
        self._ensure_tf()
        if not self._lock_streams():
            log.warning("[nearfield] no chassis RGB/depth -- fail closed")
            return LocalizedTarget(valid=False, reason="no_camera")

        self._enter("scan", c.now_s())
        last: Optional[LocalizedTarget] = None
        last_good_t = c.now_s()
        dt = TICK_S
        rgb = None
        bbox = None
        det = None

        while c.is_running and c.now_s() < deadline:
            now = c.now_s()
            pose = c.get_world_pose()
            rgb, depth, info, rgb_msg = self._read_locked()
            det = None
            if rgb is not None:
                det = self.detector.detect(rgb)
                last_good_t = now

            n_obj = 0 if det is None else int(det.get("n_obj") or 0)
            bbox = None if det is None else det.get("bbox")
            err = None if det is None or n_obj <= 0 else det.get("error")
            d_obj = None
            if n_obj > 0 and bbox is not None and depth is not None:
                d_obj = depth_in_bbox(depth, bbox, det.get("rgb_shape") or rgb.shape[:2])

            if n_obj > 0 and bbox is not None and depth is not None and pose is not None:
                k = k_from_info(info, rgb.shape[:2] if rgb is not None else (480, 640))
                R, tvec = self._cam_to_car(rgb_msg)
                cand = localize_detection(
                    bbox, depth, det.get("rgb_shape") or rgb.shape[:2], k, pose,
                    hint_xy, radius,
                    conf=det.get("conf") or 0.0,
                    name=det.get("name") or "",
                    R=R, t=tvec,
                )
                if cand.reason == "outside_radius":
                    n_obj = 0
                    err = None
                    d_obj = None
                else:
                    last = cand
                    near = d_obj is not None and d_obj <= STOP_M
                    centered = err is not None and abs(float(err)) <= REALIGN
                    already = d_obj is not None and d_obj < ALREADY_NEAR_M
                    if cand.valid and (already or (near and centered)):
                        self._stop()
                        self._dump_frame("localized", rgb, bbox, cand.name)
                        log.info("[nearfield] localized %s fwd=%.2f lat=%.2f z=%.2f d=%.2f",
                                 cand.name, cand.fwd, cand.lat, cand.z, d_obj or -1)
                        return cand

            prev_phase = self._phase
            lin, ang = self._command(now, n_obj, err, d_obj)
            if self._phase != prev_phase:
                self._dump_frame(self._phase, rgb, bbox, "" if det is None else (det.get("name") or ""))
            lin, ang = gated_cmd_vel(pose, lin, ang, hint_xy, radius, dt=dt)
            c.send_cmd_vel(linear_x=lin, angular_z=ang)
            if now - last_good_t > STALE_STOP_S:
                c.send_cmd_vel(0.0, 0.0)
            c.sleep(dt)

        self._stop()
        if last is not None and last.valid:
            self._dump_frame("last_valid", rgb if rgb is not None else None, getattr(last, "bbox_xyxy", None), last.name)
            return last
        reason = "timeout" if last is None else last.reason
        self._dump_frame("failed", rgb if rgb is not None else None, bbox if det else None, reason)
        log.warning("[nearfield] acquire failed (%s)", reason)
        return last if last is not None else LocalizedTarget(valid=False, reason=reason)

    def _dump_frame(self, tag: str, rgb, bbox=None, label: str = "") -> None:
        if tag in self._dumped_phases:
            return
        self._dumped_phases.add(tag)
        try:
            from pathlib import Path
            from marc_agent.perception.round_dump import dump_nearfield_frame
        except Exception:
            return
        dump_nearfield_frame(
            None if self._dump_dir is None else Path(self._dump_dir),
            f"{tag}.png",
            rgb,
            bbox,
            label=label or tag,
        )

    def _enter(self, phase: str, now: float):
        self._phase = phase
        self._phase_t0 = now
        self._align_ok = 0

    def _command(self, now: float, n_obj: int, err, d_obj) -> tuple:
        since = now - self._phase_t0
        has = n_obj > 0 and err is not None
        if d_obj is not None and n_obj > 0 and d_obj < ALREADY_NEAR_M:
            return 0.0, 0.0

        if not has:
            if self._phase != "scan":
                self._enter("scan", now)
            if since >= SCAN_FLIP_S:
                self._scan_sign *= -1.0
                self._enter("scan", now)
            return 0.0, self._scan_sign * SCAN_YAW

        if abs(float(err)) <= CENTER_TOL and d_obj is not None and d_obj <= STOP_M:
            return 0.0, 0.0

        if self._phase == "scan":
            self._enter("align", now)
            return 0.0, 0.0

        if self._phase == "align":
            if since < ALIGN_SETTLE_S:
                return 0.0, 0.0
            if abs(float(err)) <= CENTER_TOL:
                self._align_ok += 1
                if self._align_ok >= ALIGN_HOLD:
                    self._enter("drive", now)
                    return CRAWL_MPS, 0.0
                return 0.0, 0.0
            self._align_ok = 0
            return 0.0, _yaw_for_err(err)

        # drive
        drive_m = since * CRAWL_MPS
        if abs(float(err)) > REALIGN:
            self._enter("align", now)
            return 0.0, 0.0
        if BURST_M > 0.0 and drive_m >= BURST_M:
            self._enter("align", now)
            return 0.0, 0.0
        return min(CRAWL_MPS, MAX_STEP_M / TICK_S), 0.0

    def _stop(self):
        try:
            self.client.send_cmd_vel(0.0, 0.0)
        except Exception:
            pass

    def _lock_streams(self) -> bool:
        """Lock the taller chassis RGB (480x640) and a matching depth."""
        c = self.client
        try:
            c.send_cmd_vel(0.0, 0.0)
        except Exception:
            pass
        rgb_shapes = []
        depth_shapes = []
        n = LOCK_FRAMES
        for _ in range(n):
            rgb_msg = c.get_robot_image(CAMERA)
            depth_msg = c.get_robot_depth(DEPTH_WHICH)
            rgb = _image_to_rgb(rgb_msg)
            depth = _image_to_depth(depth_msg)
            if rgb is not None:
                rgb_shapes.append(tuple(rgb.shape[:2]))
            if depth is not None:
                depth_shapes.append(tuple(depth.shape[:2]))
            getter = getattr(c, "list_robot_image_shapes", None)
            if getter is not None:
                for sh in getter(CAMERA):
                    rgb_shapes.append(tuple(sh))
            c.sleep(0.05)
        if not rgb_shapes:
            return False
        # Prefer the taller frame (chassis 480x640 over 360x640 close-up).
        self._lock_rgb = max(rgb_shapes, key=lambda s: (int(s[0]), int(s[1])))
        self._lock_depth = max(depth_shapes, key=lambda s: (int(s[0]), int(s[1]))) if depth_shapes else None
        log.info("[nearfield] locked rgb=%s depth=%s", self._lock_rgb, self._lock_depth)
        return True

    def _read_locked(self):
        c = self.client
        rgb_msg = None
        hw = getattr(c, "get_robot_image_hw", None)
        if hw is not None and self._lock_rgb is not None:
            rgb_msg = hw(CAMERA, self._lock_rgb)
        if rgb_msg is None:
            rgb_msg = c.get_robot_image(CAMERA)
        depth_msg = None
        dhw = getattr(c, "get_robot_depth_hw", None)
        if dhw is not None and self._lock_depth is not None:
            depth_msg = dhw(DEPTH_WHICH, self._lock_depth)
        if depth_msg is None:
            depth_msg = c.get_robot_depth(DEPTH_WHICH)
        rgb = _image_to_rgb(rgb_msg)
        depth = _image_to_depth(depth_msg)
        info = None
        getter = getattr(c, "get_robot_info", None)
        if getter is not None:
            try:
                info = getter(CAMERA)
            except Exception:
                info = None
        if rgb is not None and self._lock_rgb is not None:
            if tuple(rgb.shape[:2]) != tuple(self._lock_rgb):
                rgb = None
                rgb_msg = None
        if depth is not None and self._lock_depth is not None:
            if tuple(depth.shape[:2]) != tuple(self._lock_depth):
                depth = None
        return rgb, depth, info, rgb_msg

    def _ensure_tf(self):
        if self._tf_ready:
            return
        c = self.client
        sub = getattr(c, "subscribe", None)
        if sub is None or getattr(c, "node", None) is None:
            return
        try:
            from tf2_msgs.msg import TFMessage
            from marc_sdk import protocol as P
            sub("/tf_static", TFMessage, self._on_tf, qos=P.QOS_TRANSIENT)
            sub("/tf", TFMessage, self._on_tf, qos=P.QOS_RELIABLE)
            self._tf_ready = True
        except Exception as e:
            log.info("[nearfield] TF subscribe skipped (%s) -- using optical fallback", e)

    def _on_tf(self, msg):
        for ts in getattr(msg, "transforms", []) or []:
            parent = (ts.header.frame_id or "").lstrip("/")
            child = (ts.child_frame_id or "").lstrip("/")
            tr = ts.transform.translation
            q = ts.transform.rotation
            t = np.array([float(tr.x), float(tr.y), float(tr.z)], dtype=np.float64)
            R = _quat_to_R(float(q.x), float(q.y), float(q.z), float(q.w))
            self._tf[child] = (parent, t, R)

    def _cam_to_car(self, rgb_msg):
        """Return (R, t) mapping optical camera coords into car_base, or (None, None)."""
        frame = ""
        if rgb_msg is not None:
            frame = (getattr(getattr(rgb_msg, "header", None), "frame_id", "") or "").lstrip("/")
        if not frame or not self._tf:
            return None, None
        T = _compose_tf(self._tf, frame, "car_base")
        if T is None:
            T = _compose_tf(self._tf, frame, "arm_base_link")
            if T is not None:
                R, t = T
                # T maps camera -> arm_base; shift +X so optical_to_car_base stays in car_base
                # (car_to_arm_base then subtracts ARM_BASE_FWD once).
                t = np.asarray(t, dtype=np.float64) + np.array([ARM_BASE_FWD, 0.0, 0.0])
                return R, t
            return None, None
        return T


def _quat_to_R(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _invert_Rt(t, R):
    Rt = np.asarray(R, dtype=np.float64).T
    return (-Rt @ np.asarray(t, dtype=np.float64)), Rt


def _compose_tf(tree, src, dst):
    """Compose child->parent links so ``p_dst = R @ p_src + t``. None if no path."""
    if src == dst:
        return np.eye(3), np.zeros(3)
    edges = {}
    for child, (parent, t, R) in tree.items():
        edges.setdefault(child, []).append((parent, t, R, False))
        ti, Ri = _invert_Rt(t, R)
        edges.setdefault(parent, []).append((child, ti, Ri, True))
    q = deque([src])
    T_from_src = {src: (np.eye(3), np.zeros(3))}
    while q:
        cur = q.popleft()
        if cur == dst:
            return T_from_src[cur]
        for nxt, t, R, _ in edges.get(cur, []):
            if nxt in T_from_src:
                continue
            R0, t0 = T_from_src[cur]
            T_from_src[nxt] = (R @ R0, R @ t0 + t)
            q.append(nxt)
    return None
