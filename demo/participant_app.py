"""MARC 2026 SDK demo participant (baseline code).

The baseline demo that wires the whole flow together with **marc_sdk** --
Stage 1 grounding + Stage 2 occupancy D* Lite driving (cmd_vel).
The protocol/ROS wiring (register, subscribe/publish, world_pose, occupancy) is all
handled by ``MARCClient``, so only the participant logic (mock VLA + nav) is connected
through handlers.

Run:
    source /opt/ros/humble/setup.bash
    export MARC_TEAM_ID=u1
    export MARC_TOKEN=<token>            # your assigned token (required; issued by the organizers)
    python3 participant_app.py
  or
    ./launch.sh u1

Precondition: the simulation platform must be running first.
"""

import logging
import math
import os
import threading

from marc_sdk import MARCClient

from marc_agent.command_parser import Intent
from marc_agent.navigation_runtime import Stage2Navigator
from marc_agent.gpu_handoff import (
    after_successful_pick,
    before_graspgenx,
    before_reacquire,
    unload_cctv_yolo,
    unload_qwen,
)
from marc_agent.perception.round_dump import (
    dump_json,
    dump_stage_round,
    stage1_dir,
    stage2_dir,
)
from marc_agent.perception.cctv_project import CctvProjector
from marc_agent.perception.cctv_round import discovery_timeout_from_limit
from marc_agent.perception.stage1_pipeline import (
    append_stage1_record,
    get_default_verifier,
    log_qwen_startup_status,
    run_stage1_round,
)
from marc_agent.pick_runtime import PickRuntime
from marc_agent.retrieval_state import (
    RetrievalState,
    nearfield_start_reason,
    search_area_approach_goal,
)
from marc_agent.round_clock import RoundClock
from mock_agent_vla import MockVLAGrounding
from mock_agent_navigation import lidar_cloud_to_obstacles, frontal_hit
from nearfield import NearFieldAcquire, make_detector

log = logging.getLogger("participant_app")

# -- Stage 2 navigation parameters --
# Demo-internal giveup so the drive loop cannot spin forever if a goal is unreachable. This is NOT
# the competition timeout - the platform counts and enforces the real Stage 2 time limit.
STAGE2_NAV_TIMEOUT_S = 1800.0
STAGE2_NAV_RATE_HZ = 10.0
STAGE2_NAV_DT = 1.0 / STAGE2_NAV_RATE_HZ
STAGE2_INFLATION_M = 0.25      # erode free by the robot radius (avoids road<->grass step stalls)
STAGE2_STUCK_TIME_S = 8.0      # if movement < threshold for this long, replan from the current pose
STAGE2_STUCK_MOVE_M = 0.05
STAGE2_REPLAN_CAP = 50         # fail closed if the crop stays sealed
GOAL_TOLERANCE = 0.7           # reached test (m) around the search-area approach goal

# -- Pick approach --
# Stop the chassis this far (m) in front of the object, then close the last stretch with a
# straight approach (the planner cannot enter a tight spot fully). Avoids driving over the object.
TUMBLER_STANDOFF_M = 0.58
# car_base(world_pose) -> arm_base forward offset. arm_base is ~0.18 m ahead of the chassis pose.
ARM_BASE_FWD = 0.18
# Grasp point relative to the object along +fwd. Negative = slightly past the object center
# so the gripper closes around its body (a small backoff grabbed only the edge -> dropped in
# transit), tuned slightly forward for a secure grip.
GRASP_FWD_BACKOFF = -0.01

class DemoParticipant:
    """SDK-based demo participant -- connects the VLA/nav logic through handlers."""

    def __init__(self):
        self.client = MARCClient.from_env()
        # Stage 1 never constructs or reads the scenario-backed mock. Keep the legacy
        # Stage 2 demo adapter lazy until that separate mission path is invoked.
        self._stage2_vla = None
        self._state = RetrievalState()
        self._reveal = None
        self._last_detections = None
        self._last_stage1_candidates = None
        self._stage1_lock = threading.Lock()
        self._stage1_projector = None
        self._nearfield_detector = None
        self._round_clock = RoundClock()
        self._qwen_verifier = get_default_verifier()
        self._register_handlers()

    # -- handler registration --

    def _register_handlers(self):
        c = self.client
        c.on_mission(self._on_mission)
        c.on_stage2_mission(self._on_stage2_mission)
        c.on_stage2_reveal(self._on_stage2_reveal)
        c.on_stage2_run(self._drive)
        c.on_score(self._on_score)
        c.on_state_change(self._on_state_change)
        c.on_time_remaining(self._on_time_remaining)
        c.on_time_expired(self._on_time_expired)

    # -- Stage 1 --

    def _on_mission(self, mission):
        log.info("[Stage1] round %s/%s: \"%s\"",
                 mission.round, mission.total_rounds, mission.voice_command)
        thread = threading.Thread(
            target=self._run_stage1_mission,
            args=(mission,),
            name=f"stage1-round-{mission.round}",
            daemon=True,
        )
        thread.start()

    def _run_stage1_mission(self, mission):
        with self._stage1_lock:
            try:
                self._execute_stage1(mission)
            except Exception:
                log.exception("[Stage1] round %s failed", mission.round)

    def _execute_stage1(self, mission):
        timeout_s = discovery_timeout_from_limit(getattr(mission, "time_limit", None))
        if self._stage1_projector is None:
            self._stage1_projector = CctvProjector.from_client(self.client)
        self._round_clock.begin(getattr(mission, "time_limit", None), self.client.now_s())
        round_result = run_stage1_round(
            self.client,
            mission.voice_command,
            discovery_timeout_s=timeout_s,
            projector=self._stage1_projector,
            clock=self._round_clock,
            verifier=self._qwen_verifier,
        )
        self._last_stage1_candidates = round_result.candidates
        self._last_detections = round_result.candidates.detections
        dump_path = dump_stage_round(stage1_dir(mission.round), round_result)
        log.info(
                 "[Stage1] CCTV round: %d cameras, %d frames, %d objects, "
                 "%d people, selected=%s method=%s reason=%s",
                 len(self._last_detections.camera_ids),
                 len(self._last_detections.frames),
                 len(self._last_detections.objects),
                 len(self._last_detections.people),
                 round_result.target_id
                 or self._last_stage1_candidates.selected_candidate_id,
                 round_result.method,
                 round_result.failure_reason
                 or self._last_stage1_candidates.failure_reason)
        metrics_path = os.environ.get(
            "MARC_STAGE1_METRICS_PATH",
            "artifacts/stage1_rounds.jsonl",
        )
        try:
            append_stage1_record(
                self._last_stage1_candidates,
                metrics_path,
                extra={"dump_dir": str(dump_path) if dump_path else None},
            )
        except OSError as exc:
            log.warning("[Stage1] could not append metrics to %s: %s", metrics_path, exc)

        grounding = round_result.grounding
        if grounding is None:
            log.info(
                "[Stage1] skip submit reason=%s (no zero-coordinate fallback)",
                round_result.failure_reason,
            )
            return
        seq = self.client.submit_grounding(grounding)
        log.info(
            "[Stage1] submitted grounding seq=%s camera=%s target_id=%s "
            "anchor_id=%s method=%s target_xyz=%s anchor_xyz=%s "
            "relation=%s landmark=%s situation=%s",
            seq,
            grounding.camera_id,
            round_result.target_id,
            round_result.anchor_id,
            round_result.method,
            grounding.target_coord,
            grounding.anchor_coord,
            grounding.relation,
            grounding.landmark,
            grounding.situation,
        )

    # -- Stage 2: submit interpretation -> drive --

    def _on_stage2_mission(self, mission):
        log.info("[Stage2] task: \"%s\" (limit %.0fs)",
                 mission.task_description, mission.time_limit)
        # Store owner_position for the delivery leg (#14); this issue does not drive there.
        self._state.adopt_mission(mission)
        if self._state.owner_xy is not None:
            log.info("[Stage2] owner_position(runtime-provided)=%s", self._state.owner_xy)
        self._round_clock.begin(getattr(mission, "time_limit", None), self.client.now_s())
        if os.environ.get("MARC_STAGE2_MOCK_VLA", "").strip() == "1":
            if self._stage2_vla is None:
                self._stage2_vla = MockVLAGrounding()
            result = self._stage2_vla.process_stage2(mission.task_description)
            self.client.submit_stage2_grounding(result)
            self._state.adopt_grounding(result.target_coord)
            log.info("[Stage2] submitted mock grounding target=%s (owner stored=%s)",
                     self._state.grounding_xy, self._state.owner_xy)
            dump_json(
                stage2_dir(),
                "grounding.json",
                {
                    "submitted": {
                        "target_coord": getattr(result, "target_coord", None),
                        "target_type": getattr(result, "target_type", None),
                    },
                    "method": "mock_vla",
                    "failure_reason": None,
                },
            )
            dump_json(stage2_dir(), "summary.json", {
                "method": "mock_vla",
                "task": mission.task_description,
            })
            return
        if self._stage1_projector is None:
            self._stage1_projector = CctvProjector.from_client(self.client)
        timeout_s = discovery_timeout_from_limit(getattr(mission, "time_limit", None))
        round_result = run_stage1_round(
            self.client,
            mission.task_description,
            expected_intent=Intent.BRING,
            projector=self._stage1_projector,
            discovery_timeout_s=timeout_s,
            clock=self._round_clock,
            verifier=self._qwen_verifier,
        )
        dump_stage_round(stage2_dir(), round_result)
        grounding = round_result.grounding
        if grounding is None:
            log.info(
                "[Stage2] skip submit reason=%s (no zero-coordinate fallback)",
                round_result.failure_reason,
            )
            return
        self.client.submit_stage2_grounding(grounding)
        self._state.adopt_grounding(grounding.target_coord)
        log.info(
            "[Stage2] submitted grounding camera=%s target_id=%s method=%s "
            "target_xyz=%s landmark=%s relation=%s",
            grounding.camera_id,
            round_result.target_id,
            round_result.method,
            grounding.target_coord,
            grounding.landmark,
            grounding.relation,
        )

    def _on_stage2_reveal(self, reveal):
        """Stage 2 post-grounding reveal (msg 411) -- search area, not the object.

        ``hint_center`` + ``hint_radius`` bound reacquisition (#12). ``target_type``
        selects among distractors inside the disc. A late reveal retargets the live nav leg.
        """
        log.info("[Stage2] reveal received -- score=%s, type=%s, center=%s, r=%.1f",
                 reveal.grounding_score, reveal.target_type,
                 reveal.hint_center, reveal.hint_radius)
        self._reveal = reveal
        self._state.adopt_reveal(reveal)

    def _drive(self):
        """Called once in a separate thread on entering STAGE2_RUN.

        Leg 1: D* to the reveal search-area approach goal. #12 localizes in RGB-D,
        then #13 grasps only on positive is_grasping() and loads the basket.
        Delivery to owner_position is #14 — stored, not driven here.
        """
        c = self.client
        # SIM-time deadline (via /clock) -- shared across all legs, robust to real-time factor < 1.
        deadline = c.now_s() + STAGE2_NAV_TIMEOUT_S
        dump = stage2_dir()

        self._wait_for_search_goal(timeout=5.0)
        search = self._state.search_xy()
        log.info("[Stage2] navigation start -- search=%s r=%.1f type=%s (timeout %.0fs)",
                 search, self._state.hint_radius, self._state.target_type or "-",
                 STAGE2_NAV_TIMEOUT_S)

        self._stow_arm()
        unload_cctv_yolo()
        unload_qwen()
        self._qwen_verifier = None
        detector = self._preload_nearfield()

        picked = False
        reached_search = False
        handoff = "skip_no_search"
        if search is None:
            log.warning("[Stage2] no search goal (no reveal, no grounding) -- skip nav and pick")
            dump_json(dump, "nav.json", {"skipped": True, "reason": "no_search_goal"})
        else:
            pose = c.get_world_pose()
            pose_xy = (pose[0], pose[1]) if pose is not None else search
            approach = search_area_approach_goal(search, self._state.hint_radius, pose_xy)
            log.info("[Stage2] leg1 -> search-area approach %s (center=%s r=%.2f)",
                     tuple(round(v, 2) for v in approach),
                     tuple(round(v, 2) for v in search), self._state.hint_radius)
            result = self._navigate_to(
                self._live_approach_goal,
                deadline,
                tol=GOAL_TOLERANCE,
                inflation_m=STAGE2_INFLATION_M,
                exclude_m=0.6,
                leg="search",
            )
            reached_search = result.reached
            pose_after = c.get_world_pose()
            pose_xy = (pose_after[0], pose_after[1]) if pose_after is not None else pose_xy
            handoff = nearfield_start_reason(
                reached_search, pose_xy, search, self._state.hint_radius, GOAL_TOLERANCE,
            )
            log.info("[Stage2] leg1 result=%s reason=%s replans=%d yolo=%s",
                     result.status.value, result.reason, result.replan_count, handoff)
            dump_json(dump, "nav.json", {
                "status": result.status.value,
                "reason": result.reason,
                "reached": reached_search,
                "replan_count": result.replan_count,
                "search": list(search) if search is not None else None,
                "hint_radius": self._state.hint_radius,
                "approach": list(approach),
                "pose_xy": list(pose_xy) if pose_xy is not None else None,
                "handoff": handoff,
                "last_method": result.last_method,
                "waypoint_count": len(result.path) if result.path else 0,
                "first_waypoint": list(result.path[0]) if result.path else None,
                "last_waypoint": list(result.path[-1]) if result.path else None,
            })
        if not c.is_running:
            return
        start_yolo = not str(handoff).startswith("skip")
        if not start_yolo:
            log.warning("[Stage2] skip pick -- near-field not started (%s)", handoff)
            dump_json(dump, "pick.json", {"skipped": True, "reason": handoff})
        else:
            log.info("[Stage2] starting near-field YOLO (%s)", handoff)
            reveal = self._reveal if self._reveal is not None else c.stage2_reveal
            target = NearFieldAcquire(c, detector, dump_dir=dump).acquire(reveal, deadline)
            for _ in range(5):
                c.send_cmd_vel(0.0, 0.0)
                c.sleep(0.1)
            if not target.valid:
                log.warning("[Stage2] near-field acquire failed (%s) -- skip pick (fail closed)",
                            target.reason)
                dump_json(dump, "pick.json", {
                    "skipped": True,
                    "reason": target.reason,
                    "valid": False,
                })
            else:
                log.info("[Stage2] localized %s grasp=(%.2f, %.2f, z=%.2f) conf=%.2f",
                         target.name or "?", target.fwd, target.lat, target.z, target.conf)
                before_graspgenx(detector)
                pick_rt = PickRuntime(c, log)

                def _reacquire():
                    before_reacquire(pick_rt, detector)
                    nxt = NearFieldAcquire(c, detector, dump_dir=dump).acquire(
                        reveal, deadline)
                    for _ in range(3):
                        c.send_cmd_vel(0.0, 0.0)
                        c.sleep(0.05)
                    if nxt is not None and getattr(nxt, "valid", False):
                        before_graspgenx(detector)
                    return nxt

                pick_rt._reacquire_fn = _reacquire
                pick = pick_rt.run(target, deadline=deadline, reveal=reveal)
                self._state.adopt_pick(pick)
                picked = pick.ok
                c.send_cmd_vel(0.0, 0.0)
                if picked:
                    after_successful_pick(pick_rt, detector)
                dump_json(dump, "pick.json", {
                    "ok": pick.ok,
                    "reason": pick.reason,
                    "attempts": pick.attempts,
                    "grasped": pick.grasped,
                    "basket_occupied": pick.basket_occupied,
                    "policy": pick.policy,
                    "target": {
                        "name": target.name,
                        "fwd": target.fwd,
                        "lat": target.lat,
                        "z": target.z,
                        "conf": target.conf,
                        "valid": target.valid,
                        "bbox_xyxy": list(target.bbox_xyxy) if target.bbox_xyxy else None,
                    },
                })
                log.info("[Stage2] pick %s reason=%s attempts=%d grasping=%s basket=%s policy=%s",
                         "ok" if picked else "failed", pick.reason, pick.attempts,
                         pick.grasped, pick.basket_occupied, pick.policy)

        if not picked:
            self._stow_arm()

        c.stop()
        log.info("[Stage2] search-area nav finished (reached=%s yolo=%s) -> task_complete",
                 reached_search, handoff)
        c.task_complete()

    def _preload_nearfield(self):
        if self._nearfield_detector is None:
            self._nearfield_detector = make_detector()
        try:
            self._nearfield_detector.load()
        except Exception as exc:
            log.warning("[Stage2] near-field YOLO preload failed: %s", exc)
        return self._nearfield_detector

    def _stow_arm(self):
        """Fold the arm to the HOME pose to lower the center of gravity -- tip-over prevention.

        The arm driver re-applies the last joint_command every tick, so one publish holds the pose
        for the whole leg. (Not called after a successful pick -- leg 2 keeps the carry pose.)
        """
        try:
            from arm_pick import HOME, GRIPPER_OPEN, _make_js
            self.client.send_arm_command(_make_js(HOME, GRIPPER_OPEN))
            log.info("[Stage2] arm stow -> HOME (lower CoG, tip-over prevention)")
        except Exception as e:
            log.warning("[Stage2] arm stow failed: %s", e)

    def _standoff_goal(self, dest, standoff):
        """A point ``standoff`` metres in front of ``dest`` along the robot->dest line.

        Approaching a standoff (instead of the object itself) keeps the chassis from driving
        onto/over the object.
        """
        pose = self.client.get_world_pose()
        if pose is None:
            return dest
        x, y, _ = pose
        dx, dy = dest[0] - x, dest[1] - y
        d = math.hypot(dx, dy)
        if d <= standoff:
            return (x, y)
        f = (d - standoff) / d
        return (x + dx * f, y + dy * f)

    def _object_arm_xy(self, obj_xy):
        """Object world XY -> arm_base frame (fwd, lat) using the current robot pose.

        Rotate the world offset into the car_base frame, then shift by the
        car_base->arm_base forward offset. Lets the closed-loop pick aim at the real
        object regardless of small nav misalignment.
        """
        pose = self.client.get_world_pose()
        if pose is None:
            return None
        x, y, yaw = pose
        dx, dy = obj_xy[0] - x, obj_xy[1] - y
        cc, ss = math.cos(yaw), math.sin(yaw)
        fwd = dx * cc + dy * ss            # car_base forward (+X)
        lat = -dx * ss + dy * cc           # car_base left (+Y)
        return (fwd - ARM_BASE_FWD - GRASP_FWD_BACKOFF, lat)

    def _approach_straight(self, obj_xy, deadline, target_fwd=0.40, max_t=20.0):
        """Drive straight at the object until the arm_base forward distance reaches target_fwd.

        The planner cannot enter a tight spot fully (it halts ~1 m out), so fill the remaining
        empty space by driving straight into the object, with a small lateral steering term to
        center it (lat -> 0). SIM-time paced.
        """
        c = self.client
        t_end = min(deadline, c.now_s() + max_t)
        while c.is_running and c.now_s() < t_end:
            gxy = self._object_arm_xy(obj_xy)
            if gxy is None:
                break
            fwd, lat = gxy
            if fwd <= target_fwd + 0.02:
                break
            pose = c.get_world_pose()
            if pose is not None:
                cloud = c.get_lidar()
                obs = lidar_cloud_to_obstacles(cloud, pose) if cloud is not None else []
                if frontal_hit(obs, pose, x_max=0.8, y_half=0.35):
                    c.send_cmd_vel(0.0, 0.0)
                    log.info("[Stage2] straight approach stopped -- frontal lidar hit")
                    break
            ang = max(-0.4, min(0.4, 1.5 * lat))         # center the object (lat -> 0)
            lin = 0.12 if abs(lat) < 0.15 else 0.06      # badly off -> creep slower while turning
            c.send_cmd_vel(linear_x=lin, angular_z=ang)
            c.sleep(STAGE2_NAV_DT)
        for _ in range(5):
            c.send_cmd_vel(0.0, 0.0)
            c.sleep(0.05)
        gxy = self._object_arm_xy(obj_xy)
        log.info("[Stage2] straight approach done -- object arm coords %s",
                 tuple(round(v, 3) for v in gxy) if gxy else None)

    def _wait_for_search_goal(self, timeout: float = 5.0):
        """Poll until reveal fills hint_center, else keep grounding_xy as fallback."""
        c = self.client
        deadline = c.now_s() + timeout
        while c.now_s() < deadline:
            if self._state.hint_center is not None:
                return self._state.search_xy()
            c.sleep(0.1)
        if self._state.hint_center is None:
            log.warning("[Stage2] reveal not received (%.0fs) -- fallback grounding %s",
                        timeout, self._state.grounding_xy)
        return self._state.search_xy()

    def _live_approach_goal(self):
        """Near-edge search-area goal from current pose; retargets on a late reveal."""
        center = self._state.search_xy()
        if center is None:
            return None
        pose = self.client.get_world_pose()
        pose_xy = (pose[0], pose[1]) if pose is not None else center
        return search_area_approach_goal(center, self._state.hint_radius, pose_xy)

    def _navigate_to(self, goal, deadline, tol=GOAL_TOLERANCE,
                     inflation_m=STAGE2_INFLATION_M, exclude_m=0.6, cruise=False,
                     leg="search"):
        """Drive to a goal (x, y) or callable via live occupancy D* Lite."""
        nav = Stage2Navigator(
            self.client,
            dt=STAGE2_NAV_DT,
            inflation_m=inflation_m,
            exclude_m=exclude_m,
            stuck_time_s=STAGE2_STUCK_TIME_S,
            stuck_move_m=STAGE2_STUCK_MOVE_M,
            replan_cap=STAGE2_REPLAN_CAP,
            cruise=cruise,
        )
        return nav.navigate_to(goal, deadline, leg=leg, tol=tol)

    # -- notifications --

    def _on_score(self, score):
        payload = {
            "stage": score.stage,
            "round": score.round,
            "scores": score.scores,
            "total": score.total,
            "is_final": score.is_final,
        }
        if score.is_final:
            log.info("[FINAL] %s", score.scores)
            dump_json(stage2_dir(), "score.json", payload)
        else:
            log.info("[Score] round %s: total=%s", score.round, score.total)
            if score.round is not None:
                dump_json(stage1_dir(score.round), "score.json", payload)

    def _on_state_change(self, old, new):
        log.info("[STATE] %s -> %s", old, new)

    def _on_time_remaining(self, remaining):
        self._round_clock.note(remaining, self.client.now_s())

    def _on_time_expired(self, which):
        log.info("[TIME] expired: %s", which)

    # -- run --

    def run(self):
        if not self.client.connect():
            raise SystemExit("Registration failed -- check the runtime and the token.")
        self.client.run()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log_qwen_startup_status()
    DemoParticipant().run()


if __name__ == "__main__":
    main()
