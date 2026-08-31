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
from dataclasses import replace

from marc_sdk import MARCClient

from marc_agent.command_parser import Intent
from marc_agent.navigation_runtime import NavStatus, Stage2Navigator
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
from marc_agent.perception.lost_item_priors import people_world_xy
from marc_agent.perception.cctv_round import (
    discovery_timeout_from_limit,
    fetch_cctv_round,
)
from marc_agent.perception.stage1_pipeline import (
    append_stage1_record,
    get_default_verifier,
    has_complete_camera_coverage,
    log_qwen_startup_status,
    run_stage1_round,
    static_scene_key,
)
from marc_agent.pick_runtime import PickRuntime, pick_policy_name
from marc_agent.retrieval_state import (
    RetrievalState,
    nearfield_start_reason,
    search_area_approach_goal,
    snapshot_approach_goal,
)
from marc_agent.round_clock import RoundClock
from mock_agent_vla import MockVLAGrounding
from marc_agent.lidar_safety import advise, pointcloud2_xyz
from nearfield import (
    NearFieldAcquire,
    PEEK_PERIOD_S,
    PEEK_RANGE_PAD_M,
    YoloDetector,
    peek_yolo_handoff,
)
from manip_dataset.base_hold import StationaryBase
from manip_dataset.depth_approach import ApproachConfig, DepthApproachController
from manip_dataset.graspgenx_pick import StagedGraspGenXPickRuntime
from manip_dataset.pick_target import stable_pick_target

log = logging.getLogger("participant_app")

# -- Stage 2 navigation parameters --
# Demo-internal giveup so the drive loop cannot spin forever if a goal is unreachable. This is NOT
# the competition timeout - the platform counts and enforces the real Stage 2 time limit.
STAGE2_NAV_TIMEOUT_S = 1800.0
STAGE2_NAV_RATE_HZ = 10.0
STAGE2_NAV_DT = 1.0 / STAGE2_NAV_RATE_HZ
STAGE2_INFLATION_M = 0.55      # body radius plus curb pad (keeps wheels off the grass)
STAGE2_STUCK_TIME_S = 8.0      # if movement < threshold for this long, replan from the current pose
STAGE2_STUCK_MOVE_M = 0.05
STAGE2_REPLAN_CAP = 50         # fail closed if the crop stays sealed
GOAL_TOLERANCE = 0.7           # reached test (m) around the search-area approach goal
STAGE2_APPROACH_TIMEOUT_S = 120.0
STAGE2_DELIVERY_RESERVE_S = 120.0

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
        self._stage1_grounding_cache = {}
        self._stage1_lock = threading.Lock()
        self._stage1_projector = None
        self._round_clock = RoundClock()
        self._qwen_verifier = get_default_verifier()
        self._approach_snap = None
        self._stage2_deadline = None
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
        key = static_scene_key(round_result.candidates.parsed)
        cached = self._stage1_grounding_cache.get(key)
        if cached is not None:
            round_result = replace(
                round_result,
                grounding=cached.grounding,
                method="static_scene_cache",
                target_id=cached.target_id,
                anchor_id=cached.anchor_id,
            )
            log.info("[Stage1] reused static-scene grounding key=%s", key)
        elif (
            key is not None
            and round_result.grounding is not None
            and has_complete_camera_coverage(round_result.candidates.detections)
        ):
            self._stage1_grounding_cache[key] = round_result
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
        # Store the runtime-provided delivery coordinate for the post-pick leg.
        self._state.adopt_mission(mission)
        self._approach_snap = None
        self._stage2_deadline = self.client.now_s() + float(mission.time_limit)
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
        detections = round_result.candidates.detections
        self._state.adopt_people_xy(
            people_world_xy(detections.people, self._stage1_projector)
        )
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
        Leg 2 carries the basket to the runtime-provided owner_position.
        """
        c = self.client
        self._approach_snap = None
        # SIM-time deadline (via /clock) -- shared across all legs, robust to real-time factor < 1.
        deadline = min(
            c.now_s() + STAGE2_NAV_TIMEOUT_S,
            self._stage2_deadline or float("inf"),
        )
        dump = stage2_dir()

        self._wait_for_search_goal(timeout=5.0)
        search = self._state.search_xy()
        log.info("[Stage2] navigation start -- search=%s r=%.1f type=%s (timeout %.0fs)",
                 search, self._state.hint_radius, self._state.target_type or "-",
                 max(0.0, deadline - c.now_s()))

        self._stow_arm()
        unload_cctv_yolo()
        unload_qwen()
        self._qwen_verifier = None
        target_name = self._state.target_type or ""
        approach_controller = DepthApproachController(c, ApproachConfig.from_env())
        approach_controller.set_target(target_name)
        detector = YoloDetector(approach_controller._yolo)
        try:
            detector.load()
        except Exception as exc:
            log.warning("[Stage2] near-field YOLO preload failed: %s", exc)
        nearfield = NearFieldAcquire(
            c, detector, dump_dir=dump, people_xy=self._state.people_xy,
        )

        picked = False
        reached_search = False
        handoff = "skip_no_search"
        if search is None:
            log.warning("[Stage2] no search goal (no reveal, no grounding) -- skip nav and pick")
            dump_json(dump, "nav.json", {"skipped": True, "reason": "no_search_goal"})
        else:
            pose = c.get_world_pose()
            pose_xy = (pose[0], pose[1]) if pose is not None else search
            self._approach_snap = snapshot_approach_goal(
                None, search, self._state.hint_radius, pose_xy,
            )
            approach = self._approach_snap[2] if self._approach_snap is not None else search
            log.info("[Stage2] leg1 -> search-area approach %s (center=%s r=%.2f)",
                     tuple(round(v, 2) for v in approach),
                     tuple(round(v, 2) for v in search), self._state.hint_radius)
            last_peek = [0.0]

            def _yolo_handoff(pose):
                center = self._state.search_xy()
                if center is None or pose is None or len(pose) < 2:
                    return False
                dist = math.hypot(
                    float(pose[0]) - float(center[0]),
                    float(pose[1]) - float(center[1]),
                )
                if dist > float(self._state.hint_radius or 0.0) + PEEK_RANGE_PAD_M:
                    return False
                now = c.now_s()
                if now - last_peek[0] < PEEK_PERIOD_S:
                    return False
                last_peek[0] = now
                return peek_yolo_handoff(
                    c, detector, pose, center, self._state.hint_radius,
                    target_type=self._state.target_type or "",
                )

            result = None
            while c.is_running and c.now_s() < deadline:
                result = self._navigate_to(
                    self._live_approach_goal,
                    deadline,
                    tol=GOAL_TOLERANCE,
                    inflation_m=STAGE2_INFLATION_M,
                    exclude_m=0.6,
                    leg="search",
                    search_xy=self._state.search_xy(),
                    search_radius=self._state.hint_radius,
                    handoff_fn=_yolo_handoff,
                )
                reached_search = result.reached
                pose_after = c.get_world_pose()
                pose_xy = (pose_after[0], pose_after[1]) if pose_after is not None else pose_xy
                handoff = nearfield_start_reason(
                    reached_search, pose_xy, search, self._state.hint_radius, GOAL_TOLERANCE,
                )
                if not str(handoff).startswith("skip"):
                    break
                if handoff != "skip_too_far":
                    break
                if result.status in (NavStatus.NO_MAP, NavStatus.TIMEOUT):
                    break
                log.info("[Stage2] still outside search disc (%s) -- resume occupancy nav",
                         handoff)
            log.info("[Stage2] leg1 result=%s reason=%s replans=%d yolo=%s",
                     result.status.value if result is not None else "none",
                     result.reason if result is not None else "",
                     result.replan_count if result is not None else 0,
                     handoff)
            dump_json(dump, "nav.json", {
                "status": None if result is None else result.status.value,
                "reason": None if result is None else result.reason,
                "reached": reached_search,
                "replan_count": 0 if result is None else result.replan_count,
                "search": list(search) if search is not None else None,
                "hint_radius": self._state.hint_radius,
                "approach": list(approach),
                "pose_xy": list(pose_xy) if pose_xy is not None else None,
                "handoff": handoff,
                "last_method": None if result is None else result.last_method,
                "waypoint_count": 0 if result is None or not result.path else len(result.path),
                "first_waypoint": (
                    None if result is None or not result.path else list(result.path[0])
                ),
                "last_waypoint": (
                    None if result is None or not result.path else list(result.path[-1])
                ),
            })
        if not c.is_running:
            return
        start_yolo = not str(handoff).startswith("skip")
        if not start_yolo:
            log.warning("[Stage2] skip pick -- near-field not started (%s)", handoff)
            dump_json(dump, "pick.json", {"skipped": True, "reason": handoff})
        else:
            log.info("[Stage2] starting calibrated near-field approach (%s)", handoff)
            reveal = self._reveal if self._reveal is not None else c.stage2_reveal
            approach_target = None
            approach_deadline = min(deadline, c.now_s() + STAGE2_APPROACH_TIMEOUT_S)
            if approach_controller.start():
                while c.is_running and c.now_s() < approach_deadline:
                    if approach_controller.tick():
                        approach_controller.settle()
                        approach_target = approach_controller.locked_target(target_name)
                        break
                    c.sleep(approach_controller.cfg.tick_s)
                if approach_controller.active:
                    approach_controller.cancel()
                    log.warning("[Stage2] calibrated approach timed out; using scan fallback")
            else:
                log.warning(
                    "[Stage2] calibrated approach unavailable (%s); using scan fallback",
                    approach_controller.last_failure,
                )
            target = approach_target or nearfield.acquire(reveal, deadline)
            for _ in range(5):
                c.send_cmd_vel(0.0, 0.0)
                c.sleep(0.1)
            if target.valid:
                stable = stable_pick_target(
                    nearfield,
                    target.name or self._state.target_type or "",
                    log=log.info,
                    posture_hint=approach_target,
                )
                if stable is None:
                    target.valid = False
                    target.reason = "unstable_localization"
                else:
                    target = stable
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
                policy = pick_policy_name()
                if policy == "graspgenx":
                    pick_rt = StagedGraspGenXPickRuntime(c, log, policy=policy)
                else:
                    pick_rt = PickRuntime(c, log, policy=policy)

                def _observe_target():
                    before_reacquire(pick_rt, detector)
                    return stable_pick_target(
                        nearfield,
                        target.name or self._state.target_type or "",
                        log=log.info,
                        require_reach=False,
                    )

                if isinstance(pick_rt, StagedGraspGenXPickRuntime):
                    pick_rt._target_observer = _observe_target

                    def _observe_visual_target():
                        before_reacquire(pick_rt, detector)
                        return stable_pick_target(
                            nearfield,
                            target.name or self._state.target_type or "",
                            log=log.info,
                            visual=True,
                            require_reach=False,
                        )

                    pick_rt._visual_target_observer = _observe_visual_target

                latest_target = [target]

                def _reacquire():
                    before_reacquire(pick_rt, detector)
                    nxt = stable_pick_target(
                        nearfield,
                        target.name or self._state.target_type or "",
                        log=log.info,
                    )
                    if nxt is not None:
                        latest_target[0] = nxt
                        before_graspgenx(detector)
                    return nxt

                pick_rt._reacquire_fn = _reacquire
                pick_deadline = deadline - STAGE2_DELIVERY_RESERVE_S
                pick = None
                total_attempts = 0
                batch = 0
                while c.is_running and c.now_s() < pick_deadline and not picked:
                    batch += 1
                    # Zero cmd_vel does not stop gravity-driven rolling on slopes. Keep
                    # the chassis anchored for staging, inference, contact, and lift.
                    # Match the profile-pick flow: resist rolling without
                    # disturbing the camera-composed yaw during arm motion.
                    with StationaryBase(c, log=log.warning, yaw_kp=0.0):
                        result = pick_rt.run(
                            target, deadline=pick_deadline, reveal=reveal,
                        )
                    total_attempts += result.attempts
                    pick = replace(result, attempts=total_attempts)
                    self._state.adopt_pick(pick)
                    picked = pick.ok
                    target = latest_target[0]
                    if not picked and c.now_s() < pick_deadline:
                        log.info(
                            "[Stage2] pick batch %d failed after %d total attempts; "
                            "retrying until delivery reserve",
                            batch,
                            total_attempts,
                        )
                c.send_cmd_vel(0.0, 0.0)
                if picked:
                    after_successful_pick(pick_rt, detector)
                if pick is None:
                    log.info("[Stage2] delivery reserve reached before first pick attempt")
                    dump_json(dump, "pick.json", {
                        "skipped": True,
                        "reason": "delivery_reserve",
                    })
                else:
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
                    log.info(
                        "[Stage2] pick %s reason=%s attempts=%d grasping=%s "
                        "basket=%s policy=%s",
                        "ok" if picked else "failed",
                        pick.reason,
                        pick.attempts,
                        pick.grasped,
                        pick.basket_occupied,
                        pick.policy,
                    )

        returned_to_owner = self._state.owner_xy is None
        delivered = picked and returned_to_owner
        delivery = None
        if self._state.owner_xy is not None:
            log.info("[Stage2] leg2 -> owner %s", self._state.owner_xy)
            delivery = self._navigate_to(
                self._state.owner_xy,
                deadline,
                tol=GOAL_TOLERANCE,
                inflation_m=STAGE2_INFLATION_M,
                exclude_m=0.6,
                cruise=True,
                leg="delivery",
            )
            returned_to_owner = delivery.reached
            basket_at_owner = (
                pick_rt.poll_basket() is True
                if picked else c.is_basket_occupied() is True
            )
            delivered = returned_to_owner and basket_at_owner
            log.info(
                "[Stage2] leg2 result=%s reason=%s replans=%d basket=%s",
                delivery.status.value,
                delivery.reason,
                delivery.replan_count,
                basket_at_owner,
            )
            dump_json(dump, "delivery.json", {
                "status": delivery.status.value,
                "reason": delivery.reason,
                "reached": delivery.reached,
                "replan_count": delivery.replan_count,
                "owner_xy": list(self._state.owner_xy),
                "pose_xy": list(delivery.pose_xy) if delivery.pose_xy is not None else None,
                "basket_occupied": basket_at_owner,
            })

        if not picked:
            self._stow_arm()

        c.stop()
        if returned_to_owner:
            log.info(
                "[Stage2] owner reached (object_delivered=%s) -> task_complete",
                delivered,
            )
            c.task_complete()
        else:
            log.warning(
                "[Stage2] incomplete: picked=%s owner=%s delivery=%s",
                picked,
                self._state.owner_xy,
                None if delivery is None else delivery.status.value,
            )

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
                xyz = pointcloud2_xyz(cloud) if cloud is not None else None
                if advise(xyz).stop:
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
        """Frozen near-edge search-area goal; refreshes only if the reveal hint jumps."""
        center = self._state.search_xy()
        if center is None:
            return None
        pose = self.client.get_world_pose()
        pose_xy = (pose[0], pose[1]) if pose is not None else center
        self._approach_snap = snapshot_approach_goal(
            self._approach_snap, center, self._state.hint_radius, pose_xy,
        )
        if self._approach_snap is None:
            return None
        return self._approach_snap[2]

    def _navigate_to(self, goal, deadline, tol=GOAL_TOLERANCE,
                     inflation_m=STAGE2_INFLATION_M, exclude_m=0.6, cruise=False,
                     leg="search", search_xy=None, search_radius=0.0,
                     handoff_fn=None):
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
        return nav.navigate_to(
            goal, deadline, leg=leg, tol=tol,
            search_xy=search_xy, search_radius=search_radius,
            handoff_fn=handoff_fn,
        )

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
        fetch_cctv_round(self.client)
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
