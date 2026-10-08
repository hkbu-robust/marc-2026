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
import time

from marc_sdk import MARCClient

from marc_agent.command_parser import Intent
from marc_agent.experiment_log import (
    current_experiment,
    decision_scope,
    start_experiment,
)
from marc_agent.navigation_runtime import NavStatus, Stage2Navigator
from marc_agent.gpu_handoff import (
    after_successful_pick,
    before_graspgenx,
    before_reacquire,
    unload_cctv_yolo,
    unload_qwen,
)
from marc_agent.perception.round_dump import (
    dump_stage_artifacts,
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
    StaticSceneCache,
    fallback_grounding,
    get_default_verifier,
    log_qwen_startup_status,
    run_stage1_round,
    warm_up_qwen_before_registering,
)
from marc_agent.pick_runtime import PickRuntime, pick_policy_name
from marc_agent.retrieval_state import (
    RetrievalState,
    nearfield_start_reason,
    search_area_approach_goal,
    snapshot_approach_goal,
)
from marc_agent.round_clock import RoundClock
from marc_agent.stage2_delivery import Stage2Deadline, Stage2Delivery
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
STAGE2_NAV_RATE_HZ = 10.0
STAGE2_NAV_DT = 1.0 / STAGE2_NAV_RATE_HZ
STAGE2_INFLATION_M = 0.55      # body radius plus curb pad (keeps wheels off the grass)
STAGE2_STUCK_TIME_S = 8.0      # if movement < threshold for this long, replan from the current pose
STAGE2_STUCK_MOVE_M = 0.05
STAGE2_REPLAN_CAP = 50         # fail closed if the crop stays sealed
GOAL_TOLERANCE = 0.7           # reached test (m) around the search-area approach goal
STAGE2_APPROACH_TIMEOUT_S = 120.0

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
        self._experiment = current_experiment()
        self._experiment.set_simulation_clock(self.client.now_s)
        # Stage 1 never constructs or reads the scenario-backed mock. Keep the legacy
        # Stage 2 demo adapter lazy until that separate mission path is invoked.
        self._stage2_vla = None
        self._state = RetrievalState()
        self._reveal = None
        self._last_detections = None
        self._last_stage1_candidates = None
        self._stage1_cache = StaticSceneCache.from_env()
        self._stage1_lock = threading.Lock()
        self._stage1_projector = None
        self._round_clock = RoundClock()
        self._stage2_clock = Stage2Deadline()
        self._qwen_verifier = get_default_verifier()
        self._approach_snap = None
        self._final_score_payload = None
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
        self._experiment.event(
            "stage1.round.received", stage="stage1", status="started",
            round=mission.round, total_rounds=mission.total_rounds,
            time_limit_s=getattr(mission, "time_limit", None),
        )
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
            except Exception as exc:
                self._experiment.event(
                    "stage1.round.completed", stage="stage1", status="error",
                    terminal=True, round=mission.round,
                    error_type=type(exc).__name__,
                )
                log.exception("[Stage1] round %s failed; submitting the fallback", mission.round)
                # Never end a round without a submission (#114).
                cameras = getattr(self._last_detections, "camera_ids", ()) or ()
                try:
                    seq = self.client.submit_grounding(
                        fallback_grounding(mission.voice_command, cameras))
                    self._experiment.event(
                        "stage1.submission.completed", stage="stage1", round=mission.round,
                        sequence=seq, method="fallback_after_error",
                    )
                except Exception:
                    log.exception("[Stage1] fallback submission failed")

    def _execute_stage1(self, mission):
        timeout_s = discovery_timeout_from_limit(getattr(mission, "time_limit", None))
        if self._stage1_projector is None:
            self._stage1_projector = CctvProjector.from_client(self.client)
        self._round_clock.begin(getattr(mission, "time_limit", None), self.client.now_s())
        with decision_scope("stage1", mission.round) as decisions:
            round_result = run_stage1_round(
                self.client,
                mission.voice_command,
                discovery_timeout_s=timeout_s,
                projector=self._stage1_projector,
                clock=self._round_clock,
                verifier=self._qwen_verifier,
            )
            round_result = self._stage1_cache.apply(round_result, mission.round)
        self._last_stage1_candidates = round_result.candidates
        self._last_detections = round_result.candidates.detections
        dump_path = dump_stage_artifacts(stage1_dir(mission.round), round_result)
        self._experiment.event(
            "stage1.pipeline.completed",
            stage="stage1",
            status="failed" if round_result.grounding is None else "ok",
            terminal=round_result.grounding is None,
            round=mission.round,
            artifact_dir=self._experiment.artifact_ref(dump_path),
            decisions=decisions,
            **round_result.record(),
        )
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
        grounding = round_result.grounding
        if grounding is None:  # never end a round without a submission (#114)
            log.info("[Stage1] no grounding (reason=%s); submitting the fallback",
                     round_result.failure_reason)
            grounding = fallback_grounding(mission.voice_command,
                                           self._last_detections.camera_ids)
        seq = self.client.submit_grounding(grounding)
        self._experiment.event(
            "stage1.submission.completed", stage="stage1", round=mission.round,
            sequence=seq, camera_id=grounding.camera_id,
            target_id=round_result.target_id, anchor_id=round_result.anchor_id,
            method=round_result.method,
            target_coord=grounding.target_coord,
            anchor_coord=grounding.anchor_coord,
            relation=grounding.relation,
            landmark=grounding.landmark,
            situation=grounding.situation,
            remaining_s=self._round_clock.remaining(self.client.now_s()),
        )
        self._experiment.event(
            "stage1.round.completed", stage="stage1", round=mission.round,
        )
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
        self._experiment.event(
            "stage2.mission.received", stage="stage2", status="started",
            time_limit_s=mission.time_limit, task=mission.task_description,
            owner_position=getattr(mission, "owner_position", None),
        )
        # Store the runtime-provided delivery coordinate for the post-pick leg.
        self._state.adopt_mission(mission)
        self._approach_snap = None
        deadline = self._stage2_clock.begin(mission.time_limit, self.client.now_s())
        log.info(
            "[Stage2] platform deadline=%.2f source=%s entry=%s guard=%.2f",
            deadline,
            self._stage2_clock.source,
            self._stage2_clock.entry_s,
            self._stage2_clock.config.platform_guard_s,
        )
        if self._state.owner_xy is not None:
            log.info("[Stage2] owner_position(runtime-provided)=%s", self._state.owner_xy)
        if os.environ.get("MARC_STAGE2_MOCK_VLA", "").strip() == "1":
            if self._stage2_vla is None:
                self._stage2_vla = MockVLAGrounding()
            result = self._stage2_vla.process_stage2(mission.task_description)
            self.client.submit_stage2_grounding(result)
            self._experiment.event(
                "stage2.grounding.completed", stage="stage2", method="mock_vla",
                target_coord=getattr(result, "target_coord", None),
            )
            self._state.adopt_grounding(result.target_coord)
            log.info("[Stage2] submitted mock grounding target=%s (owner stored=%s)",
                     self._state.grounding_xy, self._state.owner_xy)
            return
        if self._stage1_projector is None:
            self._stage1_projector = CctvProjector.from_client(self.client)
        timeout_s = discovery_timeout_from_limit(getattr(mission, "time_limit", None))
        with decision_scope("stage2") as decisions:
            round_result = run_stage1_round(
                self.client,
                mission.task_description,
                expected_intent=Intent.BRING,
                projector=self._stage1_projector,
                discovery_timeout_s=timeout_s,
                clock=self._stage2_clock,
                verifier=self._qwen_verifier,
            )
        dump_path = dump_stage_artifacts(stage2_dir("grounding"), round_result)
        self._experiment.event(
            "stage2.grounding.pipeline.completed",
            stage="stage2",
            status="failed" if round_result.grounding is None else "ok",
            terminal=round_result.grounding is None,
            artifact_dir=self._experiment.artifact_ref(dump_path),
            decisions=decisions,
            **round_result.record(),
        )
        grounding = round_result.grounding
        if grounding is None:
            log.info(
                "[Stage2] skip submit reason=%s (no zero-coordinate fallback)",
                round_result.failure_reason,
            )
            self._experiment.event(
                "stage2.mission.completed", stage="stage2", status="failed",
                terminal=True, failure_reason=round_result.failure_reason,
            )
            return
        self.client.submit_stage2_grounding(grounding)
        self._experiment.event(
            "stage2.grounding.submitted", stage="stage2",
            camera_id=grounding.camera_id, target_id=round_result.target_id,
            method=round_result.method,
            target_coord=grounding.target_coord,
            anchor_coord=grounding.anchor_coord,
            relation=grounding.relation,
            landmark=grounding.landmark,
        )
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
        self._experiment.event(
            "stage2.reveal.received", stage="stage2",
            grounding_score=reveal.grounding_score,
            target_type=reveal.target_type,
            hint_center=reveal.hint_center,
            hint_radius=reveal.hint_radius,
        )

    def _drive(self):
        """Called once in a separate thread on entering STAGE2_RUN.

        Leg 1: D* to the reveal search-area approach goal. #12 localizes in RGB-D,
        then #13 grasps only on positive is_grasping() and loads the basket.
        Leg 2 carries the basket to the runtime-provided owner_position.
        """
        c = self.client
        self._approach_snap = None
        # SIM-time deadline (via /clock) -- shared across all legs, robust to real-time factor < 1.
        def deadline():
            return float(self._stage2_clock.deadline_s or c.now_s())

        navigation_dump = stage2_dir("navigation")
        nearfield_dump = stage2_dir("nearfield")
        manipulation_dump = stage2_dir("manipulation")

        delivery_runtime = Stage2Delivery(
            c,
            self._stage2_clock,
            self._state.owner_xy,
            self._plan_owner_route,
            self._navigate_owner,
            log=log,
        )

        self._wait_for_search_goal(timeout=5.0)
        search = self._state.search_xy()
        log.info("[Stage2] navigation start -- search=%s r=%.1f type=%s (timeout %.0fs)",
                 search, self._state.hint_radius, self._state.target_type or "-",
                 max(0.0, deadline() - c.now_s()))

        self._stow_arm()
        gpu_handoff_started = time.perf_counter()
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
            c, detector, dump_dir=nearfield_dump, people_xy=self._state.people_xy,
        )
        self._experiment.event(
            "stage2.gpu_handoff.completed", stage="stage2",
            duration_ms=(time.perf_counter() - gpu_handoff_started) * 1000.0,
            loaded="stage2_nearfield_yolo", unloaded=["stage1_cctv_yolo", "qwen"],
        )

        picked = False
        pick_rt = None
        reached_search = False
        handoff = "skip_no_search"
        result = None
        navigation_started = time.perf_counter()
        if search is None:
            log.warning("[Stage2] no search goal (no reveal, no grounding) -- skip nav and pick")
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

            while c.is_running and c.now_s() < deadline():
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
        self._experiment.event(
            "stage2.navigation.completed",
            stage="stage2",
            status="ok" if reached_search else "failed",
            terminal=(not reached_search and str(handoff).startswith("skip")),
            duration_ms=(time.perf_counter() - navigation_started) * 1000.0,
            reached=reached_search,
            result_status=None if result is None else result.status.value,
            reason=None if result is None else result.reason,
            replan_count=0 if result is None else result.replan_count,
            stall_count=0 if result is None else result.stall_count,
            lidar_interventions=0 if result is None else result.lidar_interventions,
            recovery_count=0 if result is None else result.recovery_count,
            planner_method=None if result is None else result.last_method,
            goal_xy=None if result is None else result.goal_xy,
            path_waypoints=0 if result is None or not result.path else len(result.path),
            path_length_m=(
                0.0 if result is None or not result.path else sum(
                    math.hypot(b[0] - a[0], b[1] - a[1])
                    for a, b in zip(result.path, result.path[1:])
                )
            ),
            leg="pickup",
            handoff=handoff,
            artifact_dir=self._experiment.artifact_ref(navigation_dump),
        )
        if not c.is_running:
            return
        start_yolo = not str(handoff).startswith("skip")
        if not start_yolo:
            log.warning("[Stage2] skip pick -- near-field not started (%s)", handoff)
            self._experiment.event(
                "stage2.nearfield.completed", stage="stage2", status="skipped",
                reason=handoff,
                artifact_dir=self._experiment.artifact_ref(nearfield_dump),
            )
            self._experiment.event(
                "stage2.manipulation.completed", stage="stage2", status="skipped",
                reason=handoff,
                artifact_dir=self._experiment.artifact_ref(manipulation_dump),
            )
        else:
            log.info("[Stage2] starting calibrated near-field approach (%s)", handoff)
            reveal = self._reveal if self._reveal is not None else c.stage2_reveal
            nearfield_started = time.perf_counter()
            stable_consensus = None
            approach_target = None
            approach_deadline = min(deadline(), c.now_s() + STAGE2_APPROACH_TIMEOUT_S)
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
            target = approach_target or nearfield.acquire(reveal, deadline())
            with StationaryBase(c, log=log.warning, yaw_kp=0.0):
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
                        stable_consensus = False
                        target.valid = False
                        target.reason = "unstable_localization"
                    else:
                        stable_consensus = True
                        target = stable
            self._experiment.event(
                "stage2.nearfield.completed",
                stage="stage2",
                status="ok" if target.valid else "failed",
                terminal=not target.valid,
                duration_ms=(time.perf_counter() - nearfield_started) * 1000.0,
                target_type=target.name or self._state.target_type,
                confidence=target.conf,
                reason=target.reason,
                bbox_xyxy=target.bbox_xyxy,
                target_arm_xyz=(target.fwd, target.lat, target.z),
                depth_valid=target.depth is not None,
                detector=getattr(detector, "metrics", None),
                localization=getattr(nearfield, "diagnostics", None),
                stable_consensus=stable_consensus,
                artifact_dir=self._experiment.artifact_ref(nearfield_dump),
            )
            if not target.valid:
                log.warning("[Stage2] near-field acquire failed (%s) -- skip pick (fail closed)",
                            target.reason)
                self._experiment.event(
                    "stage2.manipulation.completed", stage="stage2", status="skipped",
                    reason=target.reason,
                    artifact_dir=self._experiment.artifact_ref(manipulation_dump),
                )
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
                # Zero cmd_vel does not stop gravity-driven rolling on slopes. Keep
                # the chassis anchored for staging, inference, contact, and lift.
                manipulation_started = time.perf_counter()
                # Match the profile-pick flow: resist rolling without disturbing
                # the camera-composed yaw during arm motion.
                with StationaryBase(c, log=log.warning, yaw_kp=0.0):
                    pick = pick_rt.run(
                        target,
                        deadline=deadline,
                        reveal=reveal,
                        retry_allowed=delivery_runtime.retry_allowed,
                    )
                self._experiment.event(
                    "stage2.manipulation.completed",
                    stage="stage2",
                    status="ok" if pick.ok else "failed",
                    terminal=not pick.ok,
                    duration_ms=(time.perf_counter() - manipulation_started) * 1000.0,
                    reason=pick.reason,
                    attempts=pick.attempts,
                    grasped=pick.grasped,
                    basket_occupied=pick.basket_occupied,
                    policy=pick.policy,
                    artifact_dir=self._experiment.artifact_ref(manipulation_dump),
                )
                self._state.adopt_pick(pick)
                picked = pick.ok
                target = latest_target[0]
                c.send_cmd_vel(0.0, 0.0)
                if picked:
                    after_successful_pick(pick_rt, detector)
                log.info("[Stage2] pick %s reason=%s attempts=%d grasping=%s basket=%s policy=%s",
                         "ok" if picked else "failed", pick.reason, pick.attempts,
                         pick.grasped, pick.basket_occupied, pick.policy)

        if not picked:
            self._stow_arm()

        def _basket_state():
            try:
                value = c.is_basket_occupied()
            except Exception:
                return None
            return None if value is None else bool(value)

        def _recover_drop():
            if pick_rt is None:
                return False
            before_reacquire(pick_rt, detector)
            dropped_target = stable_pick_target(
                nearfield,
                self._state.target_type or "",
                log=log.info,
            )
            if dropped_target is None:
                return False
            with StationaryBase(c, log=log.warning, yaw_kp=0.0):
                recovered = pick_rt.run(
                    dropped_target,
                    deadline=deadline,
                    retry_allowed=lambda: False,
                )
            return recovered.ok

        log.info("[Stage2] leg2 -> owner %s", self._state.owner_xy)
        delivery_started = time.perf_counter()
        delivery = delivery_runtime.return_to_owner(
            _basket_state,
            _recover_drop if picked else None,
        )
        self._experiment.event(
            "stage2.delivery.completed",
            stage="stage2",
            status="ok" if delivery.completion_sent else "failed",
            terminal=not delivery.completion_sent,
            duration_ms=(time.perf_counter() - delivery_started) * 1000.0,
            leg="delivery",
            result=delivery.status,
            reason=delivery.reason,
            owner_xy=delivery.owner_xy,
            owner_distance_m=delivery.owner_distance_m,
            basket_occupied=delivery.basket_occupied,
            basket_dropped=delivery.basket_dropped,
            recovery_attempted=delivery.recovery_attempted,
            recovery_succeeded=delivery.recovery_succeeded,
            navigation_status=delivery.navigation_status,
            completion_sent=delivery.completion_sent,
        )
        self._experiment.event(
            "stage2.mission.completed",
            stage="stage2",
            status="ok" if delivery.completion_sent else "failed",
            terminal=True,
            failure_reason=None if delivery.completion_sent else delivery.reason,
            picked=picked,
            reached_search=reached_search,
            handoff=handoff,
            delivered=delivery.completion_sent,
        )
        log.info(
            "[Stage2] delivery status=%s reason=%s owner_distance=%s "
            "basket=%s dropped=%s recovery=%s complete=%s",
            delivery.status,
            delivery.reason,
            delivery.owner_distance_m,
            delivery.basket_occupied,
            delivery.basket_dropped,
            delivery.recovery_succeeded,
            delivery.completion_sent,
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
                     handoff_fn=None, observe_fn=None):
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
        result = nav.navigate_to(
            goal, deadline, leg=leg, tol=tol,
            search_xy=search_xy, search_radius=search_radius,
            handoff_fn=handoff_fn, observe_fn=observe_fn,
        )
        result.stall_count = nav.stall_count
        result.lidar_interventions = nav.lidar_interventions
        result.recovery_count = nav.recovery_count
        return result

    def _plan_owner_route(self, owner_xy):
        nav = Stage2Navigator(
            self.client,
            dt=STAGE2_NAV_DT,
            inflation_m=STAGE2_INFLATION_M,
            exclude_m=0.6,
            stuck_time_s=STAGE2_STUCK_TIME_S,
            stuck_move_m=STAGE2_STUCK_MOVE_M,
            replan_cap=STAGE2_REPLAN_CAP,
            cruise=True,
        )
        return nav.plan_to(owner_xy)

    def _navigate_owner(self, owner_xy, deadline, observe_fn):
        return self._navigate_to(
            owner_xy,
            deadline,
            tol=self._stage2_clock.config.owner_tolerance_m,
            inflation_m=STAGE2_INFLATION_M,
            exclude_m=0.6,
            cruise=True,
            leg="delivery",
            observe_fn=observe_fn,
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
        self._experiment.event("score.received", **payload)
        if score.is_final:
            log.info("[FINAL] %s", score.scores)
            self._final_score_payload = payload
            self._experiment.finish(
                "completed", final_score=score.scores, total=score.total,
            )
        else:
            log.info("[Score] round %s: total=%s", score.round, score.total)

    def _on_state_change(self, old, new):
        log.info("[STATE] %s -> %s", old, new)
        self._experiment.event("platform.state.changed", old=old, new=new)
        if new == "STAGE2_RUN":
            self._stage2_clock.enter(self.client.now_s())
        elif old == "STAGE2_RUN" and new != "STAGE2_RUN":
            self._stage2_clock.leave()

    def _on_time_remaining(self, remaining):
        self._round_clock.note(remaining, self.client.now_s())
        self._stage2_clock.note_remaining(remaining, self.client.now_s())

    def _on_time_expired(self, which):
        log.info("[TIME] expired: %s", which)
        self._experiment.event(
            "platform.time.expired", status="timeout", terminal=True, which=which,
        )

    # -- run --

    def run(self):
        if not self.client.connect():
            raise SystemExit("Registration failed -- check the runtime and the token.")
        fetch_cctv_round(self.client)
        self.client.run()
        if self._final_score_payload is not None:
            self._experiment.finish(
                "completed",
                final_score=self._final_score_payload["scores"],
                total=self._final_score_payload["total"],
            )


def main():
    experiment = start_experiment()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    experiment.attach_python_logging()
    try:
        log_qwen_startup_status()
        warm_up_qwen_before_registering()
        DemoParticipant().run()
    except BaseException as exc:
        log.exception("Participant terminated")
        experiment.finish("failed", error_type=type(exc).__name__)
        raise
    else:
        experiment.finish("stopped")


if __name__ == "__main__":
    main()
