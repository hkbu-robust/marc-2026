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
import time

from marc_sdk import MARCClient

from mock_agent_vla import MockVLAGrounding
from mock_agent_navigation import (
    load_obstacles, obstacles_excluding_goal, load_mock_data,
    lidar_cloud_to_obstacles, frontal_hit, run_nav_leg,
)
from nearfield import NearFieldAcquire, make_detector
from retrieval_state import RetrievalState
# Closed-loop placo pick from the manipulation kit (arm_pick.py): grasp_yaw wrist-roll +
# DLS servo, targeting the object position. Requires placo (installed in the demo image).
# The fixed-keyframe mock in mock_agent_manipulation.py remains as a dependency-free
# fallback reference.
from arm_pick import run_pick_sequence

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
GOAL_TOLERANCE = 0.7           # reached test (m)
# Opt-in yaml discs (unavailable in scoring). Live lidar is the default overlay.
NAV_USE_MOCK_OBSTACLES = os.environ.get("NAV_USE_MOCK_OBSTACLES", "").strip().lower() in (
    "1", "true", "yes", "on",
)

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

# -- Stage 2 delivery goal --
# The Stage 2 task drives to the target object and then delivers it to the owner zone.
# The pickup search area comes from the reveal (msg 411) at runtime, with the
# Stage 2 grounding answer's target_coord as a fallback. The
# delivery goal comes from owner_position in the mission (msg 211), with delivery_coord from
# mock_demo_data.yaml as a fallback. delivery_coord is pre-extracted from the scenario by
# tools/mock_data_builder/gen_mock_demo_data.py; real participants do not have the scenario.
_DELIVERY_COORD = load_mock_data().get("delivery_coord") or [-55.29138, 142.09763, 16.47326]
DELIVERY_XY = (_DELIVERY_COORD[0], _DELIVERY_COORD[1])


class DemoParticipant:
    """SDK-based demo participant -- connects the VLA/nav logic through handlers."""

    def __init__(self):
        self.client = MARCClient.from_env()
        self.vla = MockVLAGrounding()
        self._state = RetrievalState()
        self._reveal = None
        # Yaml discs only when debugging without lidar. Scoring has no mock_demo_data.yaml.
        self._obstacles = load_obstacles() if NAV_USE_MOCK_OBSTACLES else []
        if NAV_USE_MOCK_OBSTACLES:
            log.info("[Stage2] NAV_USE_MOCK_OBSTACLES=1 -- injecting %d yaml discs",
                     len(self._obstacles))
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
        c.on_time_expired(self._on_time_expired)

    # -- Stage 1 --

    def _on_mission(self, mission):
        log.info("[Stage1] round %s/%s: \"%s\"",
                 mission.round, mission.total_rounds, mission.voice_command)
        result = self.vla.process(mission.voice_command, self.client.list_cctv())
        # result is a GroundingResult -- submit it directly.
        self.client.submit_grounding(result)

    # -- Stage 2: submit interpretation -> drive --

    def _on_stage2_mission(self, mission):
        log.info("[Stage2] task: \"%s\" (limit %.0fs)",
                 mission.task_description, mission.time_limit)
        # Together with the VLA command, the platform provides the delivery goal (owner) coordinates (msg 211 owner_position).
        op = getattr(mission, "owner_position", None)
        if op and len(op) >= 2:
            self._state.adopt_owner(op)
            log.info("[Stage2] owner_position(runtime-provided)=%s", self._state.owner_xy)
        result = self.vla.process_stage2(mission.task_description)
        self.client.submit_stage2_grounding(result)
        self._state.adopt_grounding(result.target_coord)
        log.info("[Stage2] submitted grounding target=%s, delivery point=%s",
                 self._state.grounding_xy, self._state.owner_xy or DELIVERY_XY)

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
        """Called once in a separate thread on entering STAGE2_RUN -- pick up the object then deliver it.

        Flow: stow the arm to lower the CoG -> D* to the reveal search disc ->
        RGB-D reacquire inside the disc -> pick (read grip-hold) -> stow again if
        empty-handed -> deliver to the owner zone -> task_complete.
        """
        c = self.client
        # SIM-time deadline (via /clock) -- shared across all legs, robust to real-time factor < 1.
        deadline = c.now_s() + STAGE2_NAV_TIMEOUT_S

        self._wait_for_search_goal(timeout=5.0)
        pickup_xy = self._state.owner_xy if self._state.owner_xy is not None else DELIVERY_XY
        search = self._state.search_xy()
        log.info("[Stage2] navigation start -- search=%s r=%.1f type=%s deliver=%s (timeout %.0fs)",
                 search, self._state.hint_radius, self._state.target_type or "-",
                 pickup_xy, STAGE2_NAV_TIMEOUT_S)

        self._stow_arm()

        picked = False
        reached_search = False
        if search is None:
            log.warning("[Stage2] no search goal (no reveal, no grounding) -- skip nav and pick")
        else:
            log.info("[Stage2] leg1 -> search area %s tol=%.2f (hint r=%.2f)",
                     tuple(round(v, 2) for v in search),
                     self._state.search_tol(GOAL_TOLERANCE), self._state.hint_radius)
            reached_search = self._navigate_to(
                self._state.search_xy,
                deadline,
                tol=self._state.search_tol,
                inflation_m=STAGE2_INFLATION_M,
                exclude_m=0.6,
            )
        if not c.is_running:
            return
        if not reached_search:
            log.warning("[Stage2] search area not reached -- skip pick (fail closed)")
        else:
            reveal = self._reveal if self._reveal is not None else c.stage2_reveal
            target = NearFieldAcquire(c, make_detector()).acquire(reveal, deadline)
            for _ in range(5):
                c.send_cmd_vel(0.0, 0.0)
                c.sleep(0.1)
            if not target.valid:
                log.warning("[Stage2] near-field acquire failed (%s) -- skip pick (fail closed)",
                            target.reason)
            else:
                log.info("[Stage2] localized %s grasp=(%.2f, %.2f, z=%.2f) conf=%.2f",
                         target.name or "?", target.fwd, target.lat, target.z, target.conf)
                try:
                    held = run_pick_sequence(c, log, (target.fwd, target.lat))
                    c.send_cmd_vel(0.0, 0.0)
                    picked = True if held is None else bool(held)
                    log.info("[Stage2] pick %s (grip held=%s)", "ok" if picked else "failed", held)
                except Exception as e:
                    log.warning("[Stage2] pick failed: %s", e)

        # If the arm is left raised (front_high) without an object, driving the delivery leg with
        # that top-heavy CoG can tip the robot over on turns. Empty-handed -> fold to HOME.
        if not picked:
            self._stow_arm()

        # -- leg 2: deliver to the owner zone --
        log.info("[Stage2] leg2 -> deliver to owner zone %s", pickup_xy)
        reached = self._navigate_to(pickup_xy, deadline)

        c.stop()
        log.info("[Stage2] driving finished (delivery_reached=%s) -> task_complete", reached)
        c.task_complete()

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
                if self._obstacles:
                    obs = list(self._obstacles) + obs
                obs = obstacles_excluding_goal(obs, obj_xy, exclude_m=0.6)
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

    def _navigate_to(self, goal, deadline, tol=GOAL_TOLERANCE,
                     inflation_m=STAGE2_INFLATION_M, exclude_m=0.6, cruise=False):
        """Drive to a goal (x, y) or callable via live occupancy D* Lite."""
        return run_nav_leg(
            self.client,
            goal,
            deadline,
            tol=tol,
            inflation_m=inflation_m,
            exclude_m=exclude_m,
            cruise=cruise,
            extra_obstacles=self._obstacles,
            dt=STAGE2_NAV_DT,
            stuck_time_s=STAGE2_STUCK_TIME_S,
            stuck_move_m=STAGE2_STUCK_MOVE_M,
            replan_cap=STAGE2_REPLAN_CAP,
        )

    # -- notifications --

    def _on_score(self, score):
        if score.is_final:
            log.info("[FINAL] %s", score.scores)
        else:
            log.info("[Score] round %s: total=%s", score.round, score.total)

    def _on_state_change(self, old, new):
        log.info("[STATE] %s -> %s", old, new)

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
    DemoParticipant().run()


if __name__ == "__main__":
    main()
