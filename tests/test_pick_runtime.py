"""PickRuntime: positive is_grasping / is_basket_occupied gates and bounded retry."""

from __future__ import annotations

import math
import os
import unittest
from types import SimpleNamespace

import numpy as np
import yaml

from marc_agent.pick_runtime import PickRuntime, PickResult, clamp_grasp_xyz
from marc_agent.retrieval_state import RetrievalState


_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "pick_runtime_cases.yaml")


def _load():
    with open(_FIXTURE, encoding="utf-8") as stream:
        return yaml.safe_load(stream)["cases"]


def _flag(value):
    if value is None:
        return None
    return bool(value)


class ScriptClient:
    def __init__(self):
        self.grasping = None
        self.basket = None
        self.is_running = True
        self.t = 0.0
        self.recovered = 0

    def now_s(self):
        return self.t

    def sleep(self, dt):
        self.t += float(dt)

    def is_grasping(self):
        return self.grasping

    def is_basket_occupied(self):
        return self.basket

    def send_cmd_vel(self, *args, **kwargs):
        return None

    def send_arm_command(self, *args, **kwargs):
        return None


def _cloud_target(fwd=0.40, lat=0.0, z=-0.30):
    """Near-field target with a bbox crop large enough for GraspGen-X."""
    depth = np.full((48, 64), 0.55, dtype=np.float32)
    return SimpleNamespace(
        fwd=fwd,
        lat=lat,
        z=z,
        valid=True,
        name="tumbler",
        depth=depth,
        bbox_xyxy=(8, 8, 56, 40),
        k=(50.0, 50.0, 32.0, 24.0),
        rgb_shape=(48, 64),
        R=None,
        t=None,
    )


class PickRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = _load()

    def test_fixture_size(self):
        self.assertGreaterEqual(len(self.cases), 8)
        self.assertEqual(len(self.cases), len({c["id"] for c in self.cases}))

    def test_fixture_cases(self):
        for case in self.cases:
            with self.subTest(case=case["id"]):
                self._run_case(case)

    def _run_case(self, case):
        client = ScriptClient()
        grasps = list(case.get("grasp") or [])
        baskets = list(case.get("basket") or [])
        drops = list(case.get("drop_after_place") or [])
        g_i = [0]
        b_i = [0]

        def grasp(_target):
            if not grasps:
                client.grasping = None
                return
            client.grasping = _flag(grasps[min(g_i[0], len(grasps) - 1)])
            g_i[0] += 1

        def place():
            if not baskets:
                client.basket = None
                return
            idx = min(b_i[0], len(baskets) - 1)
            client.basket = _flag(baskets[idx])
            if drops:
                if _flag(drops[min(idx, len(drops) - 1)]):
                    client.grasping = False
            b_i[0] += 1

        def recover():
            client.recovered += 1
            client.grasping = False

        reacquire_fn = None
        if case.get("reacquire") == "fail":
            reacquire_fn = lambda: SimpleNamespace(valid=False, reason="gone")

        target = SimpleNamespace(fwd=0.40, lat=0.0, z=-0.30, valid=True, name="tumbler")
        if case.get("invalid_target"):
            target = SimpleNamespace(fwd=0.0, lat=0.0, z=0.0, valid=False, reason="no")

        deadline = None
        if "deadline_s" in case:
            deadline = client.now_s() + float(case["deadline_s"])

        runtime = PickRuntime(
            client, log=lambda _m: None,
            max_attempts=int(case.get("max_attempts", 3)),
            policy="ik",
            grasp_fn=grasp,
            place_fn=place,
            recover_fn=recover,
            reacquire_fn=reacquire_fn,
            poll_dt=0.0,
            grasp_poll=2,
            basket_poll=2,
        )
        result = runtime.run(target, deadline=deadline)
        self.assertEqual(case["expect_ok"], result.ok, result)
        self.assertEqual(case["expect_reason"], result.reason, result)
        self.assertEqual(int(case["expect_attempts"]), result.attempts, result)
        self.assertEqual(bool(case["expect_grasped"]), result.grasped, result)
        self.assertEqual(bool(case["expect_basket"]), result.basket_occupied, result)

    def test_none_grasp_is_not_success(self):
        client = ScriptClient()
        client.grasping = None
        client.basket = True
        runtime = PickRuntime(
            client, max_attempts=1, grasp_fn=lambda t: None, place_fn=lambda: None,
            recover_fn=lambda: None, poll_dt=0.0, grasp_poll=2, basket_poll=1,
        )
        target = SimpleNamespace(fwd=0.4, lat=0.0, z=-0.3, valid=True)
        result = runtime.run(target)
        self.assertFalse(result.ok)
        self.assertEqual("no_grasp", result.reason)

    def test_openvla_policy_runs_instead_of_ik(self):
        client = ScriptClient()
        policy = object()
        runtime = PickRuntime(client, policy="openvla", openvla_policy=policy)
        used = []
        runtime._geom_or_openvla_grasp = lambda target, actual: used.append(actual)
        runtime._ik_grasp = lambda target: self.fail("IK should not run when OpenVLA loaded")
        runtime._grasp(SimpleNamespace(fwd=0.4, lat=0.0, z=-0.3))
        self.assertEqual([policy], used)

    def test_openvla_close_uses_teleop_physical_ceiling(self):
        client = ScriptClient()
        client.grasping = False
        commands = []
        arm = SimpleNamespace(
            cfg=SimpleNamespace(grip_close=0.68),
            apply_delta=lambda **kwargs: commands.append(kwargs),
        )
        policy = SimpleNamespace(
            stats=object(),
            act=lambda _obs: np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        )
        runtime = PickRuntime(client, policy="openvla", policy_ticks=1)
        runtime._teleop = arm
        runtime._snapshot_obs = lambda _target: None
        runtime._geom_or_openvla_grasp(SimpleNamespace(), policy)
        self.assertEqual(0.68, commands[-1]["grip"])

    def test_adopt_pick_on_retrieval_state(self):
        state = RetrievalState()
        state.adopt_pick(PickResult(
            ok=True, reason="ok", attempts=2, grasped=True, basket_occupied=True,
        ))
        self.assertTrue(state.picked)
        self.assertTrue(state.basket_occupied)
        self.assertEqual(2, state.pick_attempts)
        self.assertEqual("ok", state.pick_reason)

    def test_production_adapter_does_not_treat_none_as_success(self):
        root = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(root, "demo", "participant_app.py")
        with open(path, encoding="utf-8") as stream:
            src = stream.read()
        self.assertNotIn("held is None", src)
        self.assertNotIn("run_pick_sequence", src)
        self.assertIn("PickRuntime", src)
        self.assertIn("adopt_pick", src)
        self.assertIn("unload_cctv_yolo", src)
        self.assertIn("unload_qwen", src)
        self.assertIn("after_successful_pick", src)

    def test_graspgenx_generator_pose_feeds_ik(self):
        client = ScriptClient()
        client.grasping = True
        client.basket = True
        seen = []
        runtime = PickRuntime(
            client, log=lambda _m: None,
            max_attempts=1, policy="graspgenx",
            place_fn=lambda: None,
            recover_fn=lambda: None,
            poll_dt=0.0, grasp_poll=1, basket_poll=1,
            grasp_generator=lambda _cloud: (0.41, 0.02, -0.27),
        )
        runtime._ik_grasp = lambda target: seen.append(
            (float(target.fwd), float(target.lat), float(target.z))
        )
        result = runtime.run(_cloud_target())
        self.assertTrue(result.ok)
        self.assertEqual([(0.41, 0.02, -0.27)], seen)

    def test_graspgenx_none_skips_ik(self):
        client = ScriptClient()
        client.grasping = False
        client.basket = False
        seen = []
        runtime = PickRuntime(
            client, log=lambda _m: None,
            max_attempts=1, policy="graspgenx",
            place_fn=lambda: None,
            recover_fn=lambda: None,
            poll_dt=0.0, grasp_poll=1, basket_poll=1,
        )
        runtime._ik_grasp = lambda target: seen.append(
            (float(target.fwd), float(target.lat), float(target.z))
        )
        target = SimpleNamespace(fwd=0.40, lat=0.0, z=-0.30, valid=True, name="tumbler")
        result = runtime.run(target)
        self.assertFalse(result.ok)
        self.assertEqual("no_grasp", result.reason)
        self.assertEqual(1, result.attempts)
        self.assertEqual([], seen)
        self.assertEqual("graspgenx", result.policy)

    def test_graspgenx_empty_cloud_skips_ik(self):
        client = ScriptClient()
        client.grasping = False
        client.basket = False
        seen = []
        runtime = PickRuntime(
            client, log=lambda _m: None,
            max_attempts=1, policy="graspgenx",
            place_fn=lambda: None,
            recover_fn=lambda: None,
            poll_dt=0.0, grasp_poll=1, basket_poll=1,
            grasp_generator=lambda _cloud: (0.41, 0.02, -0.27),
        )
        runtime._ik_grasp = lambda target: seen.append(
            (float(target.fwd), float(target.lat), float(target.z))
        )
        target = SimpleNamespace(fwd=0.40, lat=0.0, z=-0.30, valid=True, name="tumbler")
        result = runtime.run(target)
        self.assertFalse(result.ok)
        self.assertEqual("no_grasp", result.reason)
        self.assertEqual(1, result.attempts)
        self.assertEqual([], seen)

    def test_graspgenx_matrix_sets_grasp_yaw(self):
        client = ScriptClient()
        client.grasping = True
        client.basket = True
        seen = []
        yaw = 0.4
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = (0.41, 0.02, -0.27)
        matrix[0, 0] = math.cos(yaw)
        matrix[0, 1] = -math.sin(yaw)
        matrix[1, 0] = math.sin(yaw)
        matrix[1, 1] = math.cos(yaw)
        runtime = PickRuntime(
            client, log=lambda _m: None,
            max_attempts=1, policy="graspgenx",
            place_fn=lambda: None,
            recover_fn=lambda: None,
            poll_dt=0.0, grasp_poll=1, basket_poll=1,
            grasp_generator=lambda _cloud: matrix,
        )
        runtime._ik_grasp = lambda target: seen.append(
            (
                float(target.fwd),
                float(target.lat),
                float(target.z),
                float(getattr(target, "grasp_yaw", float("nan"))),
            )
        )
        result = runtime.run(_cloud_target())
        self.assertTrue(result.ok)
        self.assertEqual(1, len(seen))
        self.assertAlmostEqual(0.41, seen[0][0], places=6)
        self.assertAlmostEqual(yaw, seen[0][3], places=6)

    def test_clamp_grasp_xyz_raises_z_to_floor(self):
        got = clamp_grasp_xyz((0.41, 0.02, -0.44), (0.40, 0.0, -0.22), -0.28)
        self.assertAlmostEqual(0.41, got[0], places=6)
        self.assertAlmostEqual(0.02, got[1], places=6)
        self.assertAlmostEqual(-0.28, got[2], places=6)

    def test_clamp_grasp_xyz_keeps_xy(self):
        loc = (0.40, 0.0, -0.22)
        got = clamp_grasp_xyz((0.48, 0.10, -0.22), loc, -0.28)
        self.assertAlmostEqual(0.48, got[0], places=6)
        self.assertAlmostEqual(0.10, got[1], places=6)
        self.assertAlmostEqual(-0.22, got[2], places=6)

    def test_graspgenx_deep_z_uses_floor_not_rim(self):
        client = ScriptClient()
        client.grasping = True
        client.basket = True
        seen = []
        runtime = PickRuntime(
            client, log=lambda _m: None,
            max_attempts=1, policy="graspgenx",
            place_fn=lambda: None,
            recover_fn=lambda: None,
            poll_dt=0.0, grasp_poll=1, basket_poll=1,
            grasp_generator=lambda _cloud: (0.41, 0.02, -0.44),
        )
        runtime._ik_grasp = lambda target: seen.append(
            (float(target.fwd), float(target.lat), float(target.z))
        )
        result = runtime.run(_cloud_target(z=-0.22))
        self.assertTrue(result.ok)
        self.assertEqual(1, len(seen))
        self.assertAlmostEqual(0.41, seen[0][0], places=6)
        self.assertAlmostEqual(0.02, seen[0][1], places=6)
        self.assertAlmostEqual(-0.28, seen[0][2], places=2)
        self.assertNotAlmostEqual(-0.22, seen[0][2], places=2)

    def test_graspgenx_xy_not_snapped_to_bbox(self):
        client = ScriptClient()
        client.grasping = True
        client.basket = True
        seen = []
        runtime = PickRuntime(
            client, log=lambda _m: None,
            max_attempts=1, policy="graspgenx",
            place_fn=lambda: None,
            recover_fn=lambda: None,
            poll_dt=0.0, grasp_poll=1, basket_poll=1,
            grasp_generator=lambda _cloud: (0.45, 0.04, -0.22),
        )
        runtime._ik_grasp = lambda target: seen.append(
            (float(target.fwd), float(target.lat), float(target.z))
        )
        result = runtime.run(_cloud_target(z=-0.22))
        self.assertTrue(result.ok)
        self.assertEqual(1, len(seen))
        self.assertAlmostEqual(0.45, seen[0][0], places=6)
        self.assertAlmostEqual(0.04, seen[0][1], places=6)

    def test_graspgenx_far_from_yolo_skips_ik(self):
        client = ScriptClient()
        client.grasping = False
        client.basket = False
        seen = []
        runtime = PickRuntime(
            client, log=lambda _m: None,
            max_attempts=1, policy="graspgenx",
            place_fn=lambda: None,
            recover_fn=lambda: None,
            poll_dt=0.0, grasp_poll=1, basket_poll=1,
            grasp_generator=lambda _cloud: (0.35, -0.176, -0.059),
        )
        runtime._ik_grasp = lambda target: seen.append(
            (float(target.fwd), float(target.lat), float(target.z))
        )
        result = runtime.run(_cloud_target(fwd=0.251, lat=-0.005, z=-0.016))
        self.assertFalse(result.ok)
        self.assertEqual("no_grasp", result.reason)
        self.assertEqual(1, result.attempts)
        self.assertEqual([], seen)

    def test_graspgenx_aerial_z_skips_ik(self):
        client = ScriptClient()
        client.grasping = False
        client.basket = False
        seen = []
        runtime = PickRuntime(
            client, log=lambda _m: None,
            max_attempts=1, policy="graspgenx",
            place_fn=lambda: None,
            recover_fn=lambda: None,
            poll_dt=0.0, grasp_poll=1, basket_poll=1,
            grasp_generator=lambda _cloud: (0.40, 0.0, 0.128),
        )
        runtime._ik_grasp = lambda target: seen.append(
            (float(target.fwd), float(target.lat), float(target.z))
        )
        result = runtime.run(_cloud_target(z=-0.023))
        self.assertFalse(result.ok)
        self.assertEqual("no_grasp", result.reason)
        self.assertEqual([], seen)

    def test_graspgenx_out_of_reach_skips_ik(self):
        client = ScriptClient()
        client.grasping = False
        client.basket = False
        seen = []
        runtime = PickRuntime(
            client, log=lambda _m: None,
            max_attempts=1, policy="graspgenx",
            place_fn=lambda: None,
            recover_fn=lambda: None,
            poll_dt=0.0, grasp_poll=1, basket_poll=1,
            grasp_generator=lambda _cloud: (0.60, 0.20, -0.22),
        )
        runtime._ik_grasp = lambda target: seen.append(
            (float(target.fwd), float(target.lat), float(target.z))
        )
        result = runtime.run(_cloud_target(z=-0.22))
        self.assertFalse(result.ok)
        self.assertEqual("no_grasp", result.reason)
        self.assertEqual(1, result.attempts)
        self.assertEqual([], seen)

    def test_practice_client_picks_when_is_running_false(self):
        client = ScriptClient()
        client.is_running = False
        client.grasping = True
        client.basket = True
        grasped = []
        runtime = PickRuntime(
            client, log=lambda _m: None,
            max_attempts=1, policy="ik",
            grasp_fn=lambda t: grasped.append(t),
            place_fn=lambda: None,
            recover_fn=lambda: None,
            poll_dt=0.0, grasp_poll=1, basket_poll=1,
        )
        target = SimpleNamespace(fwd=0.40, lat=0.0, z=-0.30, valid=True)
        result = runtime.run(target)
        self.assertTrue(result.ok)
        self.assertEqual("ok", result.reason)
        self.assertEqual(1, len(grasped))

    def test_stage2_session_end_is_timeout(self):
        client = ScriptClient()
        client.is_running = False
        client._stage2_active = True
        grasped = []
        runtime = PickRuntime(
            client, log=lambda _m: None,
            max_attempts=1, policy="ik",
            grasp_fn=lambda t: grasped.append(t),
            place_fn=lambda: None,
            recover_fn=lambda: None,
            poll_dt=0.0, grasp_poll=1, basket_poll=1,
        )
        target = SimpleNamespace(fwd=0.40, lat=0.0, z=-0.30, valid=True)
        result = runtime.run(target)
        self.assertFalse(result.ok)
        self.assertEqual("timeout", result.reason)
        self.assertEqual([], grasped)


if __name__ == "__main__":
    unittest.main()
