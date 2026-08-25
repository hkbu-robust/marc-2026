"""PickRuntime: positive is_grasping / is_basket_occupied gates and bounded retry."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

import yaml

from marc_agent.pick_runtime import PickRuntime, PickResult
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


if __name__ == "__main__":
    unittest.main()
