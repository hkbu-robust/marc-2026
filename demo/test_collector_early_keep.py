"""Placement-time episode acceptance tests."""

import time
import unittest
from types import SimpleNamespace

from manip_dataset.app import CollectorApp


class _Keyboard:
    def __init__(self, keys):
        self._keys = iter(keys)

    def poll(self):
        return next(self._keys, None)


class _Client:
    def is_grasping(self):
        return True


class _Arm:
    def __init__(self):
        self.placed = False
        self.cancelled = False

    def place_in_basket(self, on_step=None, cancel_event=None):
        self.placed = True
        if cancel_event is None:
            return
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.wait(0.005):
                self.cancelled = True
                return


class CollectorEarlyKeepTests(unittest.TestCase):
    def _app(self):
        app = object.__new__(CollectorApp)
        app.client = _Client()
        app.arm = _Arm()
        app._running = True
        return app

    def test_e_during_placement_cancels_and_manually_keeps(self):
        app = self._app()
        app._keyboard = _Keyboard(["E"])
        ended = []
        app._end_attempt = lambda **kwargs: ended.append(kwargs)
        app._wait_basket_occupied = lambda: self.fail(
            "latched E should use the normal end-attempt path"
        )

        app._place_in_basket()

        self.assertTrue(app.arm.cancelled)
        self.assertEqual(
            [{"manual_accept": True, "open_grip_on_reset": True}], ended
        )

    def test_without_early_e_placement_still_waits_for_manual_keep(self):
        app = self._app()
        app._keyboard = None
        app._wait_basket_occupied = lambda: True
        ended = []
        app._end_attempt = lambda: ended.append(True)

        app._place_in_basket()

        self.assertEqual([], ended)

    def test_manual_approval_keeps_without_basket_and_resets_open(self):
        app = self._app()
        app._base_hold = SimpleNamespace(stop=lambda: None)
        app.client.send_cmd_vel = lambda *_args, **_kwargs: None
        app.client.is_basket_occupied = lambda: False
        app.recorder = SimpleNamespace(
            steps=[{"action": "pick"}],
            recording=True,
            keep=lambda: "/tmp/kept-episode",
            discard=lambda **_kwargs: self.fail("manual approval must keep"),
        )
        app.arm.refresh_grasp = lambda: self.fail(
            "manual approval must not wait for grasp sensors"
        )
        resets = []
        app.begin_episode_cycle = lambda open_grip=False: resets.append(open_grip)

        app._end_attempt(manual_accept=True, open_grip_on_reset=True)

        self.assertEqual([True], resets)


if __name__ == "__main__":
    unittest.main()
