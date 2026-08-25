"""OpenVLA policy stub and weight-loader fallback (no torch required)."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from marc_agent.openvla_policy import (
    ACTION_DIM,
    GeometricGraspStub,
    PolicyObservation,
    load_openvla,
    resolve_weights_path,
)


class OpenVLAPolicyTests(unittest.TestCase):
    def test_geom_stub_moves_then_closes(self):
        stub = GeometricGraspStub((0.40, 0.0, -0.30), step_m=0.05, close_m=0.04)
        far = PolicyObservation(ee_xyz=np.array([0.10, 0.0, 0.0]))
        action = stub.act(far)
        self.assertEqual((ACTION_DIM,), action.shape)
        self.assertGreater(float(action[0]), 0.0)
        self.assertEqual(0.0, float(action[6]))
        near = PolicyObservation(ee_xyz=np.array([0.40, 0.0, -0.30]))
        close = stub.act(near)
        self.assertEqual(1.0, float(close[6]))
        np.testing.assert_allclose(close[:3], 0.0, atol=1e-6)

    def test_load_without_weights_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_openvla(tmp))
            self.assertIsNone(resolve_weights_path(tmp))

    def test_load_missing_path_is_none(self):
        self.assertIsNone(load_openvla(os.path.join("no", "such", "openvla.pt")))


if __name__ == "__main__":
    unittest.main()
