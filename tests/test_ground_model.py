"""Calibrated ground and the landmark footprint prior in the runtime projector (#112)."""

from __future__ import annotations

import dataclasses
import math
import os
import unittest
from unittest import mock

import numpy as np

from marc_agent.perception.cctv_project import (
    LANDMARK_FOOTPRINT,
    CctvProjector,
    static_lookup,
)
from marc_agent.perception.ground_model import GroundModel, GroundSurface
from scripts.stage1_replay import load_geometry

GEOMETRY = load_geometry()
MODEL = GroundModel.load()
SIZE = (1280, 720)
BOX = (600.0, 400.0, 680.0, 470.0)  # mid-frame, fully visible
ALL_ON = {"MARC_GROUND_CALIBRATION": "1", "MARC_LANDMARK_FOOTPRINT": "1"}
ALL_OFF = {"MARC_GROUND_CALIBRATION": "0", "MARC_LANDMARK_FOOTPRINT": "0"}


def _projector(geometry=GEOMETRY, ground=MODEL) -> CctvProjector:
    return CctvProjector(static_lookup(geometry), ground)


class GroundModelTests(unittest.TestCase):
    def test_every_camera_is_calibrated_for_its_live_pose(self):
        for camera, geometry in GEOMETRY.items():
            surface, status = MODEL.for_camera(camera, geometry)
            self.assertEqual("applied", status, camera)
            self.assertGreater(len(surface.points), 1000)

    def test_a_moved_camera_falls_back_to_the_flat_plane(self):
        live = GEOMETRY["rig_3_c"]
        moved = dataclasses.replace(
            live, translation=(live.translation[0] + 0.05, *live.translation[1:])
        )
        turned = dataclasses.replace(live, rotation=tuple(
            tuple(float(v) for v in row)
            for row in np.asarray(live.rotation) @ np.array(
                [[1, 0, 0], [0, math.cos(0.01), -math.sin(0.01)], [0, math.sin(0.01), math.cos(0.01)]]
            )
        ))
        lowered = dataclasses.replace(live, ground_height=live.ground_height - 0.1)
        for geometry, reason in ((moved, "translation"), (turned, "rotation"),
                                 (lowered, "ground_height")):
            surface, status = MODEL.for_camera("rig_3_c", geometry)
            self.assertIsNone(surface)
            self.assertEqual(f"fingerprint_mismatch:{reason}", status)
            projector = _projector({"rig_3_c": geometry})
            flat = CctvProjector(static_lookup({"rig_3_c": geometry}))
            with mock.patch.dict(os.environ, ALL_ON):
                self.assertEqual(flat.project_contact("rig_3_c", BOX),
                                 projector.project_contact("rig_3_c", BOX))

    def test_calibrated_points_lie_on_the_surface(self):
        surface, _ = MODEL.for_camera("rig_3_c", GEOMETRY["rig_3_c"])
        with mock.patch.dict(os.environ, ALL_ON):
            x, y, z = _projector().project_contact("rig_3_c", BOX)
        self.assertAlmostEqual(surface.z(x, y), z, delta=2e-3)
        # rig_3_c's ground lies well below its published plane (#112).
        self.assertLess(z, GEOMETRY["rig_3_c"].ground_height - 0.2)

    def test_flags_off_reproduce_the_flat_plane_exactly(self):
        flat = CctvProjector(static_lookup(GEOMETRY))
        with mock.patch.dict(os.environ, ALL_OFF):
            for camera in GEOMETRY:
                self.assertEqual(
                    flat.project_contact(camera, BOX), _projector().project_contact(camera, BOX)
                )
                self.assertEqual(
                    flat.project_contact(camera, BOX),
                    _projector().project_landmark(camera, BOX, "taxi", SIZE),
                )

    def test_people_and_raised_placements_keep_the_flat_plane(self):
        flat = CctvProjector(static_lookup(GEOMETRY))
        with mock.patch.dict(os.environ, ALL_ON):
            self.assertEqual(
                flat.project_contact("rig_3_c", BOX, pose_class="person_prone"),
                _projector().project_contact("rig_3_c", BOX, pose_class="person_prone"),
            )
            self.assertEqual(
                flat.project_contact("rig_3_c", BOX, relation="on", support_class="bench"),
                _projector().project_contact("rig_3_c", BOX, relation="on", support_class="bench"),
            )

    def test_landmark_prior_uses_the_class_fraction_and_height(self):
        fu, fv, dz = LANDMARK_FOOTPRINT["taxi"]
        projector = _projector()
        with mock.patch.dict(os.environ, ALL_ON):
            got = projector.project_landmark("rig_3_c", BOX, "taxi", SIZE)
            ground = projector.project_pixel(
                "rig_3_c", BOX[0] + fu * (BOX[2] - BOX[0]), BOX[1] + fv * (BOX[3] - BOX[1])
            )
        self.assertEqual((ground[0], ground[1]), (got[0], got[1]))
        self.assertAlmostEqual(ground[2] + dz, got[2])

    def test_clipped_or_unknown_landmarks_keep_the_contact_rule(self):
        projector = _projector()
        clipped = (0.0, 400.0, 80.0, 470.0)
        with mock.patch.dict(os.environ, ALL_ON):
            self.assertEqual(projector.project_contact("rig_3_c", clipped),
                             projector.project_landmark("rig_3_c", clipped, "taxi", SIZE))
            self.assertEqual(projector.project_contact("rig_3_c", BOX),
                             projector.project_landmark("rig_3_c", BOX, "spaceship", SIZE))
            self.assertEqual(projector.project_contact("rig_3_c", BOX),
                             projector.project_landmark("rig_3_c", BOX, "taxi", None))

    def test_ground_status_reports_why(self):
        with mock.patch.dict(os.environ, ALL_ON):
            self.assertEqual("applied", _projector().ground_status("rig_1_a"))
            self.assertEqual("no_model", _projector(ground=None).ground_status("rig_1_a"))
        with mock.patch.dict(os.environ, ALL_OFF):
            self.assertEqual("disabled", _projector().ground_status("rig_1_a"))

    def test_surface_drops_points_on_furniture(self):
        ground = [(x, y, 0.0) for x in range(-2, 3) for y in range(-2, 3)]
        table = [(0.1, 0.1, 0.75), (-0.1, 0.1, 0.74), (0.1, -0.1, 0.76)]
        surface = GroundSurface(np.asarray(ground + table, dtype=float))
        self.assertAlmostEqual(0.0, surface.z(0.0, 0.0), places=6)


if __name__ == "__main__":
    unittest.main()
