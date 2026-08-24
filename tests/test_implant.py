"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Unit Tests: test_implant.py
"""

import unittest
import numpy as np
import vtk
from dental.nerve_tracer import NerveTracer, NerveChannel
from dental.implant_simulator import (
    DentalImplant,
    ImplantManager,
    ImplantSafetyState,
    STANDARD_IMPLANT_PRESETS
)


class TestImplantSimulator(unittest.TestCase):
    """Tests for Parametric Dental Implant Geometry, Kinematics, and Nerve Safety Clearances."""

    def setUp(self):
        self.tracer = NerveTracer()
        # Add left mandibular canal points along Z = -15 to -20
        self.tracer.add_point(-15.0, -10.0, -18.0, channel=NerveChannel.LEFT)
        self.tracer.add_point(-15.0, 0.0, -18.0, channel=NerveChannel.LEFT)
        self.tracer.add_point(-15.0, 10.0, -18.0, channel=NerveChannel.LEFT)

    def test_parametric_implant_geometry(self):
        implant = DentalImplant(
            implant_id="test_imp_1",
            tooth_number=19,
            diameter_mm=4.0,
            length_mm=11.5,
            position=(-15.0, 0.0, 0.0)
        )
        self.assertEqual(implant.diameter_mm, 4.0)
        self.assertEqual(implant.length_mm, 11.5)
        self.assertIsNotNone(implant.implant_polydata)
        self.assertGreater(implant.implant_polydata.GetNumberOfPoints(), 20)
        self.assertIsNotNone(implant.sleeve_polydata)
        self.assertGreater(implant.sleeve_polydata.GetNumberOfPoints(), 20)

        # Test apex and platform world coordinates
        top = implant.get_platform_center_world()
        self.assertEqual(top, (-15.0, 0.0, 0.0))

        apex = implant.get_apical_tip_world()
        self.assertAlmostEqual(apex[0], -15.0, places=2)
        self.assertAlmostEqual(apex[1], 0.0, places=2)
        self.assertAlmostEqual(apex[2], -11.5, places=2)

    def test_implant_angulation_transform(self):
        implant = DentalImplant(
            implant_id="test_imp_2",
            diameter_mm=4.0,
            length_mm=10.0,
            position=(0.0, 0.0, 0.0),
            bl_angle_deg=30.0
        )
        apex = implant.get_apical_tip_world()
        # With 30 deg rotation around X, apex Y should shift: -10 * sin(-30) = +5.0 or -5.0
        self.assertNotEqual(apex[1], 0.0)
        self.assertGreater(abs(apex[1]), 3.0)

    def test_surface_sampling(self):
        implant = DentalImplant(
            implant_id="test_imp_3",
            diameter_mm=4.0,
            length_mm=10.0,
            position=(10.0, 10.0, 5.0)
        )
        samples = implant.get_surface_sample_points(num_rings=6, pts_per_ring=8)
        self.assertIsInstance(samples, np.ndarray)
        self.assertGreater(len(samples), 30)

    def test_nerve_safety_clearance_states(self):
        mgr = ImplantManager()

        # 1. Safe Placement (Apex at Z = -10, Nerve at Z = -18 -> Clearance ~ 7.0 mm)
        imp_safe = mgr.add_implant(
            tooth_number=19,
            diameter_mm=4.0,
            length_mm=10.0,
            position=(-15.0, 0.0, 0.0) # apex at Z = -10.0
        )
        mgr.evaluate_nerve_clearance(self.tracer)
        self.assertEqual(imp_safe.safety_state, ImplantSafetyState.SAFE)
        self.assertGreaterEqual(imp_safe.min_nerve_dist_mm, 2.0)

        # 2. Warning Placement (Apex at Z = -16.2, Nerve at Z = -18 -> Clearance ~ 1.8 mm - 1.0 = 0.8 mm or adjusted)
        imp_safe.position = [-15.0, 0.0, -5.2] # apex at Z = -15.2 -> dist to -18 is 2.8 - 1.0 = 1.8 mm
        imp_safe.update_transform()
        mgr.evaluate_nerve_clearance(self.tracer)
        self.assertEqual(imp_safe.safety_state, ImplantSafetyState.WARNING)

        # 3. Critical Collision Breach (Apex at Z = -17.5, Nerve at Z = -18 -> Clearance < 1.0 mm)
        imp_safe.position = [-15.0, 0.0, -7.0] # apex at Z = -17.0 -> dist to -18 is 1.0 - 1.0 = 0.0 mm
        imp_safe.update_transform()
        mgr.evaluate_nerve_clearance(self.tracer)
        self.assertEqual(imp_safe.safety_state, ImplantSafetyState.BREACH)

    def test_implant_manager_registry(self):
        mgr = ImplantManager()
        imp1 = mgr.add_implant(tooth_number=19, diameter_mm=4.0, length_mm=10.0)
        self.assertIn(imp1.implant_id, mgr.implants)
        self.assertEqual(mgr.active_implant_id, imp1.implant_id)

        imp2 = mgr.add_implant(tooth_number=30, diameter_mm=4.5, length_mm=11.5)
        self.assertEqual(len(mgr.implants), 2)
        self.assertEqual(mgr.active_implant_id, imp2.implant_id)

        mgr.remove_implant(imp2.implant_id)
        self.assertEqual(len(mgr.implants), 1)
        self.assertEqual(mgr.active_implant_id, imp1.implant_id)


if __name__ == "__main__":
    unittest.main()
