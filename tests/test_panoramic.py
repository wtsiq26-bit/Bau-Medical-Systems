"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Unit Tests: test_panoramic.py
"""

import unittest
import numpy as np
import vtk
from core.volume_data import VolumeData, DicomMetadata
from dental.panoramic_mpr import DentalArchCurve, PanoramicGenerator, CrossSectionManager


class TestPanoramicMPR(unittest.TestCase):
    """Tests for Curved MPR Dental Arch, Panoramic Synthesis, and Cross-Sections."""

    def setUp(self):
        # Create synthetic volume 100x100x80 with dental-like cylinder
        arr = np.full((80, 100, 100), -1000, dtype=np.int16)
        # Add high-density bone horseshoe/cylinder
        yy, xx = np.ogrid[:100, :100]
        mask = (xx - 50) ** 2 + (yy - 40) ** 2 <= 25 ** 2
        mask &= (xx - 50) ** 2 + (yy - 40) ** 2 >= 15 ** 2
        arr[20:60, mask] = 1200  # Cortical bone

        meta = DicomMetadata(
            patient_name="TEST^PANORAMIC",
            patient_id="PANO001",
            series_description="Synthetic CBCT Arch",
            window_center=500.0,
            window_width=2500.0
        )
        self.volume = VolumeData(
            array=arr,
            spacing=(0.5, 0.5, 0.5),
            origin=(-25.0, -25.0, -20.0),
            metadata=meta
        )

    def test_dental_arch_curve_spline(self):
        arch = DentalArchCurve(step_size_mm=1.0)
        # Add 5 points
        arch.add_seed_point(-15.0, 5.0, 0.0)
        arch.add_seed_point(-10.0, -8.0, 0.0)
        arch.add_seed_point(0.0, -14.0, 0.0)
        arch.add_seed_point(10.0, -8.0, 0.0)
        arch.add_seed_point(15.0, 5.0, 0.0)

        self.assertGreater(arch.total_length_mm, 35.0)
        self.assertGreaterEqual(len(arch.sampled_points), 35)
        self.assertEqual(len(arch.tangents), len(arch.sampled_points))
        self.assertEqual(len(arch.normals), len(arch.sampled_points))

        # Check orthonormal properties: |T| = 1, |N| = 1, T . N = 0
        for i in range(len(arch.sampled_points)):
            t = arch.tangents[i]
            n = arch.normals[i]
            self.assertAlmostEqual(np.linalg.norm(t), 1.0, places=3)
            self.assertAlmostEqual(np.linalg.norm(n), 1.0, places=3)
            self.assertAlmostEqual(np.dot(t, n), 0.0, places=3)

    def test_auto_fit_parabola(self):
        arch = DentalArchCurve(step_size_mm=1.0)
        arch.auto_fit_parabola(self.volume, z_world=0.0)

        self.assertGreater(len(arch.seed_points), 5)
        self.assertGreater(arch.total_length_mm, 20.0)
        self.assertGreater(len(arch.sampled_points), 20)

    def test_panoramic_image_synthesis(self):
        arch = DentalArchCurve(step_size_mm=1.0)
        arch.auto_fit_parabola(self.volume, z_world=0.0)

        generator = PanoramicGenerator()
        generator.focal_trough_thickness_mm = 6.0
        generator.projection_mode = "mip"

        pano_arr = generator.generate_panoramic_image(self.volume, arch)
        self.assertIsNotNone(pano_arr)
        self.assertEqual(len(pano_arr.shape), 2)
        self.assertGreater(pano_arr.shape[0], 20) # Height Z
        self.assertEqual(pano_arr.shape[1], len(arch.sampled_points)) # Columns S

        # Verify VTK Image conversion
        vtk_img = generator.generate_panoramic_vtk_image(self.volume, arch)
        self.assertIsNotNone(vtk_img)
        dims = vtk_img.GetDimensions()
        self.assertEqual(dims[0], pano_arr.shape[1])
        self.assertEqual(dims[1], pano_arr.shape[0])
        self.assertEqual(dims[2], 1)

    def test_cross_section_manager(self):
        arch = DentalArchCurve(step_size_mm=1.0)
        arch.auto_fit_parabola(self.volume, z_world=0.0)

        mgr = CrossSectionManager(arch)
        self.assertGreater(mgr.total_cross_sections, 10)

        mgr.set_active_index(5)
        self.assertEqual(mgr.active_index, 5)

        matrix = mgr.get_reslice_matrix_for_index(5)
        self.assertIsNotNone(matrix)
        self.assertIsInstance(matrix, vtk.vtkMatrix4x4)

        # Check that origin in column 3 matches curve point
        pt = arch.sampled_points[5]
        self.assertAlmostEqual(matrix.GetElement(0, 3), pt[0], places=3)
        self.assertAlmostEqual(matrix.GetElement(1, 3), pt[1], places=3)
        self.assertAlmostEqual(matrix.GetElement(2, 3), pt[2], places=3)


if __name__ == "__main__":
    unittest.main()
