"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Unit Tests: tests/test_core.py
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
from PIL import Image
import vtk
from core.volume_data import VolumeData, DicomMetadata
from core.dicom_loader import DicomLoaderWorker, natural_sort_key
from core.presets import WL_PRESETS, VOLUME_3D_PRESETS


class TestVolumeData(unittest.TestCase):
    """Tests for VolumeData memory management, conversions, and coordinate transforms."""

    def setUp(self):
        self.nz, self.ny, self.nx = 20, 30, 40
        self.spacing = (0.5, 0.5, 1.0)
        self.origin = (10.0, 20.0, 30.0)

        arr = np.arange(self.nz * self.ny * self.nx, dtype=np.int16).reshape((self.nz, self.ny, self.nx))
        self.arr = arr
        self.volume = VolumeData(
            array=arr,
            spacing=self.spacing,
            origin=self.origin,
            metadata=DicomMetadata(patient_name="Test Patient", patient_id="12345")
        )

    def test_dimensions_and_spacing(self):
        self.assertEqual(self.volume.dimensions, (self.nx, self.ny, self.nz))
        self.assertEqual(self.volume.spacing, self.spacing)
        self.assertEqual(self.volume.origin, self.origin)

    def test_zero_copy_buffer_reference(self):
        vtk_image = self.volume.vtk_image_data
        self.assertIsNotNone(vtk_image)
        self.assertEqual(vtk_image.GetDimensions(), (self.nx, self.ny, self.nz))

        scalars = vtk_image.GetPointData().GetScalars()
        self.assertIsNotNone(scalars)
        self.assertEqual(scalars.GetNumberOfTuples(), self.nx * self.ny * self.nz)

    def test_coordinate_transforms(self):
        i, j, k = 10, 15, 8
        world_x, world_y, world_z = self.volume.index_to_world(i, j, k)
        i_res, j_res, k_res = self.volume.world_to_index(world_x, world_y, world_z)

        self.assertEqual(i, i_res)
        self.assertEqual(j, j_res)
        self.assertEqual(k, k_res)

    def test_hu_sampling(self):
        i, j, k = 5, 12, 18
        expected_val = float(self.arr[k, j, i])
        sampled_val = self.volume.get_hu_at_voxel(i, j, k)
        self.assertEqual(expected_val, sampled_val)

    def test_reslice_matrices(self):
        center = self.volume.get_center()
        for plane in ("axial", "coronal", "sagittal"):
            m = self.volume.get_reslice_matrix_for_plane(plane, center)
            self.assertIsInstance(m, vtk.vtkMatrix4x4)
            self.assertAlmostEqual(m.GetElement(0, 3), center[0])
            self.assertAlmostEqual(m.GetElement(1, 3), center[1])
            self.assertAlmostEqual(m.GetElement(2, 3), center[2])


class TestLoaders(unittest.TestCase):
    """Tests for synthetic CBCT generator and 2D image sequence loader."""

    def test_synthetic_cbct_generation(self):
        worker = DicomLoaderWorker(is_synthetic=True)
        volume = worker._generate_synthetic_dental_cbct()

        self.assertIsNotNone(volume)
        self.assertEqual(volume.dimensions, (160, 160, 140))
        self.assertEqual(volume.spacing, (0.5, 0.5, 0.5))
        self.assertEqual(volume.metadata.patient_name, "DEMO^DENTAL_CBCT")
        self.assertGreater(volume.max_hu, 2000)
        self.assertLess(volume.min_hu, -900)

    def test_png_image_sequence_loading(self):
        # Create a temporary folder with 10 PNG slices
        temp_dir = tempfile.mkdtemp(prefix="test_png_sequence_")
        try:
            for i in range(10):
                img_arr = np.full((64, 64), fill_value=i * 25, dtype=np.uint8)
                img = Image.fromarray(img_arr)
                img.save(os.path.join(temp_dir, f"slice_{i:03d}.png"))

            worker = DicomLoaderWorker(directory_path=temp_dir)
            volume = worker._load_from_image_sequence(temp_dir)

            self.assertIsNotNone(volume)
            self.assertEqual(volume.dimensions, (64, 64, 10))
            self.assertEqual(volume.spacing, (0.4, 0.4, 0.4))
            self.assertTrue(volume.metadata.patient_name.startswith("DATASET_"))
            self.assertLess(volume.min_hu, -900) # 0 maps to -1000 HU
            self.assertGreater(volume.max_hu, 1500)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_natural_sort_key(self):
        names = ["slice_1.png", "slice_10.png", "slice_2.png", "slice_20.png"]
        sorted_names = sorted(names, key=natural_sort_key)
        self.assertEqual(sorted_names, ["slice_1.png", "slice_2.png", "slice_10.png", "slice_20.png"])


if __name__ == "__main__":
    unittest.main()
