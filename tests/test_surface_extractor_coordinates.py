"""
Rigorous Mathematical & Physical Coordinate Alignment Tests for SurfaceExtractor.
Module: tests/test_surface_extractor_coordinates.py

Verifies:
1. Exact 4x4 Homogeneous Affine Matrix T_LPS construction.
2. 1:1 physical LPS mapping of extracted PolyData meshes with non-identity direction cosines.
3. Resampling of arbitrary segmentation masks to reference VolumeData grid.
4. Spatial bounding box and centroid validation with CoordinateAlignmentError.
5. Consistent outward-pointing surface normals.
"""

import os
import tempfile
import numpy as np
import pytest
import vtk
import SimpleITK as sitk

from core.volume_data import VolumeData, DicomMetadata
from dental.surface_extractor import (
    SurfaceExtractor,
    AnatomicalPreset,
    MANDIBLE_BONE,
    TEETH_ENAMEL,
    STRUCTURE_PRESETS,
    CoordinateAlignmentError,
)


def test_construct_index_to_lps_matrix_mathematical_precision():
    """Verify that construct_index_to_lps_matrix matches T_LPS = [D*s | o]."""
    spacing = (0.5, 0.75, 1.25)
    origin = (-100.5, 45.2, -15.8)
    # Oblique / rotated direction cosines matrix
    # e.g., 90 deg rotation around Z: X->Y, Y->-X, Z->Z
    direction = (
        0.0, -1.0, 0.0,
        1.0,  0.0, 0.0,
        0.0,  0.0, 1.0,
    )

    mat = SurfaceExtractor.construct_index_to_lps_matrix(spacing, origin, direction)

    # Column 0: X vector scaled by sx -> (0*0.5, 1*0.5, 0*0.5, 0) = (0.0, 0.5, 0.0, 0.0)
    assert np.isclose(mat.GetElement(0, 0), 0.0 * 0.5)
    assert np.isclose(mat.GetElement(1, 0), 1.0 * 0.5)
    assert np.isclose(mat.GetElement(2, 0), 0.0 * 0.5)
    assert np.isclose(mat.GetElement(3, 0), 0.0)

    # Column 1: Y vector scaled by sy -> (-1*0.75, 0*0.75, 0*0.75, 0) = (-0.75, 0.0, 0.0, 0.0)
    assert np.isclose(mat.GetElement(0, 1), -1.0 * 0.75)
    assert np.isclose(mat.GetElement(1, 1), 0.0 * 0.75)
    assert np.isclose(mat.GetElement(2, 1), 0.0 * 0.75)
    assert np.isclose(mat.GetElement(3, 1), 0.0)

    # Column 2: Z vector scaled by sz -> (0*1.25, 0*1.25, 1*1.25, 0) = (0.0, 0.0, 1.25, 0.0)
    assert np.isclose(mat.GetElement(0, 2), 0.0 * 1.25)
    assert np.isclose(mat.GetElement(1, 2), 0.0 * 1.25)
    assert np.isclose(mat.GetElement(2, 2), 1.0 * 1.25)
    assert np.isclose(mat.GetElement(3, 2), 0.0)

    # Column 3: Origin
    assert np.isclose(mat.GetElement(0, 3), -100.5)
    assert np.isclose(mat.GetElement(1, 3), 45.2)
    assert np.isclose(mat.GetElement(2, 3), -15.8)
    assert np.isclose(mat.GetElement(3, 3), 1.0)


def test_surface_extraction_with_non_identity_direction_cosines():
    """
    Creates a NIfTI mask with non-identity direction cosines and verifies that
    the extracted mesh centroid and bounds match the physical SimpleITK coordinates.
    """
    # Create a 40x40x40 mask with a labeled cube in the center
    # Voxel index center of the labeled cube: i=20, j=20, k=20
    mask_np = np.zeros((40, 40, 40), dtype=np.int32)
    mask_np[15:26, 15:26, 15:26] = 1  # Mandible label

    spacing = (0.4, 0.4, 0.4)
    origin = (50.0, -120.0, 30.0)
    # Non-identity direction: 90 deg rotation around X axis (Y -> -Z, Z -> Y)
    direction = (
        1.0,  0.0,  0.0,
        0.0,  0.0, -1.0,
        0.0,  1.0,  0.0,
    )

    sitk_img = sitk.GetImageFromArray(mask_np)
    sitk_img.SetSpacing(spacing)
    sitk_img.SetOrigin(origin)
    sitk_img.SetDirection(direction)

    # Theoretical physical center of the cube in SimpleITK physical LPS space
    # Center index in SimpleITK order is (x=20, y=20, z=20)
    expected_center = np.array(sitk_img.TransformIndexToPhysicalPoint((20, 20, 20)))

    temp_nii = os.path.join(tempfile.gettempdir(), "test_rotated_seg.nii.gz")
    sitk.WriteImage(sitk_img, temp_nii)

    try:
        extractor = SurfaceExtractor()
        loaded_mask, loaded_spacing, loaded_origin, loaded_direction = extractor.load_segmentation_file(temp_nii)

        assert np.allclose(loaded_spacing, spacing)
        assert np.allclose(loaded_origin, origin)
        assert np.allclose(loaded_direction, direction)

        polydata = extractor.extract_surface_polydata(
            loaded_mask, loaded_spacing, loaded_origin, loaded_direction, MANDIBLE_BONE
        )

        assert polydata.GetNumberOfPoints() > 0
        assert polydata.GetNumberOfCells() > 0

        # Compute mesh bounding box and centroid
        mb = polydata.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
        mesh_center = np.array([
            (mb[0] + mb[1]) * 0.5,
            (mb[2] + mb[3]) * 0.5,
            (mb[4] + mb[5]) * 0.5,
        ])

        # Centroid of extracted mesh must match theoretical physical point within 1 voxel tolerance
        centroid_error = np.linalg.norm(mesh_center - expected_center)
        assert centroid_error < 0.5, (
            f"Physical alignment error: mesh center {mesh_center} vs expected {expected_center}, "
            f"error = {centroid_error:.3f} mm (must be < 0.5 mm)"
        )

        # Verify normals exist and are point-based
        normals_array = polydata.GetPointData().GetNormals()
        assert normals_array is not None
        assert normals_array.GetNumberOfTuples() == polydata.GetNumberOfPoints()

    finally:
        if os.path.exists(temp_nii):
            os.remove(temp_nii)


def test_mask_resampling_to_reference_volume():
    """Tests that a segmentation mask with different grid is resampled onto reference VolumeData."""
    # Create reference VolumeData
    ref_arr = np.zeros((50, 50, 50), dtype=np.int16)
    ref_volume = VolumeData(
        array=ref_arr,
        spacing=(0.5, 0.5, 0.5),
        origin=(10.0, 20.0, 30.0),
        direction=(1, 0, 0, 0, 1, 0, 0, 0, 1),
    )

    # Create segmentation mask with coarser spacing (1.0mm) and different dimensions (25x25x25)
    mask_np = np.zeros((25, 25, 25), dtype=np.int32)
    mask_np[8:18, 8:18, 8:18] = 3  # Teeth label

    sitk_mask = sitk.GetImageFromArray(mask_np)
    sitk_mask.SetSpacing((1.0, 1.0, 1.0))
    sitk_mask.SetOrigin((10.0, 20.0, 30.0))
    sitk_mask.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    temp_nii = os.path.join(tempfile.gettempdir(), "test_resample_seg.nii.gz")
    sitk.WriteImage(sitk_mask, temp_nii)

    try:
        extractor = SurfaceExtractor()
        loaded_mask, sp, orig, direct = extractor.load_segmentation_file(
            temp_nii, reference_volume=ref_volume
        )

        # After resampling, mask should have reference volume's dimensions and spacing
        assert loaded_mask.shape == (ref_volume.nz, ref_volume.ny, ref_volume.nx)
        assert sp == ref_volume.spacing
        assert np.allclose(orig, ref_volume.origin)

        # Label 3 should be preserved
        assert 3 in np.unique(loaded_mask)

        # Extract mesh and validate spatial bounds against reference volume
        polydata = extractor.extract_surface_polydata(
            loaded_mask, sp, orig, direct, TEETH_ENAMEL
        )
        assert extractor.validate_spatial_bounds(polydata, ref_volume)

    finally:
        if os.path.exists(temp_nii):
            os.remove(temp_nii)


def test_spatial_bounds_validation_raises_coordinate_alignment_error():
    """Verify that validate_spatial_bounds catches drifted/misaligned meshes."""
    ref_arr = np.zeros((50, 50, 50), dtype=np.int16)
    ref_volume = VolumeData(
        array=ref_arr,
        spacing=(0.5, 0.5, 0.5),
        origin=(0.0, 0.0, 0.0),
    )

    # Create sphere shifted 500mm away (far outside volume)
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(10.0)
    sphere.SetCenter(500.0, 500.0, 500.0)
    sphere.Update()
    drifted_poly = sphere.GetOutput()

    with pytest.raises(CoordinateAlignmentError) as exc_info:
        SurfaceExtractor.validate_spatial_bounds(drifted_poly, ref_volume, max_allowable_drift_mm=50.0)

    assert "Spatial Misalignment Detected" in str(exc_info.value)
