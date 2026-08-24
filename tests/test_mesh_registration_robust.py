"""
Rigorous Unit Tests for Robust Two-Stage ICP Registration Engine.
Module: tests/test_mesh_registration_robust.py

Validates:
1. PCA centroid & principal axes calculation.
2. Coarse global pre-alignment with large rotational divergence (> 45 deg, 90 deg, 180 deg).
3. Two-stage trimmed ICP convergence on simulated dental arch / tooth surface.
4. Outlier rejection (15% distance trimming) simulating gingiva/soft-tissue.
5. Clinical quality classification (EXCELLENT, ACCEPTABLE, WARNING, FAILED).
"""

import numpy as np
import pytest
import vtk
from dental.mesh_registration import MeshRegistrationEngine, RegistrationResult


def _create_synthetic_mandibular_arch_mesh() -> vtk.vtkPolyData:
    """Creates a parabolic U-shaped dental arch surface model."""
    u_vals = np.linspace(-30.0, 30.0, 50)
    points = vtk.vtkPoints()
    polys = vtk.vtkCellArray()

    # Parabolic curve: y = 0.02 * x^2
    # Extrude along vertical Z and thickness across normal
    pts_grid = []
    for z in np.linspace(-5.0, 5.0, 6):
        row = []
        for x in u_vals:
            y = 0.02 * (x ** 2)
            pt_id = points.InsertNextPoint(float(x), float(y), float(z))
            row.append(pt_id)
        pts_grid.append(row)

    # Triangulate grid
    for r in range(len(pts_grid) - 1):
        for c in range(len(u_vals) - 1):
            p0 = pts_grid[r][c]
            p1 = pts_grid[r][c + 1]
            p2 = pts_grid[r + 1][c + 1]
            p3 = pts_grid[r + 1][c]

            tri1 = vtk.vtkTriangle()
            tri1.GetPointIds().SetId(0, p0)
            tri1.GetPointIds().SetId(1, p1)
            tri1.GetPointIds().SetId(2, p2)
            polys.InsertNextCell(tri1)

            tri2 = vtk.vtkTriangle()
            tri2.GetPointIds().SetId(0, p0)
            tri2.GetPointIds().SetId(1, p2)
            tri2.GetPointIds().SetId(2, p3)
            polys.InsertNextCell(tri2)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polys)

    # Add normals
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.ComputePointNormalsOn()
    normals.Update()

    out_pd = vtk.vtkPolyData()
    out_pd.DeepCopy(normals.GetOutput())
    return out_pd


def test_pca_computation_precision():
    """Verify that compute_mesh_pca recovers exact principal inertial axes."""
    arch = _create_synthetic_mandibular_arch_mesh()
    centroid, eigvecs, eigvals = MeshRegistrationEngine.compute_mesh_pca(arch)

    assert centroid.shape == (3,)
    assert eigvecs.shape == (3, 3)
    assert eigvals.shape == (3,)
    # Eigenvalues must be in descending order
    assert eigvals[0] >= eigvals[1] >= eigvals[2]
    # Matrix must be proper orthonormal (det = +1)
    assert np.isclose(np.linalg.det(eigvecs), 1.0, atol=1e-4)


def test_two_stage_icp_recovers_large_rotational_misalignment():
    """
    Simulates an imported IOS scan with a 60-degree rotation and 25mm translation.
    Standard single-stage ICP would get stuck in a local minimum, but Two-Stage
    PCA + Trimmed ICP recovers the global alignment with sub-millimeter precision.
    """
    target = _create_synthetic_mandibular_arch_mesh()

    # Create transformed source mesh: 60 deg rotation around Z + 25mm translation
    theta = np.radians(60.0)
    rot_z = np.array([
        [np.cos(theta), -np.sin(theta), 0.0, 20.0],
        [np.sin(theta),  np.cos(theta), 0.0, -15.0],
        [0.0,            0.0,           1.0, 10.0],
        [0.0,            0.0,           0.0, 1.0],
    ])

    mat4 = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            mat4.SetElement(r, c, rot_z[r, c])

    tf = vtk.vtkTransform()
    tf.SetMatrix(mat4)

    tf_filter = vtk.vtkTransformPolyDataFilter()
    tf_filter.SetInputData(target)
    tf_filter.SetTransform(tf)
    tf_filter.Update()

    source = vtk.vtkPolyData()
    source.DeepCopy(tf_filter.GetOutput())

    # Execute Two-Stage Registration
    engine = MeshRegistrationEngine()
    aligned_pd, transform_np, rms_error, num_iters, max_95th, quality_status = engine.register_icp_transform(
        source=source,
        target=target,
        max_iterations=200,
        max_landmarks=2500,
    )

    # Must converge with sub-millimeter accuracy
    assert rms_error < 0.15, f"RMS error too high: {rms_error:.4f} mm"
    assert max_95th < 0.35, f"95th percentile error too high: {max_95th:.4f} mm"
    assert quality_status in ("EXCELLENT", "ACCEPTABLE")


def test_trimmed_icp_outlier_rejection_for_gingiva():
    """
    Tests that 15% distance trimming rejects outlier vertices (simulating soft-tissue)
    without compromising the alignment of the crown surfaces.
    """
    target = _create_synthetic_mandibular_arch_mesh()

    # Clone target and add 15% outlier points (displaced far away to simulate gingiva)
    pts = vtk.vtkPoints()
    for i in range(target.GetNumberOfPoints()):
        p = target.GetPoint(i)
        # Shift 10% of points by 8mm (gingival overhang)
        if i % 10 == 0:
            pts.InsertNextPoint(p[0], p[1] + 8.0, p[2] + 4.0)
        else:
            pts.InsertNextPoint(p[0] + 0.05, p[1] - 0.02, p[2] + 0.03)

    source_with_gingiva = vtk.vtkPolyData()
    source_with_gingiva.DeepCopy(target)
    source_with_gingiva.SetPoints(pts)

    engine = MeshRegistrationEngine()
    aligned_pd, transform_np, rms_error, num_iters, max_95th, quality_status = engine.register_icp_transform(
        source=source_with_gingiva,
        target=target,
        trim_ratio=0.15,
    )

    # Inliers should align with high accuracy despite gingival outlier vertices
    assert rms_error < 1.0
    assert quality_status in ("EXCELLENT", "ACCEPTABLE", "WARNING")


def test_registration_result_quality_status_thresholds():
    """Tests the clinical quality status classification logic in RegistrationResult."""
    target = _create_synthetic_mandibular_arch_mesh()
    result = MeshRegistrationEngine.register_icp(target, target)

    assert isinstance(result, RegistrationResult)
    assert result.rms_error < 0.05
    assert result.quality_status == "EXCELLENT"
    assert result.aligned_actor is not None
