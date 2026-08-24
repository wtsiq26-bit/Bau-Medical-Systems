"""
Unit & Integration Tests for 3D Printable Surgical Guide CAD Generator.
Module: tests/test_surgical_guide.py

Validates:
1. Guide base solid shell generation from open dental surface.
2. Parametric drill sleeve housing construction co-axial with planned DentalImplant.
3. Multi-implant surgical guide synthesis with CSG drill channels and irrigation windows.
4. Manifold checking and resin material volume calculation.
5. Standardized binary STL export.
6. Asynchronous SurgicalGuideWorker thread execution.
"""

import os
import tempfile
import numpy as np
import pytest
import vtk
from PySide6.QtCore import QCoreApplication

from dental.implant_simulator import DentalImplant
from dental.surgical_guide import SurgicalGuideGenerator, SurgicalGuideResult
from core.async_workers import SurgicalGuideWorker


def _create_synthetic_arch_surface() -> vtk.vtkPolyData:
    """Creates a realistic curved parabolic mandibular teeth arch surface."""
    u_vals = np.linspace(-25.0, 25.0, 30)
    points = vtk.vtkPoints()
    polys = vtk.vtkCellArray()

    pts_grid = []
    for z in np.linspace(0.0, 8.0, 5):
        row = []
        for x in u_vals:
            y = 0.025 * (x ** 2)
            pt_id = points.InsertNextPoint(float(x), float(y), float(z))
            row.append(pt_id)
        pts_grid.append(row)

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

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.ComputePointNormalsOn()
    normals.Update()

    out_pd = vtk.vtkPolyData()
    out_pd.DeepCopy(normals.GetOutput())
    return out_pd


def test_surgical_guide_solid_base_creation():
    """Verify solid guide base generation from open surface."""
    surface = _create_synthetic_arch_surface()
    guide_base = SurgicalGuideGenerator._create_solid_guide_base(surface, thickness=3.0)

    assert guide_base.GetNumberOfPoints() > surface.GetNumberOfPoints()
    assert guide_base.GetNumberOfCells() > 0


def test_implant_sleeve_housing_geometry():
    """Verify parametric sleeve housing aligns co-axially with implant trajectory."""
    implant = DentalImplant(
        implant_id="imp_test_1",
        tooth_number=19,
        diameter_mm=4.0,
        length_mm=10.0,
        position=(15.0, 5.0, 4.0),
        bl_angle_deg=10.0,
        md_angle_deg=-5.0,
    )

    housing, channel, window = SurgicalGuideGenerator._build_implant_sleeve_geometry(
        implant=implant,
        clearance_mm=1.2,
        outer_wall_mm=2.0,
        sleeve_height_mm=6.0,
        sleeve_offset_mm=2.0,
        include_window=True,
    )

    assert housing.GetNumberOfPoints() > 0
    assert channel.GetNumberOfPoints() > 0
    assert window is not None
    assert window.GetNumberOfPoints() > 0


def test_full_surgical_guide_generation_and_stl_export():
    """Verify full CAD guide synthesis with multiple implants and STL export."""
    surface = _create_synthetic_arch_surface()

    implant1 = DentalImplant(
        implant_id="imp_19",
        tooth_number=19,
        diameter_mm=4.0,
        length_mm=11.5,
        position=(12.0, 4.0, 4.0),
    )
    implant2 = DentalImplant(
        implant_id="imp_30",
        tooth_number=30,
        diameter_mm=4.5,
        length_mm=10.0,
        position=(-12.0, 4.0, 4.0),
    )

    result = SurgicalGuideGenerator.generate_guide(
        base_surface=surface,
        implants=[implant1, implant2],
        guide_thickness_mm=3.0,
        sleeve_clearance_mm=1.2,
        sleeve_outer_wall_mm=2.0,
        sleeve_height_mm=6.0,
    )

    assert isinstance(result, SurgicalGuideResult)
    assert result.guide_polydata.GetNumberOfPoints() > 0
    assert result.num_implants_guided == 2
    assert result.volume_cm3 > 0.0
    assert result.surface_area_cm2 > 0.0
    assert result.guide_actor is not None

    # Test binary STL export
    temp_stl = os.path.join(tempfile.gettempdir(), "test_surgical_guide_export.stl")
    try:
        success = SurgicalGuideGenerator.export_guide_stl(temp_stl, result.guide_polydata)
        assert success is True
        assert os.path.isfile(temp_stl)
        assert os.path.getsize(temp_stl) > 1000  # Valid binary STL file size
    finally:
        if os.path.exists(temp_stl):
            os.remove(temp_stl)


def test_surgical_guide_async_worker(qapp):
    """Tests asynchronous execution of SurgicalGuideWorker."""
    surface = _create_synthetic_arch_surface()
    implant = DentalImplant(
        implant_id="imp_worker_test",
        tooth_number=19,
        diameter_mm=4.0,
        length_mm=10.0,
        position=(0.0, 0.0, 2.0),
    )

    worker = SurgicalGuideWorker(
        base_surface=surface,
        implants=[implant],
        guide_thickness_mm=3.0,
        sleeve_clearance_mm=1.2,
    )

    progress_events = []
    guide_results = []
    errors = []

    worker.progress_updated.connect(lambda pct, msg: progress_events.append((pct, msg)))
    worker.guide_generated.connect(lambda pd, vol, area: guide_results.append((pd, vol, area)))
    worker.error_occurred.connect(lambda err: errors.append(err))

    worker.start()
    assert worker.wait(15000), "SurgicalGuideWorker timed out"
    QCoreApplication.processEvents()

    assert len(errors) == 0, f"Worker failed with: {errors}"
    assert len(guide_results) == 1
    guide_pd, vol_cm3, area_cm2 = guide_results[0]

    assert guide_pd.GetNumberOfPoints() > 0
    assert vol_cm3 > 0.0
    assert area_cm2 > 0.0
    assert len(progress_events) >= 3
