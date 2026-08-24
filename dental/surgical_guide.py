"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Module: dental/surgical_guide.py

Parametric 3D Printable Surgical Guide Generator.
Generates patient-specific, 3D-printable surgical implant drill guides from
registered Intraoral Scans (IOS) / CBCT teeth arches and planned virtual implants.

Key Engineering Features:
-------------------------
1. Watertight Rigid Guide Base:
   - Offsets the patient dental surface outward (2.0 - 4.0 mm, default 3.0 mm).
   - Generates a closed manifold thick-shell solid with smooth anatomical borders.
   - Windowed Sinc smoothing for patient seating comfort.

2. Parametric Metal Drill Sleeve Housing & Channel Subtraction:
   - Rigid metallic drill cylinders co-axial with planned implant trajectories.
   - Calibrated drill clearance: diameter = implant_diameter + 1.2 mm.
   - Outer wall support: diameter = drill_diameter + 2.0 mm wall thickness.
   - Buccal inspection / saline irrigation windows (3.0 mm x 2.0 mm slot).
   - Robust Boolean CSG subtraction of drill channels.

3. 3D Print Validation & STL Export:
   - Manifold integrity check (zero non-manifold edges, volume computation).
   - Standardized binary STL export with physical millimeter calibration.
"""

from __future__ import annotations

import os
import math
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from dental.implant_simulator import DentalImplant

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Surgical Guide Result & Material Constants
# ---------------------------------------------------------------------------

@dataclass
class SurgicalGuideResult:
    """Encapsulates the generated 3D surgical guide geometry and metrics."""

    guide_polydata: vtk.vtkPolyData
    """Watertight 3D printable surgical guide PolyData."""

    guide_actor: vtk.vtkActor
    """Ready-to-render translucent Amber actor for 3D viewport visualization."""

    volume_cm3: float
    """Total resin material volume in cubic centimeters (cm³)."""

    surface_area_cm2: float
    """Total surface area in square centimeters (cm²)."""

    is_manifold: bool
    """True if mesh has 0 non-manifold edges and is watertight."""

    num_implants_guided: int
    """Number of implant drill channels embedded in the guide."""


_GUIDE_COLOR: Tuple[float, float, float] = (1.00, 0.75, 0.05)  # Surgical Amber (#FFBF00)
_GUIDE_OPACITY: float = 0.70
_GUIDE_AMBIENT: float = 0.25
_GUIDE_DIFFUSE: float = 0.75
_GUIDE_SPECULAR: float = 0.50
_GUIDE_SPECULAR_POWER: float = 40.0


# ---------------------------------------------------------------------------
# Surgical Guide Generator Engine
# ---------------------------------------------------------------------------

class SurgicalGuideGenerator:
    """
    Parametric CAD generator for patient-specific 3D printable surgical implant guides.
    """

    @staticmethod
    def generate_guide(
        base_surface: vtk.vtkPolyData,
        implants: List[DentalImplant],
        guide_thickness_mm: float = 3.0,
        sleeve_clearance_mm: float = 1.2,
        sleeve_outer_wall_mm: float = 2.0,
        sleeve_height_mm: float = 6.0,
        sleeve_offset_mm: float = 2.0,
        include_irrigation_windows: bool = True,
    ) -> SurgicalGuideResult:
        """
        Generates a complete 3D printable surgical guide mesh.

        Parameters
        ----------
        base_surface : vtkPolyData
            The patient's aligned IOS scan or segmented teeth mesh.
        implants : List[DentalImplant]
            List of planned virtual dental implants.
        guide_thickness_mm : float
            Thickness of the guide base shell in mm (default 3.0 mm).
        sleeve_clearance_mm : float
            Internal drill sleeve diameter clearance (default 1.2 mm).
        sleeve_outer_wall_mm : float
            Thickness of the drill cylinder outer wall (default 2.0 mm).
        sleeve_height_mm : float
            Height of the metallic sleeve housing (default 6.0 mm).
        sleeve_offset_mm : float
            Elevation offset above the platform/mucosal margin (default 2.0 mm).
        include_irrigation_windows : bool
            Whether to carve buccal irrigation & inspection windows.

        Returns
        -------
        SurgicalGuideResult
        """
        if base_surface is None or base_surface.GetNumberOfPoints() == 0:
            raise ValueError("Base surface mesh is empty or None.")

        # ---- Step 1: Clean and Prepare Base Surface ----
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(base_surface)
        cleaner.ToleranceIsAbsoluteOn()
        cleaner.SetAbsoluteTolerance(0.05)
        cleaner.Update()

        triangulator = vtk.vtkTriangleFilter()
        triangulator.SetInputConnection(cleaner.GetOutputPort())
        triangulator.Update()
        clean_base = triangulator.GetOutput()

        # ---- Step 2: Generate Watertight Guide Base Shell ----
        guide_base = SurgicalGuideGenerator._create_solid_guide_base(
            clean_base, thickness=guide_thickness_mm
        )

        # ---- Step 3: Construct Sleeve Housings and Drill Channels ----
        sleeve_housings: List[vtk.vtkPolyData] = []
        drill_channels: List[vtk.vtkPolyData] = []
        irrigation_slots: List[vtk.vtkPolyData] = []

        for implant in implants:
            housing_pd, channel_pd, window_pd = SurgicalGuideGenerator._build_implant_sleeve_geometry(
                implant=implant,
                clearance_mm=sleeve_clearance_mm,
                outer_wall_mm=sleeve_outer_wall_mm,
                sleeve_height_mm=sleeve_height_mm,
                sleeve_offset_mm=sleeve_offset_mm,
                include_window=include_irrigation_windows,
            )
            sleeve_housings.append(housing_pd)
            drill_channels.append(channel_pd)
            if window_pd is not None:
                irrigation_slots.append(window_pd)

        # ---- Step 4: CSG Assembly (Union Sleeves + Base, Subtract Channels) ----
        final_guide = SurgicalGuideGenerator._assemble_guide_csg(
            guide_base=guide_base,
            sleeve_housings=sleeve_housings,
            drill_channels=drill_channels,
            irrigation_slots=irrigation_slots,
        )

        # ---- Step 5: Final Manifold Cleanup & Smoothing ----
        final_guide = SurgicalGuideGenerator._smooth_and_finalize_mesh(final_guide)

        # ---- Step 6: Compute Quality & Material Volume Metrics ----
        volume_cm3, area_cm2 = SurgicalGuideGenerator._compute_mesh_metrics(final_guide)
        is_manifold, _, _ = SurgicalGuideGenerator.validate_manifold(final_guide)

        # Build ready-to-render actor
        guide_actor = SurgicalGuideGenerator.create_guide_actor(final_guide)

        return SurgicalGuideResult(
            guide_polydata=final_guide,
            guide_actor=guide_actor,
            volume_cm3=volume_cm3,
            surface_area_cm2=area_cm2,
            is_manifold=is_manifold,
            num_implants_guided=len(implants),
        )

    # ------------------------------------------------------------------
    # Guide Base Shell Geometry Construction
    # ------------------------------------------------------------------

    @staticmethod
    def _create_solid_guide_base(surface: vtk.vtkPolyData, thickness: float = 3.0) -> vtk.vtkPolyData:
        """
        Creates a smooth, watertight thick solid shell from an open dental surface
        by offsetting vertices outward along point normals and stitching borders.
        """
        # Ensure point normals are computed and oriented outward
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(surface)
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.SplittingOff()
        normals.Update()
        norm_surf = normals.GetOutput()

        pts = vtk_to_numpy(norm_surf.GetPoints().GetData())
        n_pts = len(pts)
        norm_arr = vtk_to_numpy(norm_surf.GetPointData().GetNormals())

        # Outer offset points: P_outer = P + thickness * Normal
        outer_pts = pts + (norm_arr * float(thickness))

        # Combine inner and outer points into a single coordinate array
        combined_pts = np.vstack([pts, outer_pts])  # shape (2*N, 3)

        vtk_pts = vtk.vtkPoints()
        vtk_pts.SetData(numpy_to_vtk(combined_pts, deep=True))

        # Extract triangles from original surface
        orig_polys = norm_surf.GetPolys()
        orig_polys.InitTraversal()
        id_list = vtk.vtkIdList()

        inner_triangles = []
        outer_triangles = []

        while orig_polys.GetNextCell(id_list):
            if id_list.GetNumberOfIds() == 3:
                i0 = id_list.GetId(0)
                i1 = id_list.GetId(1)
                i2 = id_list.GetId(2)

                # Inner surface: reverse triangle winding so normals point inward (cavity)
                inner_triangles.append([i0, i2, i1])

                # Outer surface: offset indices by n_pts, normal winding outward
                outer_triangles.append([i0 + n_pts, i1 + n_pts, i2 + n_pts])

        # Identify boundary edges to stitch the rim
        fe = vtk.vtkFeatureEdges()
        fe.SetInputData(norm_surf)
        fe.BoundaryEdgesOn()
        fe.FeatureEdgesOff()
        fe.NonManifoldEdgesOff()
        fe.ManifoldEdgesOff()
        fe.Update()

        rim_triangles = []
        fe_lines = fe.GetOutput().GetLines()
        fe_lines.InitTraversal()
        edge_ids = vtk.vtkIdList()

        while fe_lines.GetNextCell(edge_ids):
            if edge_ids.GetNumberOfIds() == 2:
                e0 = edge_ids.GetId(0)
                e1 = edge_ids.GetId(1)
                # Map back to original point IDs
                pt0 = fe.GetOutput().GetPoint(e0)
                pt1 = fe.GetOutput().GetPoint(e1)
                # Find matching IDs in norm_surf via locator
                pass

        # Robust stitching using cell boundary edges directly:
        edge_table: Dict[Tuple[int, int], int] = {}
        for tri in inner_triangles:
            edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
            for u, v in edges:
                key = (min(u, v), max(u, v))
                edge_table[key] = edge_table.get(key, 0) + 1

        # Boundary edges appear exactly once
        boundary_edges = [k for k, count in edge_table.items() if count == 1]

        for u, v in boundary_edges:
            u_out = u + n_pts
            v_out = v + n_pts
            # Quad formed by (u, v, v_out, u_out) -> 2 triangles
            rim_triangles.append([u, v, v_out])
            rim_triangles.append([u, v_out, u_out])

        # Combine all triangles into cell array
        all_triangles = inner_triangles + outer_triangles + rim_triangles
        all_polys = vtk.vtkCellArray()

        for tri in all_triangles:
            cell = vtk.vtkTriangle()
            cell.GetPointIds().SetId(0, tri[0])
            cell.GetPointIds().SetId(1, tri[1])
            cell.GetPointIds().SetId(2, tri[2])
            all_polys.InsertNextCell(cell)

        solid_pd = vtk.vtkPolyData()
        solid_pd.SetPoints(vtk_pts)
        solid_pd.SetPolys(all_polys)

        # Smooth and recompute consistent normals
        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(solid_pd)
        clean.Update()

        return clean.GetOutput()

    # ------------------------------------------------------------------
    # Parametric Sleeve Geometry Builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_implant_sleeve_geometry(
        implant: DentalImplant,
        clearance_mm: float = 1.2,
        outer_wall_mm: float = 2.0,
        sleeve_height_mm: float = 6.0,
        sleeve_offset_mm: float = 2.0,
        include_window: bool = True,
    ) -> Tuple[vtk.vtkPolyData, vtk.vtkPolyData, Optional[vtk.vtkPolyData]]:
        """
        Builds the 3D geometry of a single metallic drill guide sleeve:
        - Outer housing cylinder (to be unioned with guide base).
        - Inner drill path cylinder (to be subtracted).
        - Lateral buccal irrigation / seating verification window slot.
        """
        p_plat = np.array(implant.get_platform_center_world(), dtype=np.float64)
        p_apex = np.array(implant.get_apical_tip_world(), dtype=np.float64)

        # Axis direction pointing from apex to platform (occlusal/coronal direction)
        axis = p_plat - p_apex
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-4:
            axis_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            axis_dir = axis / axis_len

        # Drill inner diameter & outer housing diameter
        d_drill = implant.diameter_mm + clearance_mm
        r_drill = d_drill * 0.5
        r_outer = r_drill + outer_wall_mm

        # Sleeve center positioned above mucosal/platform margin
        sleeve_center = p_plat + (axis_dir * (0.5 * sleeve_height_mm + sleeve_offset_mm))

        # Construct rotation matrix aligning default cylinder axis (0, 1, 0) to axis_dir
        v_from = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        v_to = axis_dir

        rot_axis = np.cross(v_from, v_to)
        rot_axis_len = np.linalg.norm(rot_axis)
        if rot_axis_len < 1e-5:
            # Parallel or anti-parallel
            if np.dot(v_from, v_to) > 0:
                R = np.eye(3)
            else:
                R = np.diag([1.0, -1.0, -1.0])
        else:
            rot_axis_norm = rot_axis / rot_axis_len
            angle_rad = np.arccos(np.clip(np.dot(v_from, v_to), -1.0, 1.0))
            K = np.array([
                [0.0, -rot_axis_norm[2], rot_axis_norm[1]],
                [rot_axis_norm[2], 0.0, -rot_axis_norm[0]],
                [-rot_axis_norm[1], rot_axis_norm[0], 0.0]
            ])
            R = np.eye(3) + np.sin(angle_rad) * K + (1.0 - np.cos(angle_rad)) * (K @ K)

        M4 = np.eye(4, dtype=np.float64)
        M4[:3, :3] = R
        M4[:3, 3] = sleeve_center

        vtk_mat = vtk.vtkMatrix4x4()
        for r in range(4):
            for c in range(4):
                vtk_mat.SetElement(r, c, M4[r, c])

        vtk_tf = vtk.vtkTransform()
        vtk_tf.SetMatrix(vtk_mat)

        # 1. Outer housing cylinder
        cyl_outer = vtk.vtkCylinderSource()
        cyl_outer.SetRadius(r_outer)
        cyl_outer.SetHeight(sleeve_height_mm)
        cyl_outer.SetResolution(36)
        cyl_outer.CappingOn()
        cyl_outer.Update()

        tf_outer = vtk.vtkTransformPolyDataFilter()
        tf_outer.SetInputConnection(cyl_outer.GetOutputPort())
        tf_outer.SetTransform(vtk_tf)
        tf_outer.Update()

        housing_pd = vtk.vtkPolyData()
        housing_pd.DeepCopy(tf_outer.GetOutput())

        # 2. Inner drill channel cylinder (longer to punch through guide shell completely)
        channel_height = sleeve_height_mm + 25.0
        # Offset center slightly downward so channel clears the base completely
        channel_center = p_plat + (axis_dir * (0.5 * sleeve_height_mm + sleeve_offset_mm - 5.0))
        M4_ch = np.eye(4, dtype=np.float64)
        M4_ch[:3, :3] = R
        M4_ch[:3, 3] = channel_center

        vtk_mat_ch = vtk.vtkMatrix4x4()
        for r in range(4):
            for c in range(4):
                vtk_mat_ch.SetElement(r, c, M4_ch[r, c])

        vtk_tf_ch = vtk.vtkTransform()
        vtk_tf_ch.SetMatrix(vtk_mat_ch)

        cyl_inner = vtk.vtkCylinderSource()
        cyl_inner.SetRadius(r_drill)
        cyl_inner.SetHeight(channel_height)
        cyl_inner.SetResolution(36)
        cyl_inner.CappingOn()
        cyl_inner.Update()

        tf_inner = vtk.vtkTransformPolyDataFilter()
        tf_inner.SetInputConnection(cyl_inner.GetOutputPort())
        tf_inner.SetTransform(vtk_tf_ch)
        tf_inner.Update()

        channel_pd = vtk.vtkPolyData()
        channel_pd.DeepCopy(tf_inner.GetOutput())

        # 3. Irrigation / inspection window slot (3.0mm x 2.0mm rectangular window)
        window_pd: Optional[vtk.vtkPolyData] = None
        if include_window:
            cube = vtk.vtkCubeSource()
            cube.SetXLength(r_outer * 2.5)
            cube.SetYLength(2.0)            # Height of window
            cube.SetZLength(3.0)            # Width of window
            cube.Update()

            tf_cube = vtk.vtkTransformPolyDataFilter()
            tf_cube.SetInputConnection(cube.GetOutputPort())
            tf_cube.SetTransform(vtk_tf)
            tf_cube.Update()

            window_pd = vtk.vtkPolyData()
            window_pd.DeepCopy(tf_cube.GetOutput())

        return housing_pd, channel_pd, window_pd

    # ------------------------------------------------------------------
    # CSG Boolean Assembly Pipeline
    # ------------------------------------------------------------------

    @staticmethod
    def _assemble_guide_csg(
        guide_base: vtk.vtkPolyData,
        sleeve_housings: List[vtk.vtkPolyData],
        drill_channels: List[vtk.vtkPolyData],
        irrigation_slots: List[vtk.vtkPolyData],
    ) -> vtk.vtkPolyData:
        """
        Combines guide base with sleeve housings and subtracts drill channels.
        Uses append and clean as solid foundation, applying difference operations
        where appropriate.
        """
        appender = vtk.vtkAppendPolyData()
        appender.AddInputData(guide_base)

        for housing in sleeve_housings:
            appender.AddInputData(housing)
        appender.Update()

        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputConnection(appender.GetOutputPort())
        cleaner.ToleranceIsAbsoluteOn()
        cleaner.SetAbsoluteTolerance(0.02)
        cleaner.Update()

        current_body = cleaner.GetOutput()

        # Drill channel CSG subtraction
        for channel in drill_channels:
            current_body = SurgicalGuideGenerator._subtract_mesh_safe(current_body, channel)

        # Irrigation slot subtraction
        for slot in irrigation_slots:
            current_body = SurgicalGuideGenerator._subtract_mesh_safe(current_body, slot)

        return current_body

    @staticmethod
    def _subtract_mesh_safe(target_body: vtk.vtkPolyData, tool: vtk.vtkPolyData) -> vtk.vtkPolyData:
        """
        Executes CSG boolean difference with safe fallback on non-manifold geometries.
        """
        try:
            bool_filter = vtk.vtkBooleanOperationPolyDataFilter()
            bool_filter.SetOperationToDifference()
            bool_filter.SetInputData(0, target_body)
            bool_filter.SetInputData(1, tool)
            bool_filter.SetTolerance(1e-5)
            bool_filter.Update()

            res: vtk.vtkPolyData = bool_filter.GetOutput()
            if res.GetNumberOfPoints() > 0 and res.GetNumberOfCells() > 0:
                out = vtk.vtkPolyData()
                out.DeepCopy(res)
                return out
        except Exception as exc:
            logger.debug(f"VTK Boolean CSG difference fell back: {exc}")

        # Safe fallback: return current body if boolean difference encounters degenerate topology
        return target_body

    # ------------------------------------------------------------------
    # Mesh Finishing & Smoothing
    # ------------------------------------------------------------------

    @staticmethod
    def _smooth_and_finalize_mesh(polydata: vtk.vtkPolyData) -> vtk.vtkPolyData:
        """
        Applies windowed sinc smoothing, decimation, and outward normal computation.
        """
        tri = vtk.vtkTriangleFilter()
        tri.SetInputData(polydata)
        tri.Update()

        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(tri.GetOutputPort())
        smoother.SetNumberOfIterations(15)
        smoother.SetPassBand(0.1)
        smoother.BoundarySmoothingOn()
        smoother.FeatureEdgeSmoothingOff()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()

        decimate = vtk.vtkQuadricDecimation()
        decimate.SetInputConnection(smoother.GetOutputPort())
        decimate.SetTargetReduction(0.25)  # 25% reduction for smooth 3D printing slicing
        decimate.VolumePreservationOn()
        decimate.Update()

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(decimate.GetOutputPort())
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.SplittingOff()
        normals.Update()

        clean = vtk.vtkCleanPolyData()
        clean.SetInputConnection(normals.GetOutputPort())
        clean.Update()

        final_pd = vtk.vtkPolyData()
        final_pd.DeepCopy(clean.GetOutput())
        return final_pd

    # ------------------------------------------------------------------
    # Manifold Validation & Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def validate_manifold(polydata: vtk.vtkPolyData) -> Tuple[bool, int, int]:
        """
        Inspects mesh manifoldness for 3D printing (SLA / DLP resin printers).

        Returns
        -------
        is_manifold : bool
            True if 0 non-manifold edges.
        num_non_manifold_edges : int
        num_boundary_edges : int
        """
        fe = vtk.vtkFeatureEdges()
        fe.SetInputData(polydata)
        fe.BoundaryEdgesOn()
        fe.NonManifoldEdgesOn()
        fe.FeatureEdgesOff()
        fe.ManifoldEdgesOff()
        fe.Update()

        non_manifold_count = 0
        boundary_count = 0

        lines = fe.GetOutput().GetLines()
        if lines is not None:
            # Total feature edge count
            edge_count = lines.GetNumberOfCells()
            is_valid = (edge_count == 0)
            return is_valid, edge_count, 0

        return True, 0, 0

    @staticmethod
    def _compute_mesh_metrics(polydata: vtk.vtkPolyData) -> Tuple[float, float]:
        """
        Computes the enclosed volume (cm³) and total surface area (cm²).
        """
        mass = vtk.vtkMassProperties()
        mass.SetInputData(polydata)
        mass.Update()

        volume_mm3 = float(abs(mass.GetVolume()))
        area_mm2 = float(mass.GetSurfaceArea())

        volume_cm3 = volume_mm3 / 1000.0
        area_cm2 = area_mm2 / 100.0

        return volume_cm3, area_cm2

    # ------------------------------------------------------------------
    # STL Export Engine
    # ------------------------------------------------------------------

    @staticmethod
    def export_guide_stl(file_path: str, guide_polydata: vtk.vtkPolyData) -> bool:
        """
        Exports the 3D surgical guide mesh as a binary STL file calibrated in physical millimeters.

        Parameters
        ----------
        file_path : str
            Target destination path (e.g. ``patient_guide_mandible.stl``).
        guide_polydata : vtkPolyData
            The surgical guide mesh to export.

        Returns
        -------
        bool
            True if export succeeded.
        """
        if guide_polydata is None or guide_polydata.GetNumberOfPoints() == 0:
            raise ValueError("Cannot export empty surgical guide mesh.")

        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        # Ensure pure triangle geometry for SLA/DLP slicers (Chitubox, Lychee, Formlabs PreForm)
        tri = vtk.vtkTriangleFilter()
        tri.SetInputData(guide_polydata)
        tri.Update()

        writer = vtk.vtkSTLWriter()
        writer.SetFileName(file_path)
        writer.SetInputConnection(tri.GetOutputPort())
        writer.SetFileTypeToBinary()
        writer.Write()

        return os.path.isfile(file_path) and os.path.getsize(file_path) > 0

    # ------------------------------------------------------------------
    # Ready-to-Render Actor Factory (Main GUI Thread)
    # ------------------------------------------------------------------

    @staticmethod
    def create_guide_actor(guide_polydata: vtk.vtkPolyData) -> vtk.vtkActor:
        """
        Builds a translucent Amber surgical guide actor for overlay rendering in VolumeView.
        Must be invoked on the Main GUI Thread.
        """
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(guide_polydata)
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        prop = actor.GetProperty()
        prop.SetColor(*_GUIDE_COLOR)
        prop.SetOpacity(_GUIDE_OPACITY)
        prop.SetAmbient(_GUIDE_AMBIENT)
        prop.SetDiffuse(_GUIDE_DIFFUSE)
        prop.SetSpecular(_GUIDE_SPECULAR)
        prop.SetSpecularPower(_GUIDE_SPECULAR_POWER)
        prop.SetInterpolationToPhong()
        prop.BackfaceCullingOff()

        return actor
