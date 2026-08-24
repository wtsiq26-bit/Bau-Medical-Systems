"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Module: dental/mesh_registration.py

Intraoral Scan (IOS) ICP Registration Engine.
Registers imported STL/PLY optical surface scans onto CBCT-extracted tooth
surfaces using VTK's vtkIterativeClosestPointTransform.

Engineering Notes:
- Centroid pre-alignment is enabled via StartByMatchingCentroidsOn() so that
  spatially distant scans converge reliably without a manual initial guess.
- The landmark transform is locked to rigid-body mode (rotation + translation
  only — no scaling or affine shear).
- Returns the 4×4 homogeneous transformation matrix, RMS fit error, and a
  pre-configured vtkActor with a distinctive translucent blue-teal material
  to visually distinguish the IOS scan from CBCT surfaces.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import vtk


# ---------------------------------------------------------------------------
# Registration Result Container
# ---------------------------------------------------------------------------

@dataclass
class RegistrationResult:
    """Encapsulates the output of an ICP registration run."""

    aligned_polydata: vtk.vtkPolyData
    """Source mesh after rigid transformation into the target coordinate frame."""

    aligned_actor: vtk.vtkActor
    """Ready-to-render actor with translucent IOS material applied."""

    transform_matrix: np.ndarray
    """4×4 homogeneous rigid-body transformation matrix (row-major)."""

    rms_error: float
    """Root-mean-square point-to-point residual error (mm)."""

    num_iterations: int
    """Actual number of ICP iterations before convergence / limit."""


# ---------------------------------------------------------------------------
# IOS Actor Material Constants
# ---------------------------------------------------------------------------

_IOS_COLOR: Tuple[float, float, float] = (0.30, 0.75, 0.90)
_IOS_OPACITY: float = 0.65
_IOS_AMBIENT: float = 0.25
_IOS_DIFFUSE: float = 0.70
_IOS_SPECULAR: float = 0.45
_IOS_SPECULAR_POWER: float = 35.0


# ---------------------------------------------------------------------------
# Mesh Registration Engine
# ---------------------------------------------------------------------------

class MeshRegistrationEngine:
    """
    Loads and registers intraoral optical scans (IOS) onto CBCT tooth surfaces
    via rigid ICP.

    Typical usage::

        engine = MeshRegistrationEngine()
        ios_pd = engine.load_mesh("scan.stl")
        ios_actor = engine.create_ios_actor(ios_pd)

        # After extracting teeth surface from segmentation:
        result = engine.register_icp(source=ios_pd, target=teeth_polydata)
        volume_view.remove_ios_scan_actor()
        volume_view.add_ios_scan_actor(result.aligned_actor)
    """

    # ------------------------------------------------------------------
    # Mesh I/O
    # ------------------------------------------------------------------

    @staticmethod
    def load_mesh(file_path: str) -> vtk.vtkPolyData:
        """
        Load a triangulated surface mesh from STL or PLY.

        Parameters
        ----------
        file_path : str
            Absolute path to an ``.stl`` or ``.ply`` file.

        Returns
        -------
        vtkPolyData
            Loaded triangulated mesh.

        Raises
        ------
        FileNotFoundError
            If *file_path* does not exist.
        ValueError
            If the file extension is unsupported.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Mesh file not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".stl":
            reader = vtk.vtkSTLReader()
        elif ext == ".ply":
            reader = vtk.vtkPLYReader()
        else:
            raise ValueError(
                f"Unsupported mesh format '{ext}'. "
                "Only .stl and .ply are supported."
            )

        reader.SetFileName(file_path)
        reader.Update()

        polydata: vtk.vtkPolyData = reader.GetOutput()
        if polydata.GetNumberOfPoints() == 0:
            raise RuntimeError(
                f"Mesh file loaded but contains 0 vertices: {file_path}"
            )

        # Ensure normals exist for consistent rendering
        if polydata.GetPointData().GetNormals() is None:
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(polydata)
            normals.ComputePointNormalsOn()
            normals.ConsistencyOn()
            normals.AutoOrientNormalsOn()
            normals.SplittingOff()
            normals.Update()
            polydata = normals.GetOutput()

        return polydata

    # ------------------------------------------------------------------
    # ICP Registration
    # ------------------------------------------------------------------

    @staticmethod
    def register_icp(
        source: vtk.vtkPolyData,
        target: vtk.vtkPolyData,
        max_iterations: int = 200,
        max_landmarks: int = 2000,
        tolerance: float = 1e-6,
    ) -> RegistrationResult:
        """
        Perform rigid ICP alignment of *source* onto *target*.

        Parameters
        ----------
        source : vtkPolyData
            The IOS scan to be transformed (source → target).
        target : vtkPolyData
            The CBCT-extracted tooth surface (reference frame).
        max_iterations : int
            Maximum number of ICP iterations (default 200).
        max_landmarks : int
            Number of point correspondences sampled per iteration (default 2000).
        tolerance : float
            Convergence threshold on mean distance (default 1e-6 mm).

        Returns
        -------
        RegistrationResult
            Contains aligned mesh, actor, 4×4 matrix, RMS error, and
            iteration count.
        """
        if source.GetNumberOfPoints() == 0:
            raise ValueError("ICP source mesh is empty (0 vertices).")
        if target.GetNumberOfPoints() == 0:
            raise ValueError("ICP target mesh is empty (0 vertices).")

        # ---- Configure ICP ----
        icp = vtk.vtkIterativeClosestPointTransform()
        icp.SetSource(source)
        icp.SetTarget(target)
        icp.SetMaximumNumberOfIterations(max_iterations)
        icp.SetMaximumNumberOfLandmarks(max_landmarks)
        icp.SetMaximumMeanDistance(tolerance)
        icp.SetCheckMeanDistance(1)

        # Engineering Refinement #2: centroid pre-alignment + rigid body lock
        icp.StartByMatchingCentroidsOn()
        icp.GetLandmarkTransform().SetModeToRigidBody()

        icp.Modified()
        icp.Update()

        # ---- Extract 4×4 transformation matrix ----
        vtk_mat: vtk.vtkMatrix4x4 = icp.GetMatrix()
        transform_np = np.eye(4, dtype=np.float64)
        for r in range(4):
            for c in range(4):
                transform_np[r, c] = vtk_mat.GetElement(r, c)

        # ---- Apply transform to a copy of the source mesh ----
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputData(source)
        transform_filter.SetTransform(icp)
        transform_filter.Update()

        aligned_pd: vtk.vtkPolyData = vtk.vtkPolyData()
        aligned_pd.DeepCopy(transform_filter.GetOutput())

        # ---- Compute RMS residual ----
        rms = MeshRegistrationEngine._compute_rms_error(aligned_pd, target)

        # ---- Build rendering actor ----
        aligned_actor = MeshRegistrationEngine.create_ios_actor(aligned_pd)

        num_iters: int = icp.GetMaximumNumberOfIterations()  # VTK doesn't expose actual; use max as upper bound
        # Attempt to read actual iteration count if available
        try:
            num_iters = int(icp.GetNumberOfIterations())  # type: ignore[attr-defined]
        except AttributeError:
            pass

        return RegistrationResult(
            aligned_polydata=aligned_pd,
            aligned_actor=aligned_actor,
            transform_matrix=transform_np,
            rms_error=rms,
            num_iterations=num_iters,
        )

    # ------------------------------------------------------------------
    # IOS Actor Factory
    # ------------------------------------------------------------------

    @staticmethod
    def create_ios_actor(polydata: vtk.vtkPolyData) -> vtk.vtkActor:
        """
        Create a distinctive IOS scan actor with translucent blue-teal
        material to distinguish it from CBCT-extracted surfaces.

        Parameters
        ----------
        polydata : vtkPolyData
            The intraoral scan mesh.

        Returns
        -------
        vtkActor
            Configured rendering actor.
        """
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        prop = actor.GetProperty()
        prop.SetColor(*_IOS_COLOR)
        prop.SetOpacity(_IOS_OPACITY)
        prop.SetAmbient(_IOS_AMBIENT)
        prop.SetDiffuse(_IOS_DIFFUSE)
        prop.SetSpecular(_IOS_SPECULAR)
        prop.SetSpecularPower(_IOS_SPECULAR_POWER)
        prop.SetInterpolationToPhong()
        prop.BackfaceCullingOff()        # IOS scans may not be watertight

        return actor

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rms_error(
        source: vtk.vtkPolyData,
        target: vtk.vtkPolyData,
    ) -> float:
        """
        Compute RMS point-to-surface distance from *source* to *target*.
        Uses vtkCellLocator for efficient closest-point queries.
        """
        locator = vtk.vtkCellLocator()
        locator.SetDataSet(target)
        locator.BuildLocator()

        n_pts = source.GetNumberOfPoints()
        if n_pts == 0:
            return 0.0

        # Sample up to 5000 points for performance
        step = max(1, n_pts // 5000)
        sum_sq = 0.0
        count = 0

        closest_point = [0.0, 0.0, 0.0]
        cell_id = vtk.reference(0)
        sub_id = vtk.reference(0)
        dist2 = vtk.reference(0.0)

        for i in range(0, n_pts, step):
            pt = source.GetPoint(i)
            locator.FindClosestPoint(pt, closest_point, cell_id, sub_id, dist2)
            sum_sq += float(dist2)
            count += 1

        return float(np.sqrt(sum_sq / max(count, 1)))
