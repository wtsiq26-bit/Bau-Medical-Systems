"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Module: dental/mesh_registration.py

Intraoral Scan (IOS) Robust Two-Stage Registration Engine.
Registers imported STL/PLY optical surface scans onto CBCT-extracted tooth
surfaces using:
  Stage 1: Coarse Global Pre-Alignment (PCA / Oriented Bounding Box + 4-Quadrant Ambiguity Resolution)
  Stage 2: Fine Trimmed Robust ICP (vtkIterativeClosestPointTransform + SVD Inlier Trimming)

Mathematical Formulation:
-------------------------
1. Stage 1 (PCA / Inertial Axes):
   - Computes center of mass c_S and c_T for Source and Target.
   - Computes covariance matrix C = (1/N) * (P - c)^T * (P - c).
   - Eigendecomposition yields principal inertial axes V_S and V_T (sorted by eigenvalue).
   - Right-handedness is enforced: det(V) = +1.
   - Evaluates 4 candidate 180-degree quadrant reflection matrices:
       R_k = V_T * diag(s_x, s_y, s_z) * V_S^T  for (s_x, s_y, s_z) in {(1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)}
   - Selects candidate minimizing point-to-surface Chamfer distance to eliminate local minima traps.

2. Stage 2 (Fine Trimmed ICP & SVD Inlier Optimization):
   - Initial rigid body registration using vtkIterativeClosestPointTransform.
   - Iterative Trimmed SVD refinement (Horn/Umeyama) on the 85% closest inliers,
     completely discarding gingival soft-tissue and scan boundary outliers.
   - Computes comprehensive clinical quality metrics:
     * inlier_rms: RMS error on 85% inlier points (teeth contact zone).
     * max_error_95th: 95th percentile surface-to-surface residual distance.
     * quality_status: EXCELLENT (<0.15mm), ACCEPTABLE (<0.35mm), WARNING (<0.50mm), FAILED (>=0.50mm).
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registration Result Container
# ---------------------------------------------------------------------------

@dataclass
class RegistrationResult:
    """Encapsulates the output of a two-stage ICP registration run."""

    aligned_polydata: vtk.vtkPolyData
    """Source mesh after rigid transformation into the target coordinate frame."""

    aligned_actor: vtk.vtkActor
    """Ready-to-render actor with translucent IOS material applied."""

    transform_matrix: np.ndarray
    """4x4 homogeneous rigid-body transformation matrix (row-major)."""

    rms_error: float
    """Inlier Root-mean-square point-to-point residual error (mm)."""

    max_error_95th: float
    """95th percentile surface-to-surface residual distance (mm)."""

    inlier_rms: float
    """RMS error computed strictly on inlier points after trimming (mm)."""

    quality_status: str
    """Clinical status: 'EXCELLENT' | 'ACCEPTABLE' | 'WARNING' | 'FAILED'."""

    num_iterations: int
    """Actual number of ICP iterations before convergence."""


# ---------------------------------------------------------------------------
# IOS Actor Material Constants
# ---------------------------------------------------------------------------

_IOS_COLOR: Tuple[float, float, float] = (0.30, 0.75, 0.90)  # Electric Blue-Teal
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
    Two-Stage Rigid Surface Registration Engine for aligning Intraoral Scans (IOS)
    onto CBCT-segmented teeth meshes with PCA Global Pre-Alignment and Trimmed ICP.
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
            Absolute path to an .stl or .ply file.

        Returns
        -------
        vtkPolyData
            Loaded triangulated mesh.
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

        # Ensure point normals exist for consistent shading
        if polydata.GetPointData().GetNormals() is None:
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(polydata)
            normals.ComputePointNormalsOn()
            normals.ConsistencyOn()
            normals.AutoOrientNormalsOn()
            normals.SplittingOff()
            normals.Update()
            polydata = normals.GetOutput()

        out_pd = vtk.vtkPolyData()
        out_pd.DeepCopy(polydata)
        return out_pd

    # ------------------------------------------------------------------
    # Stage 1: Coarse Global Pre-Alignment (PCA + Quadrant Search)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_mesh_pca(polydata: vtk.vtkPolyData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the physical centroid, orthonormal principal axes (eigenvectors),
        and eigenvalues of a mesh. Enforces right-handedness det(V) = +1.

        Returns
        -------
        centroid : np.ndarray, shape (3,)
        eigenvectors : np.ndarray, shape (3, 3) (columns are v1, v2, v3)
        eigenvalues : np.ndarray, shape (3,) (descending order)
        """
        pts = vtk_to_numpy(polydata.GetPoints().GetData())
        if len(pts) == 0:
            raise ValueError("Mesh has 0 points for PCA.")

        centroid = np.mean(pts, axis=0)
        centered = pts - centroid

        cov = np.cov(centered, rowvar=False)  # shape (3, 3)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort descending by eigenvalue
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Enforce right-handed coordinate frame
        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, 2] = -eigvecs[:, 2]

        return centroid, eigvecs, eigvals

    @staticmethod
    def compute_global_initial_alignment(
        source: vtk.vtkPolyData,
        target: vtk.vtkPolyData,
        subsample_points: int = 1200,
    ) -> Tuple[vtk.vtkPolyData, np.ndarray, float]:
        """
        Computes a coarse global rigid alignment between source and target using PCA
        and evaluates all 4 valid 180-degree quadrant orientations to find the minimum
        Chamfer distance.

        Returns
        -------
        pre_aligned_polydata : vtkPolyData
        initial_transform_matrix : np.ndarray (4x4)
        best_chamfer_error : float (mm)
        """
        c_S, V_S, _ = MeshRegistrationEngine.compute_mesh_pca(source)
        c_T, V_T, _ = MeshRegistrationEngine.compute_mesh_pca(target)

        # 4 candidate reflection sign matrices (proper rotations with det = +1)
        sign_candidates = [
            np.diag([1.0,  1.0,  1.0]),  # Standard orientation
            np.diag([1.0, -1.0, -1.0]),  # 180 deg rotation around axis 1
            np.diag([-1.0,  1.0, -1.0]), # 180 deg rotation around axis 2
            np.diag([-1.0, -1.0,  1.0]), # 180 deg rotation around axis 3
        ]

        # Extract source points for Chamfer distance evaluation
        src_pts = vtk_to_numpy(source.GetPoints().GetData())
        n_pts = len(src_pts)
        if n_pts > subsample_points:
            stride = max(1, n_pts // subsample_points)
            eval_pts = src_pts[::stride]
        else:
            eval_pts = src_pts

        centered_src = eval_pts - c_S

        # Build target cell locator for closest-point queries
        locator = vtk.vtkCellLocator()
        locator.SetDataSet(target)
        locator.BuildLocator()

        best_error = float("inf")
        best_matrix = np.eye(4, dtype=np.float64)

        closest_point = [0.0, 0.0, 0.0]
        cell_id = vtk.reference(0)
        sub_id = vtk.reference(0)
        dist2 = vtk.reference(0.0)

        for S_mat in sign_candidates:
            R_cand = V_T @ S_mat @ V_S.T  # 3x3 rotation

            # Ensure proper rotation matrix
            if np.linalg.det(R_cand) < 0:
                continue

            t_cand = c_T - (R_cand @ c_S)

            # Transform evaluation points: P' = (R @ (P - c_S)^T)^T + c_T
            transformed_pts = (centered_src @ R_cand.T) + c_T

            # Compute mean closest-point distance (Chamfer metric)
            total_dist = 0.0
            for pt in transformed_pts:
                locator.FindClosestPoint(pt.tolist(), closest_point, cell_id, sub_id, dist2)
                total_dist += float(np.sqrt(dist2.get()))

            mean_dist = total_dist / len(transformed_pts)

            if mean_dist < best_error:
                best_error = mean_dist
                best_matrix = np.eye(4, dtype=np.float64)
                best_matrix[:3, :3] = R_cand
                best_matrix[:3, 3] = t_cand

        # Apply best initial transform matrix to source mesh
        mat4 = vtk.vtkMatrix4x4()
        for r in range(4):
            for c in range(4):
                mat4.SetElement(r, c, best_matrix[r, c])

        vtk_tf = vtk.vtkTransform()
        vtk_tf.SetMatrix(mat4)

        tf_filter = vtk.vtkTransformPolyDataFilter()
        tf_filter.SetInputData(source)
        tf_filter.SetTransform(vtk_tf)
        tf_filter.Update()

        pre_aligned_pd = vtk.vtkPolyData()
        pre_aligned_pd.DeepCopy(tf_filter.GetOutput())

        return pre_aligned_pd, best_matrix, best_error

    # ------------------------------------------------------------------
    # Stage 2: Fine Trimmed Robust ICP & SVD Optimization
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rigid_svd(
        source_pts: np.ndarray,
        target_pts: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the optimal rigid rotation matrix R (3x3) and translation vector t (3,)
        mapping source_pts -> target_pts via Kabsch/Horn/Umeyama SVD.
        """
        c_src = np.mean(source_pts, axis=0)
        c_tgt = np.mean(target_pts, axis=0)

        P = source_pts - c_src
        Q = target_pts - c_tgt

        H = P.T @ Q
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Reflect if det < 0
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = c_tgt - (R @ c_src)
        return R, t

    @staticmethod
    def register_icp_transform(
        source: vtk.vtkPolyData,
        target: vtk.vtkPolyData,
        max_iterations: int = 200,
        max_landmarks: int = 2500,
        tolerance: float = 0.001,
        trim_ratio: float = 0.15,
    ) -> Tuple[vtk.vtkPolyData, np.ndarray, float, int, float, str]:
        """
        Executes the full Two-Stage Robust Registration Pipeline:
          1. Global PCA Pre-Alignment + 4-Quadrant ambiguity search.
          2. Fine Rigid ICP registration.
          3. Iterative Trimmed SVD refinement on 85% inlier points (gingival rejection).
          4. Quality classification (EXCELLENT / ACCEPTABLE / WARNING / FAILED).

        Returns
        -------
        aligned_polydata : vtkPolyData
            Transformed source mesh in target coordinate frame.
        transform_matrix : np.ndarray (4x4)
            Full composite transformation matrix.
        inlier_rms : float
            Inlier RMS point-to-surface residual distance in mm.
        num_iterations : int
            Actual iterations performed.
        max_error_95th : float
            95th percentile residual surface distance in mm.
        quality_status : str
            'EXCELLENT' | 'ACCEPTABLE' | 'WARNING' | 'FAILED'.
        """
        if source.GetNumberOfPoints() == 0:
            raise ValueError("ICP source mesh is empty (0 vertices).")
        if target.GetNumberOfPoints() == 0:
            raise ValueError("ICP target mesh is empty (0 vertices).")

        # ---- Step 1: Global PCA Pre-Alignment ----
        pre_aligned_pd, M_init, _ = MeshRegistrationEngine.compute_global_initial_alignment(
            source, target, subsample_points=1200
        )

        # ---- Step 2: Fine Rigid ICP Registration ----
        icp = vtk.vtkIterativeClosestPointTransform()
        icp.SetSource(pre_aligned_pd)
        icp.SetTarget(target)
        icp.SetMaximumNumberOfIterations(max_iterations)
        icp.SetMaximumNumberOfLandmarks(max_landmarks)
        icp.SetMaximumMeanDistance(tolerance)
        icp.SetCheckMeanDistance(1)
        icp.StartByMatchingCentroidsOn()
        icp.GetLandmarkTransform().SetModeToRigidBody()
        icp.Modified()
        icp.Update()

        # Extract 4x4 matrix from initial ICP step
        vtk_mat: vtk.vtkMatrix4x4 = icp.GetMatrix()
        M_icp = np.eye(4, dtype=np.float64)
        for r in range(4):
            for c in range(4):
                M_icp[r, c] = vtk_mat.GetElement(r, c)

        M_composite = M_icp @ M_init

        # ---- Step 3: Trimmed SVD Inlier Refinement (Gingival Outlier Filtering) ----
        # Extract raw source points for iterative refinement
        raw_src_pts = vtk_to_numpy(source.GetPoints().GetData())
        n_pts = len(raw_src_pts)
        sample_stride = max(1, n_pts // max_landmarks)
        sample_indices = np.arange(0, n_pts, sample_stride)
        sampled_src = raw_src_pts[sample_indices]

        # Build target cell locator
        locator = vtk.vtkCellLocator()
        locator.SetDataSet(target)
        locator.BuildLocator()

        closest_point = [0.0, 0.0, 0.0]
        cell_id = vtk.reference(0)
        sub_id = vtk.reference(0)
        dist2 = vtk.reference(0.0)

        actual_iters = max_iterations
        try:
            actual_iters = int(icp.GetNumberOfIterations())
        except AttributeError:
            pass

        # Perform up to 15 Trimmed SVD iterations
        inlier_count = int(len(sampled_src) * (1.0 - trim_ratio))
        inlier_count = max(10, inlier_count)

        for trimmed_iter in range(15):
            # Transform sampled source points with current composite matrix
            R_curr = M_composite[:3, :3]
            t_curr = M_composite[:3, 3]
            transformed_sampled = (sampled_src @ R_curr.T) + t_curr

            # Find closest target points & distances
            target_matches = np.empty_like(transformed_sampled)
            dists = np.empty(len(transformed_sampled), dtype=np.float64)

            for i, pt in enumerate(transformed_sampled):
                locator.FindClosestPoint(pt.tolist(), closest_point, cell_id, sub_id, dist2)
                target_matches[i] = closest_point
                dists[i] = np.sqrt(dist2.get())

            # Sort and select top (1 - trim_ratio) inliers
            inlier_idx = np.argsort(dists)[:inlier_count]
            inlier_src = transformed_sampled[inlier_idx]
            inlier_tgt = target_matches[inlier_idx]

            # Solve incremental rigid transformation on inliers
            delta_R, delta_t = MeshRegistrationEngine._compute_rigid_svd(inlier_src, inlier_tgt)

            # Update composite matrix: T_new = [delta_R | delta_t] @ T_curr
            delta_M = np.eye(4, dtype=np.float64)
            delta_M[:3, :3] = delta_R
            delta_M[:3, 3] = delta_t
            M_composite = delta_M @ M_composite

            actual_iters += 1
            if np.linalg.norm(delta_t) < 1e-4 and np.abs(np.trace(delta_R) - 3.0) < 1e-5:
                break

        # ---- Step 4: Apply Final Composite Transform to Source Mesh ----
        mat4_total = vtk.vtkMatrix4x4()
        for r in range(4):
            for c in range(4):
                mat4_total.SetElement(r, c, M_composite[r, c])

        vtk_tf_total = vtk.vtkTransform()
        vtk_tf_total.SetMatrix(mat4_total)

        tf_filter = vtk.vtkTransformPolyDataFilter()
        tf_filter.SetInputData(source)
        tf_filter.SetTransform(vtk_tf_total)
        tf_filter.Update()

        aligned_pd = vtk.vtkPolyData()
        aligned_pd.DeepCopy(tf_filter.GetOutput())

        # ---- Step 5: Compute Final Residual Metrics on Transformed Mesh ----
        distances = MeshRegistrationEngine._compute_point_surface_distances(aligned_pd, target)
        if len(distances) == 0:
            distances = np.array([0.0])

        sorted_dists = np.sort(distances)
        cutoff_idx = int(len(distances) * (1.0 - trim_ratio))
        inlier_dists = sorted_dists[:max(1, cutoff_idx)]
        inlier_rms = float(np.sqrt(np.mean(inlier_dists ** 2)))
        max_error_95th = float(np.percentile(distances, 95))

        # ---- Step 6: Clinical Quality Classification ----
        if inlier_rms < 0.15:
            quality_status = "EXCELLENT"
        elif inlier_rms < 0.35:
            quality_status = "ACCEPTABLE"
        elif inlier_rms < 0.50:
            quality_status = "WARNING"
        else:
            quality_status = "FAILED"

        return aligned_pd, M_composite, inlier_rms, actual_iters, max_error_95th, quality_status

    # ------------------------------------------------------------------
    # IOS Actor Factory (Must execute on Main GUI Thread)
    # ------------------------------------------------------------------

    @staticmethod
    def create_ios_actor(polydata: vtk.vtkPolyData) -> vtk.vtkActor:
        """
        Create a distinctive IOS scan actor with translucent blue-teal
        material to distinguish it from CBCT-extracted surfaces.
        Must be called on the Main GUI Thread.
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
        prop.BackfaceCullingOff()

        return actor

    # ------------------------------------------------------------------
    # Legacy / Synchronous Convenience Wrapper
    # ------------------------------------------------------------------

    @classmethod
    def register_icp(
        cls,
        source: vtk.vtkPolyData,
        target: vtk.vtkPolyData,
        max_iterations: int = 200,
        max_landmarks: int = 2500,
        tolerance: float = 0.001,
        trim_ratio: float = 0.15,
    ) -> RegistrationResult:
        """Synchronous wrapper returning a comprehensive RegistrationResult."""
        aligned_pd, transform_np, inlier_rms, iters, max_95, status = cls.register_icp_transform(
            source, target, max_iterations, max_landmarks, tolerance, trim_ratio
        )
        aligned_actor = cls.create_ios_actor(aligned_pd)

        return RegistrationResult(
            aligned_polydata=aligned_pd,
            aligned_actor=aligned_actor,
            transform_matrix=transform_np,
            rms_error=inlier_rms,
            max_error_95th=max_95,
            inlier_rms=inlier_rms,
            quality_status=status,
            num_iterations=iters,
        )

    # ------------------------------------------------------------------
    # Distance Measurement Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_point_surface_distances(
        source: vtk.vtkPolyData,
        target: vtk.vtkPolyData,
        max_samples: int = 5000,
    ) -> np.ndarray:
        """
        Compute point-to-surface Euclidean distances from *source* vertices to *target*.
        Uses vtkCellLocator for fast O(log N) tree queries.
        """
        locator = vtk.vtkCellLocator()
        locator.SetDataSet(target)
        locator.BuildLocator()

        pts = vtk_to_numpy(source.GetPoints().GetData())
        n_pts = len(pts)
        if n_pts == 0:
            return np.array([0.0])

        if n_pts > max_samples:
            stride = max(1, n_pts // max_samples)
            sample_pts = pts[::stride]
        else:
            sample_pts = pts

        distances = np.empty(len(sample_pts), dtype=np.float64)
        closest_point = [0.0, 0.0, 0.0]
        cell_id = vtk.reference(0)
        sub_id = vtk.reference(0)
        dist2 = vtk.reference(0.0)

        for i, pt in enumerate(sample_pts):
            locator.FindClosestPoint(pt.tolist(), closest_point, cell_id, sub_id, dist2)
            distances[i] = float(np.sqrt(dist2.get()))

        return distances

    @staticmethod
    def _compute_rms_error(source: vtk.vtkPolyData, target: vtk.vtkPolyData) -> float:
        """Backward-compatible helper returning overall RMS distance."""
        dists = MeshRegistrationEngine._compute_point_surface_distances(source, target)
        return float(np.sqrt(np.mean(dists ** 2)))
