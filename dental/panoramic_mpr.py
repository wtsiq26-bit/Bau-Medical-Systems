"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Module: dental/panoramic_mpr.py

Curved Multi-Planar Reconstruction (Curved MPR) and Bucco-Lingual Cross-Sectional Slicing Engine.
Features:
- Equidistant Arc-Length Parameterized Dental Arch Spline $\\mathcal{C}(s)$.
- Local Orthonormal Darboux / Frenet Frame computation (Tangent $\\vec{T}$, Normal $\\vec{N}$, Vertical $\\vec{V}_z$).
- Fast Vectorized Focal Trough Panoramic Volume Unrolling via scipy.ndimage.map_coordinates (MIP, Average, Thin Slice).
- Transverse Bucco-Lingual Cross-Section Matrix Generation ($4\\times 4$ vtkMatrix4x4) for implant planning.
- Parabolic Mandibular Arch Auto-Fitting.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import math
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import map_coordinates
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from PySide6.QtCore import QObject, Signal
from core.volume_data import VolumeData


class DentalArchSignals(QObject):
    """Signals emitted when the dental arch curve or cross-sections are updated."""
    arch_updated = Signal(int, float)            # num_points, total_length_mm
    cross_section_changed = Signal(int, float)   # slice_index, arc_pos_mm
    panoramic_regenerated = Signal()


class DentalArchCurve:
    """
    Manages the Dental Arch Spline Curve $\\mathcal{C}(s)$ drawn on the Axial slice.
    """

    def __init__(self, step_size_mm: float = 0.5) -> None:
        self.step_size_mm = step_size_mm
        self.seed_points: List[Tuple[float, float, float]] = []

        # Parameterized sampled curve points, tangents, and normals
        self.sampled_points: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.tangents: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.normals: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.arc_lengths: np.ndarray = np.empty((0,), dtype=np.float32)
        self.total_length_mm: float = 0.0
        self.arch_z: float = 0.0

    def add_seed_point(self, x: float, y: float, z: float) -> None:
        """Adds a seed point to the dental arch curve."""
        self.seed_points.append((float(x), float(y), float(z)))
        self.arch_z = float(z)
        self.rebuild_curve()

    def undo_last_point(self) -> Optional[Tuple[float, float, float]]:
        """Removes the last seed point."""
        if not self.seed_points:
            return None
        removed = self.seed_points.pop()
        self.rebuild_curve()
        return removed

    def clear(self) -> None:
        """Clears all seed points and sampled geometry."""
        self.seed_points.clear()
        self.sampled_points = np.empty((0, 3), dtype=np.float32)
        self.tangents = np.empty((0, 3), dtype=np.float32)
        self.normals = np.empty((0, 3), dtype=np.float32)
        self.arc_lengths = np.empty((0,), dtype=np.float32)
        self.total_length_mm = 0.0

    def auto_fit_parabola(self, volume: VolumeData, z_world: Optional[float] = None) -> None:
        """
        Auto-fits an anatomical parabolic mandibular arch curve based on volume physical bounds.
        """
        cx, cy, cz = volume.get_center()
        target_z = z_world if z_world is not None else cz
        bounds = volume.get_bounds()
        w = (bounds[1] - bounds[0]) * 0.35
        h = (bounds[3] - bounds[2]) * 0.28

        # 7 parabolic anatomical anchor points from right condyle/molar to left molar
        xs = np.linspace(-w, w, 7)
        self.clear()
        for x in xs:
            # Parabolic equation: y = a * x^2 + y_offset
            y = cy - h + (0.015 * (x ** 2))
            self.seed_points.append((float(cx + x), float(y), float(target_z)))

        self.arch_z = float(target_z)
        self.rebuild_curve()

    def rebuild_curve(self) -> None:
        """
        Reconstructs cubic spline, performs arc-length reparameterization at uniform $\\Delta s$,
        and calculates local orthonormal Darboux frames (Tangent, Normal, Vertical).
        """
        n = len(self.seed_points)
        if n < 2:
            self.sampled_points = np.array(self.seed_points, dtype=np.float32).reshape((-1, 3))
            self.tangents = np.empty((0, 3), dtype=np.float32)
            self.normals = np.empty((0, 3), dtype=np.float32)
            self.arc_lengths = np.zeros((n,), dtype=np.float32)
            self.total_length_mm = 0.0
            return

        pts = np.array(self.seed_points, dtype=np.float64)

        # Cumulative chord distance
        dists = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
        cumulative_dist = np.insert(np.cumsum(dists), 0, 0.0)
        total_dist = cumulative_dist[-1]
        self.total_length_mm = float(total_dist)

        if total_dist < 1e-4:
            return

        # Fit cubic spline parameterization over cumulative distance
        cs_x = CubicSpline(cumulative_dist, pts[:, 0])
        cs_y = CubicSpline(cumulative_dist, pts[:, 1])
        cs_z = CubicSpline(cumulative_dist, pts[:, 2])

        # Sample at uniform arc-length intervals
        num_samples = max(2, int(np.ceil(total_dist / self.step_size_mm)))
        s_vals = np.linspace(0.0, total_dist, num_samples)

        sampled_x = cs_x(s_vals)
        sampled_y = cs_y(s_vals)
        sampled_z = cs_z(s_vals)

        self.sampled_points = np.column_stack([sampled_x, sampled_y, sampled_z]).astype(np.float32)
        self.arc_lengths = s_vals.astype(np.float32)

        # Compute 1st derivatives (Tangent vectors: dx/ds, dy/ds, dz/ds)
        dx = cs_x(s_vals, 1)
        dy = cs_y(s_vals, 1)
        dz = np.zeros_like(dx)  # Dental arch in planar axial projection

        tangent_norms = np.sqrt(dx ** 2 + dy ** 2 + 1e-8)
        tx = dx / tangent_norms
        ty = dy / tangent_norms
        tz = np.zeros_like(tx)

        self.tangents = np.column_stack([tx, ty, tz]).astype(np.float32)

        # Normal vectors (Bucco-Lingual axis: N = T x V_z = (-ty, tx, 0))
        nx = -ty
        ny = tx
        nz = np.zeros_like(nx)
        self.normals = np.column_stack([nx, ny, nz]).astype(np.float32)

    def get_cross_section_tick_endpoints(self, tick_length_mm: float = 8.0) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """
        Returns line segment coordinates for perpendicular transverse slice ticks along the curve.
        """
        ticks = []
        half_len = tick_length_mm * 0.5
        for i in range(len(self.sampled_points)):
            p = self.sampled_points[i]
            n = self.normals[i]
            p_buccal = (float(p[0] + n[0] * half_len), float(p[1] + n[1] * half_len), float(p[2]))
            p_lingual = (float(p[0] - n[0] * half_len), float(p[1] - n[1] * half_len), float(p[2]))
            ticks.append((p_buccal, p_lingual))
        return ticks


class PanoramicGenerator:
    """
    Synthesizes the unrolled 2D Panoramic Radiograph (Focal Trough) from 3D VolumeData.
    """

    def __init__(self) -> None:
        self.focal_trough_thickness_mm: float = 8.0  # Thickness of focal trough slab (mm)
        self.projection_mode: str = "mip"             # 'mip' | 'average' | 'thin'
        self.vertical_height_mm: float = 60.0         # Height of panoramic view in mm
        self.vertical_step_mm: float = 0.5            # Z resolution in mm

    def generate_panoramic_image(
        self,
        volume: VolumeData,
        arch_curve: DentalArchCurve
    ) -> Optional[np.ndarray]:
        """
        Vectorized Focal Trough Unrolling via 3D trilinear interpolation.
        Returns a 2D NumPy array of shape (Height_Z, ArcLength_S) in Hounsfield Units.
        """
        if len(arch_curve.sampled_points) < 2:
            return None

        # 1. Coordinate Grid Setup
        num_s = len(arch_curve.sampled_points)
        bounds = volume.get_bounds()
        z_min, z_max = bounds[4], bounds[5]
        z_vals = np.arange(z_min, z_max, self.vertical_step_mm, dtype=np.float32)
        num_z = len(z_vals)

        # Focal trough normal sampling offsets
        if self.projection_mode == "thin" or self.focal_trough_thickness_mm <= 1.0:
            u_offsets = np.array([0.0], dtype=np.float32)
        else:
            half_w = self.focal_trough_thickness_mm * 0.5
            num_u = max(3, int(np.ceil(self.focal_trough_thickness_mm / 0.8)))
            u_offsets = np.linspace(-half_w, half_w, num_u, dtype=np.float32)

        # 2. Build 3D Physical Sampling Coordinates Grid (U_samples x Z x S)
        pts_s = arch_curve.sampled_points  # Shape: (S, 3)
        normals_s = arch_curve.normals     # Shape: (S, 3)

        # World to Voxel continuous matrix conversion
        inv_spacing = 1.0 / np.array(volume.spacing, dtype=np.float32)
        origin = np.array(volume.origin, dtype=np.float32)

        # Accumulator for unrolled panoramic slice
        panoramic_accum = []

        for u in u_offsets:
            # Shift curve points along normal
            shifted_xy = pts_s[:, :2] + u * normals_s[:, :2]  # Shape: (S, 2)

            # Meshgrid for S and Z
            # X_world: shape (Z, S)
            X_world = np.tile(shifted_xy[:, 0], (num_z, 1))
            Y_world = np.tile(shifted_xy[:, 1], (num_z, 1))
            Z_world = np.repeat(z_vals[:, np.newaxis], num_s, axis=1)

            # Convert to continuous voxel indices (i, j, k)
            i_voxel = (X_world - origin[0]) * inv_spacing[0]
            j_voxel = (Y_world - origin[1]) * inv_spacing[1]
            k_voxel = (Z_world - origin[2]) * inv_spacing[2]

            # Shape for map_coordinates: (3, Z * S) with order (k, j, i)
            coords = np.vstack([k_voxel.ravel(), j_voxel.ravel(), i_voxel.ravel()])

            # Sample 3D scalar volume using trilinear interpolation
            sampled_slab = map_coordinates(
                volume.numpy_array,
                coords,
                order=1,
                mode='nearest',
                cval=-1000.0
            ).reshape((num_z, num_s))

            panoramic_accum.append(sampled_slab)

        # 3. Apply Projection Mode (MIP vs Average)
        stacked = np.stack(panoramic_accum, axis=0)  # Shape: (U, Z, S)

        if self.projection_mode == "mip":
            panoramic_img = np.max(stacked, axis=0)
        else:
            panoramic_img = np.mean(stacked, axis=0)

        # Flip vertically so Superior is on top (Row 0 = Superior)
        panoramic_img = np.flipud(panoramic_img).astype(np.float32)
        return panoramic_img

    def generate_panoramic_vtk_image(
        self,
        volume: VolumeData,
        arch_curve: DentalArchCurve
    ) -> Optional[vtk.vtkImageData]:
        """Converts the unrolled panoramic image into a vtkImageData object."""
        img_arr = self.generate_panoramic_image(volume, arch_curve)
        if img_arr is None:
            return None

        h, w = img_arr.shape
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(w, h, 1)
        vtk_image.SetSpacing(arch_curve.step_size_mm, self.vertical_step_mm, 1.0)
        vtk_image.SetOrigin(0.0, 0.0, 0.0)

        flat_arr = np.ascontiguousarray(img_arr.ravel(), dtype=np.int16)
        vtk_scalars = numpy_to_vtk(flat_arr, deep=True, array_type=vtk.VTK_SHORT)
        vtk_image.GetPointData().SetScalars(vtk_scalars)
        return vtk_image


class CrossSectionManager:
    """
    Manages Transverse Bucco-Lingual Cross-Section slices along the dental arch curve.
    """

    def __init__(self, arch_curve: DentalArchCurve) -> None:
        self.arch_curve = arch_curve
        self.active_index: int = 0
        self.slice_spacing_mm: float = 1.0
        self.slice_width_mm: float = 30.0   # Bucco-Lingual span
        self.slice_height_mm: float = 35.0  # Vertical span

    @property
    def total_cross_sections(self) -> int:
        return len(self.arch_curve.sampled_points)

    def set_active_index(self, index: int) -> None:
        """Sets the active cross-section index along the arch curve."""
        n = self.total_cross_sections
        if n > 0:
            self.active_index = max(0, min(index, n - 1))

    def get_active_world_position(self) -> Optional[Tuple[float, float, float]]:
        """Returns 3D world coordinate of active cross-section center."""
        if 0 <= self.active_index < len(self.arch_curve.sampled_points):
            pt = self.arch_curve.sampled_points[self.active_index]
            return (float(pt[0]), float(pt[1]), float(pt[2]))
        return None

    def get_reslice_matrix_for_index(self, index: int) -> Optional[vtk.vtkMatrix4x4]:
        """
        Constructs the 4x4 Reslice Axes Matrix for a transverse Bucco-Lingual cross-section:
        - Column 0 (Screen X / Horizontal): Normal vector N (Bucco-Lingual axis)
        - Column 1 (Screen Y / Vertical): Vertical vector V_z = (0, 0, 1) (Apical-Coronal axis)
        - Column 2 (Screen Z / Slice Normal): Tangent vector T (Mesial-Distal axis)
        - Column 3 (Origin): Center coordinate C(s)
        """
        if index < 0 or index >= len(self.arch_curve.sampled_points):
            return None

        p = self.arch_curve.sampled_points[index]
        t = self.arch_curve.tangents[index]
        n = self.arch_curve.normals[index]
        v_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        matrix = vtk.vtkMatrix4x4()
        # Col 0: Normal N (X-axis)
        matrix.SetElement(0, 0, n[0])
        matrix.SetElement(1, 0, n[1])
        matrix.SetElement(2, 0, n[2])
        matrix.SetElement(3, 0, 0.0)

        # Col 1: Vertical V_z (Y-axis)
        matrix.SetElement(0, 1, v_z[0])
        matrix.SetElement(1, 1, v_z[1])
        matrix.SetElement(2, 1, v_z[2])
        matrix.SetElement(3, 1, 0.0)

        # Col 2: Tangent T (Slice Normal)
        matrix.SetElement(0, 2, t[0])
        matrix.SetElement(1, 2, t[1])
        matrix.SetElement(2, 2, t[2])
        matrix.SetElement(3, 2, 0.0)

        # Col 3: Origin P(s)
        matrix.SetElement(0, 3, p[0])
        matrix.SetElement(1, 3, p[1])
        matrix.SetElement(2, 3, p[2])
        matrix.SetElement(3, 3, 1.0)

        return matrix
