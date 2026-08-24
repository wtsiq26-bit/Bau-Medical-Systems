"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Module: core/volume_data.py

Provides the core VolumeData container for 3D medical volumes.
Features:
- Zero-copy NumPy to VTK scalar conversion with guaranteed lifetime management.
- Medical coordinate transformations (DICOM LPS to VTK World).
- Voxel-to-World and World-to-Voxel conversion matrix helpers.
- Direction cosine and orientation matrix reconstruction.
- Real-time HU value sampling and bounds calculation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, get_vtk_array_type


@dataclass
class DicomMetadata:
    """Encapsulates clinically critical DICOM patient and acquisition metadata."""
    patient_name: str = "Anonymous"
    patient_id: str = "000000"
    patient_birth_date: str = "N/A"
    patient_sex: str = "O"
    study_description: str = "Dental CBCT"
    series_description: str = "CBCT 3D Volume"
    study_date: str = "N/A"
    modality: str = "CT"
    manufacturer: str = "Bau Medical Systems"
    rescale_slope: float = 1.0
    rescale_intercept: float = 0.0
    window_center: float = 500.0
    window_width: float = 2500.0
    fov_dimensions_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    raw_tags: Dict[str, Any] = field(default_factory=dict)


class VolumeData:
    """
    High-performance, memory-safe 3D Volume container for Dental CBCT data.

    Technical Notes on Memory & Coordinate Spaces:
    ---------------------------------------------
    1. Zero-Copy Lifetime Management:
       VTK's `numpy_to_vtk` creates a shallow pointer to the underlying NumPy array.
       If Python's garbage collector frees the NumPy ndarray while VTK render pipelines
       are reading it, segmentation faults and memory corruption occur.
       `VolumeData` explicitly stores `_numpy_array` as a persistent member, pinning
       the underlying buffer in memory for the exact lifetime of the VTK object.

    2. Memory Layout (NumPy vs. VTK):
       - NumPy array layout: Shape is (Z, Y, X) representing (Slice, Row, Column).
       - VTK ImageData layout: Dimensions are (Nx, Ny, Nz) with fast-axis X, then Y, then Z.
       - To achieve zero-copy without continuous transpositions during rendering, the NumPy
         array is arranged such that its flat 1D C-buffer matches VTK's scalar order:
         voxel(x, y, z) = buffer[x + y * Nx + z * Nx * Ny].
         Thus, we format NumPy as C-contiguous array with shape (Nz, Ny, Nx).ravel()
         or directly pass the contiguous 3D array flattened.

    3. DICOM LPS vs. VTK Coordinate System:
       - DICOM uses the LPS (Left, Posterior, Superior) patient coordinate space.
       - Direction cosines [Xx, Xy, Xz, Yx, Yy, Yz] define the row and column unit vectors.
       - The slice normal Z is defined by the cross-product of X and Y: Z = X x Y.
       - A 4x4 homogenous matrix M converts voxel index (i, j, k, 1)^T to Patient World (x, y, z, 1)^T.
    """

    def __init__(
        self,
        array: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        direction: Optional[Tuple[float, ...]] = None,
        metadata: Optional[DicomMetadata] = None,
    ) -> None:
        """
        Initialize VolumeData.

        :param array: 3D NumPy ndarray of shape (Nz, Ny, Nx) in Hounsfield Units (int16/float32).
        :param spacing: Voxel spacing (dx, dy, dz) in millimeters (mm).
        :param origin: Origin (x0, y0, z0) in millimeters (DICOM LPS space).
        :param direction: 3x3 direction cosine matrix flattened (9 elements) or None.
        :param metadata: DicomMetadata object containing DICOM header parameters.
        """
        # 1. Enforce C-contiguous memory layout for zero-copy VTK compatibility
        if not array.flags.c_contiguous:
            self._numpy_array: np.ndarray = np.ascontiguousarray(array)
        else:
            self._numpy_array = array

        # Validate 3D shape
        if self._numpy_array.ndim != 3:
            raise ValueError(f"VolumeData requires a 3D array (Nz, Ny, Nx), got shape {self._numpy_array.shape}")

        self.nz, self.ny, self.nx = self._numpy_array.shape
        self.spacing: Tuple[float, float, float] = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
        self.origin: Tuple[float, float, float] = (float(origin[0]), float(origin[1]), float(origin[2]))

        # Default direction is standard identity (LPS aligned)
        if direction is None or len(direction) != 9:
            self.direction: Tuple[float, ...] = (
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0
            )
        else:
            self.direction = tuple(float(d) for d in direction)

        self.metadata: DicomMetadata = metadata if metadata is not None else DicomMetadata()

        # Compute HU scalar statistics
        self.min_hu: float = float(np.min(self._numpy_array))
        self.max_hu: float = float(np.max(self._numpy_array))
        self.mean_hu: float = float(np.mean(self._numpy_array))

        # 2. Build Zero-Copy VTK ImageData
        self._vtk_image_data: vtk.vtkImageData = self._create_vtk_image_data()

        # 3. Construct 4x4 Index-To-World Transformation Matrix
        self._index_to_world_matrix: vtk.vtkMatrix4x4 = self._build_index_to_world_matrix()
        self._world_to_index_matrix: vtk.vtkMatrix4x4 = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(self._index_to_world_matrix, self._world_to_index_matrix)

    @property
    def numpy_array(self) -> np.ndarray:
        """Return the anchored 3D NumPy array (Nz, Ny, Nx)."""
        return self._numpy_array

    @property
    def array(self) -> np.ndarray:
        """Alias for numpy_array (Nz, Ny, Nx)."""
        return self._numpy_array

    @property
    def vtk_image_data(self) -> vtk.vtkImageData:
        """Return the zero-copy vtkImageData object."""
        return self._vtk_image_data

    @property
    def dimensions(self) -> Tuple[int, int, int]:
        """Return dimensions in VTK order (Nx, Ny, Nz)."""
        return (self.nx, self.ny, self.nz)

    @property
    def physical_size_mm(self) -> Tuple[float, float, float]:
        """Return physical Field of View (FoV) size in mm (Lx, Ly, Lz)."""
        return (
            self.nx * self.spacing[0],
            self.ny * self.spacing[1],
            self.nz * self.spacing[2],
        )

    def get_bounds(self) -> Tuple[float, float, float, float, float, float]:
        """Return spatial bounding box in VTK world space: (xmin, xmax, ymin, ymax, zmin, zmax)."""
        return self._vtk_image_data.GetBounds()

    def get_center(self) -> Tuple[float, float, float]:
        """Return the physical center of the volume (cx, cy, cz) in millimeters."""
        bounds = self.get_bounds()
        return (
            (bounds[0] + bounds[1]) * 0.5,
            (bounds[2] + bounds[3]) * 0.5,
            (bounds[4] + bounds[5]) * 0.5,
        )

    def _create_vtk_image_data(self) -> vtk.vtkImageData:
        """
        Constructs a vtkImageData object mapped directly to the NumPy array buffer.
        Zero-copy is guaranteed via `numpy_to_vtk(..., deep=False)`.
        """
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(self.nx, self.ny, self.nz)
        image_data.SetSpacing(self.spacing[0], self.spacing[1], self.spacing[2])
        image_data.SetOrigin(self.origin[0], self.origin[1], self.origin[2])

        # NumPy shape (Nz, Ny, Nx) in C-contiguous memory matches VTK's (X + Y*Nx + Z*Nx*Ny) stride
        flat_buffer = self._numpy_array.ravel()
        vtk_scalars = numpy_to_vtk(
            num_array=flat_buffer,
            deep=False,
            array_type=get_vtk_array_type(flat_buffer.dtype)
        )
        vtk_scalars.SetName("HounsfieldUnits")
        image_data.GetPointData().SetScalars(vtk_scalars)
        return image_data

    def _build_index_to_world_matrix(self) -> vtk.vtkMatrix4x4:
        """
        Constructs the 4x4 Affine Matrix transforming Voxel (i, j, k, 1) -> Patient LPS World (x, y, z, 1).
        M = [
            [ d00*dx, d01*dy, d02*dz, origin_x ],
            [ d10*dx, d11*dy, d12*dz, origin_y ],
            [ d20*dx, d21*dy, d22*dz, origin_z ],
            [      0,      0,      0,        1 ]
        ]
        """
        m = vtk.vtkMatrix4x4()
        d = self.direction
        dx, dy, dz = self.spacing
        ox, oy, oz = self.origin

        # Column 0: X vector scaled by dx
        m.SetElement(0, 0, d[0] * dx)
        m.SetElement(1, 0, d[3] * dx)
        m.SetElement(2, 0, d[6] * dx)
        m.SetElement(3, 0, 0.0)

        # Column 1: Y vector scaled by dy
        m.SetElement(0, 1, d[1] * dy)
        m.SetElement(1, 1, d[4] * dy)
        m.SetElement(2, 1, d[7] * dy)
        m.SetElement(3, 0, 0.0)

        # Column 2: Z vector scaled by dz
        m.SetElement(0, 2, d[2] * dz)
        m.SetElement(1, 2, d[5] * dz)
        m.SetElement(2, 2, d[8] * dz)
        m.SetElement(3, 2, 0.0)

        # Column 3: Origin translation
        m.SetElement(0, 3, ox)
        m.SetElement(1, 3, oy)
        m.SetElement(2, 3, oz)
        m.SetElement(3, 3, 1.0)

        return m

    def index_to_world(self, i: float, j: float, k: float) -> Tuple[float, float, float]:
        """Convert continuous voxel index (i, j, k) to Patient LPS World coordinates (x, y, z)."""
        p_in = [i, j, k, 1.0]
        p_out = [0.0, 0.0, 0.0, 0.0]
        self._index_to_world_matrix.MultiplyPoint(p_in, p_out)
        return (p_out[0], p_out[1], p_out[2])

    def world_to_index(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """Convert Patient LPS World coordinates (x, y, z) to nearest clamped discrete voxel index (i, j, k)."""
        p_in = [x, y, z, 1.0]
        p_out = [0.0, 0.0, 0.0, 0.0]
        self._world_to_index_matrix.MultiplyPoint(p_in, p_out)
        i = int(np.clip(round(p_out[0]), 0, self.nx - 1))
        j = int(np.clip(round(p_out[1]), 0, self.ny - 1))
        k = int(np.clip(round(p_out[2]), 0, self.nz - 1))
        return (i, j, k)

    def get_hu_at_voxel(self, i: int, j: int, k: int) -> float:
        """Sample Hounsfield Unit at voxel index (i, j, k)."""
        i_c = int(np.clip(i, 0, self.nx - 1))
        j_c = int(np.clip(j, 0, self.ny - 1))
        k_c = int(np.clip(k, 0, self.nz - 1))
        return float(self._numpy_array[k_c, j_c, i_c])

    def get_hu_at_world(self, x: float, y: float, z: float) -> float:
        """Sample Hounsfield Unit at physical world coordinates (x, y, z)."""
        i, j, k = self.world_to_index(x, y, z)
        return self.get_hu_at_voxel(i, j, k)

    def get_index_to_world_matrix(self) -> vtk.vtkMatrix4x4:
        """Return the 4x4 vtkMatrix4x4 Index-to-World transformation."""
        return self._index_to_world_matrix

    def get_reslice_matrix_for_plane(self, plane_type: str, center_world: Tuple[float, float, float]) -> vtk.vtkMatrix4x4:
        """
        Generates the standard 4x4 Reslice Matrix for vtkImageReslice corresponding to
        standard radiological viewing conventions:

        1. Axial View (Transverse / Horizontal):
           - Slicing along Z axis.
           - Display: Patient Right (R) on Screen Left, Anterior (A) on Screen Top.
           - Slice normal points towards Superior (+Z in LPS).
           - Reslice X vector: (1, 0, 0) [Right -> Left]
           - Reslice Y vector: (0, -1, 0) [Posterior -> Anterior] (flipped for Top=Anterior)
           - Reslice Z vector: (0, 0, 1) [Inferior -> Superior]

        2. Coronal View (Frontal):
           - Slicing along Y axis.
           - Display: Patient Right (R) on Screen Left, Superior (S) on Screen Top.
           - Reslice X vector: (1, 0, 0) [Right -> Left]
           - Reslice Y vector: (0, 0, 1) [Inferior -> Superior]
           - Reslice Z vector: (0, 1, 0) [Anterior -> Posterior]

        3. Sagittal View (Profile / Lateral):
           - Slicing along X axis.
           - Display: Anterior (A) on Screen Right / Left, Superior (S) on Screen Top.
           - Reslice X vector: (0, 1, 0) [Anterior -> Posterior]
           - Reslice Y vector: (0, 0, 1) [Inferior -> Superior]
           - Reslice Z vector: (1, 0, 0) [Right -> Left]
        """
        matrix = vtk.vtkMatrix4x4()
        matrix.Identity()
        cx, cy, cz = center_world

        plane = plane_type.lower()
        if plane == "axial":
            # X axis (Screen Right): +X (LPS +X is Left, so 1, 0, 0)
            matrix.SetElement(0, 0, 1.0)
            matrix.SetElement(1, 0, 0.0)
            matrix.SetElement(2, 0, 0.0)

            # Y axis (Screen Up): -Y (LPS -Y is Anterior, so top is Anterior)
            matrix.SetElement(0, 1, 0.0)
            matrix.SetElement(1, 1, -1.0)
            matrix.SetElement(2, 1, 0.0)

            # Z axis (Normal): +Z (Superior)
            matrix.SetElement(0, 2, 0.0)
            matrix.SetElement(1, 2, 0.0)
            matrix.SetElement(2, 2, 1.0)

        elif plane == "coronal":
            # X axis (Screen Right): +X
            matrix.SetElement(0, 0, 1.0)
            matrix.SetElement(1, 0, 0.0)
            matrix.SetElement(2, 0, 0.0)

            # Y axis (Screen Up): +Z (Superior on Top)
            matrix.SetElement(0, 1, 0.0)
            matrix.SetElement(1, 1, 0.0)
            matrix.SetElement(2, 1, 1.0)

            # Z axis (Normal): +Y (Posterior)
            matrix.SetElement(0, 2, 0.0)
            matrix.SetElement(1, 2, 1.0)
            matrix.SetElement(2, 2, 0.0)

        elif plane == "sagittal":
            # X axis (Screen Right): +Y (Posterior on Right, Anterior on Left)
            matrix.SetElement(0, 0, 0.0)
            matrix.SetElement(1, 0, 1.0)
            matrix.SetElement(2, 0, 0.0)

            # Y axis (Screen Up): +Z (Superior on Top)
            matrix.SetElement(0, 1, 0.0)
            matrix.SetElement(1, 1, 0.0)
            matrix.SetElement(2, 1, 1.0)

            # Z axis (Normal): +X (Left)
            matrix.SetElement(0, 2, 1.0)
            matrix.SetElement(1, 2, 0.0)
            matrix.SetElement(2, 2, 0.0)

        # Center translation column
        matrix.SetElement(0, 3, cx)
        matrix.SetElement(1, 3, cy)
        matrix.SetElement(2, 3, cz)
        matrix.SetElement(3, 3, 1.0)

        return matrix
