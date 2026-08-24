"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Module: dental/surface_extractor.py

Multi-Label Anatomical Surface Extraction & Coordinate Alignment Engine.
Converts NIfTI/NRRD segmentation masks into smooth, medical-grade VTK PolyData
meshes rigorously mapped into the Patient World LPS (Left-Posterior-Superior)
Coordinate Space.

Mathematical Formulation:
-------------------------
A discrete segmentation mask has voxel index coordinates:
    p_voxel = [i, j, k]^T  where i in [0, Nx-1], j in [0, Ny-1], k in [0, Nz-1]

The full 4x4 Homogeneous Affine Transformation Matrix T_LPS transforming
continuous voxel index coordinates to Patient World LPS (x, y, z) is:

    T_LPS = [ D00*sx   D01*sy   D02*sz   ox ]
            [ D10*sx   D11*sy   D12*sz   oy ]
            [ D20*sx   D21*sy   D22*sz   oz ]
            [    0        0        0      1 ]

where:
  - D is the 3x3 Direction Cosines matrix (row-major order from SimpleITK/DICOM).
  - s = (sx, sy, sz) is the physical voxel spacing in mm.
  - o = (ox, oy, oz) is the physical origin in mm.

Pipeline:
---------
  1. Discrete Voxel Iso-Surface: vtkDiscreteMarchingCubes on unit-scaled vtkImageData.
  2. Windowed-Sinc Smoothing: vtkWindowedSincPolyDataFilter.
  3. Quadric Decimation: vtkQuadricDecimation (50% target reduction).
  4. Physical LPS Affine Transform: vtkTransformPolyDataFilter applying T_LPS.
  5. Consistent Outward Normals: vtkPolyDataNormals computed AFTER affine mapping
     to guarantee correct face normal orientation under all matrix determinants.
  6. Spatial Bounding Box & Centroid Validation against reference VolumeData.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from core.volume_data import VolumeData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Coordinate Alignment Exception
# ---------------------------------------------------------------------------

class CoordinateAlignmentError(ValueError):
    """Raised when an extracted anatomical mesh is spatially incongruent with reference volume."""
    pass


# ---------------------------------------------------------------------------
# Anatomical Preset Definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnatomicalPreset:
    """Defines rendering parameters for a single segmented anatomical structure."""

    name: str
    label_value: int
    color: Tuple[float, float, float]          # RGB  [0..1]
    opacity: float                              # Alpha [0..1]
    ambient: float       = 0.20
    diffuse: float       = 0.80
    specular: float      = 0.40
    specular_power: float = 30.0
    smoothing_iterations: int   = 25
    smoothing_passband: float   = 0.05
    decimation_target: float    = 0.50          # 50 % triangle reduction


# ---------------------------------------------------------------------------
# Clinically-Calibrated Structure Presets
# ---------------------------------------------------------------------------

MANDIBLE_BONE = AnatomicalPreset(
    name="Mandible Bone",
    label_value=1,
    color=(0.92, 0.87, 0.78),
    opacity=0.85,
    ambient=0.22,
    diffuse=0.78,
    specular=0.35,
    specular_power=25.0,
    smoothing_iterations=30,
    smoothing_passband=0.04,
    decimation_target=0.50,
)

MANDIBULAR_CANAL = AnatomicalPreset(
    name="Mandibular Canal (IAN)",
    label_value=2,
    color=(1.00, 0.25, 0.25),
    opacity=0.70,
    ambient=0.35,
    diffuse=0.65,
    specular=0.20,
    specular_power=15.0,
    smoothing_iterations=20,
    smoothing_passband=0.06,
    decimation_target=0.40,
)

TEETH_ENAMEL = AnatomicalPreset(
    name="Teeth / Enamel Crowns",
    label_value=3,
    color=(0.97, 0.98, 1.00),
    opacity=0.95,
    ambient=0.18,
    diffuse=0.72,
    specular=0.60,
    specular_power=50.0,
    smoothing_iterations=20,
    smoothing_passband=0.05,
    decimation_target=0.50,
)

SOFT_TISSUE = AnatomicalPreset(
    name="Soft Tissue",
    label_value=4,
    color=(0.85, 0.65, 0.55),
    opacity=0.40,
    ambient=0.25,
    diffuse=0.75,
    specular=0.15,
    specular_power=10.0,
    smoothing_iterations=35,
    smoothing_passband=0.03,
    decimation_target=0.55,
)

STRUCTURE_PRESETS: Dict[str, AnatomicalPreset] = {
    "mandible":  MANDIBLE_BONE,
    "canal":     MANDIBULAR_CANAL,
    "teeth":     TEETH_ENAMEL,
    "soft":      SOFT_TISSUE,
}


# ---------------------------------------------------------------------------
# Surface Extractor Engine
# ---------------------------------------------------------------------------

class SurfaceExtractor:
    """
    High-precision anatomical surface extraction engine converting segmentation
    masks into smooth VTK PolyData meshes aligned with LPS World Coordinates.
    """

    # ------------------------------------------------------------------
    # File I/O & Resampling
    # ------------------------------------------------------------------

    @staticmethod
    def load_segmentation_file(
        file_path: str,
        reference_volume: Optional[VolumeData] = None,
    ) -> Tuple[np.ndarray, Tuple[float, float, float], Tuple[float, float, float], Tuple[float, ...]]:
        """
        Load a segmentation mask from NIfTI (.nii / .nii.gz) or NRRD (.nrrd).
        Optionally resamples the mask onto a reference VolumeData grid if provided.

        Returns
        -------
        mask : np.ndarray
            Integer label array in (Z, Y, X) order, C-contiguous.
        spacing : (float, float, float)
            Voxel size in mm — (sx, sy, sz).
        origin : (float, float, float)
            World-space origin — (ox, oy, oz).
        direction : tuple of 9 floats
            3x3 direction cosine matrix flattened in row-major order.
        """
        try:
            import SimpleITK as sitk
        except ImportError as exc:
            raise ImportError(
                "SimpleITK is required for loading segmentation masks. "
                "Install via:  pip install SimpleITK"
            ) from exc

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Segmentation file not found: {file_path}")

        sitk_img = sitk.ReadImage(file_path)

        # Optional reference grid resampling
        if reference_volume is not None:
            # Check if resampling is needed
            ref_size = (reference_volume.nx, reference_volume.ny, reference_volume.nz)
            ref_spacing = reference_volume.spacing
            ref_origin = reference_volume.origin
            ref_direction = reference_volume.direction

            needs_resampling = (
                sitk_img.GetSize() != ref_size
                or not np.allclose(sitk_img.GetSpacing(), ref_spacing, atol=1e-4)
                or not np.allclose(sitk_img.GetOrigin(), ref_origin, atol=1e-3)
                or not np.allclose(sitk_img.GetDirection(), ref_direction, atol=1e-4)
            )

            if needs_resampling:
                logger.info("Resampling segmentation mask to match reference VolumeData grid...")
                sitk_img = SurfaceExtractor.resample_mask_to_reference(sitk_img, reference_volume)

        mask = sitk.GetArrayFromImage(sitk_img).astype(np.int32)
        mask = np.ascontiguousarray(mask)

        spacing = tuple(float(s) for s in sitk_img.GetSpacing())
        origin  = tuple(float(o) for o in sitk_img.GetOrigin())
        direction = tuple(float(d) for d in sitk_img.GetDirection())

        return mask, spacing, origin, direction

    @staticmethod
    def resample_mask_to_reference(
        mask_image: Any,
        reference_volume: VolumeData,
    ) -> Any:
        """
        Resamples a SimpleITK segmentation mask image onto the exact physical grid
        of the reference VolumeData using nearest-neighbor interpolation to preserve
        integer label boundaries.
        """
        import SimpleITK as sitk

        resample = sitk.ResampleImageFilter()
        resample.SetSize([reference_volume.nx, reference_volume.ny, reference_volume.nz])
        resample.SetOutputSpacing(list(reference_volume.spacing))
        resample.SetOutputOrigin(list(reference_volume.origin))
        resample.SetOutputDirection(list(reference_volume.direction))
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
        resample.SetDefaultPixelValue(0)
        resample.SetOutputPixelType(sitk.sitkInt32)

        return resample.Execute(mask_image)

    # ------------------------------------------------------------------
    # Affine Matrix Construction
    # ------------------------------------------------------------------

    @staticmethod
    def construct_index_to_lps_matrix(
        spacing: Tuple[float, float, float],
        origin: Tuple[float, float, float],
        direction: Tuple[float, ...],
    ) -> vtk.vtkMatrix4x4:
        """
        Constructs the exact 4x4 Homogeneous Affine Transformation Matrix T_LPS
        mapping continuous voxel indices [i, j, k, 1]^T to Patient LPS World [x, y, z, 1]^T:

            T_LPS = [ D00*sx   D01*sy   D02*sz   ox ]
                    [ D10*sx   D11*sy   D12*sz   oy ]
                    [ D20*sx   D21*sy   D22*sz   oz ]
                    [    0        0        0      1 ]

        Parameters
        ----------
        spacing : (sx, sy, sz)
            Voxel spacing in mm.
        origin : (ox, oy, oz)
            Physical origin in mm.
        direction : 9 floats
            3x3 direction matrix in row-major order: (D00, D01, D02, D10, D11, D12, D20, D21, D22).

        Returns
        -------
        vtk.vtkMatrix4x4
        """
        mat = vtk.vtkMatrix4x4()
        sx, sy, sz = spacing
        ox, oy, oz = origin
        d = direction if (direction is not None and len(direction) == 9) else (1,0,0, 0,1,0, 0,0,1)

        # Column 0: X vector scaled by sx
        mat.SetElement(0, 0, d[0] * sx)
        mat.SetElement(1, 0, d[3] * sx)
        mat.SetElement(2, 0, d[6] * sx)
        mat.SetElement(3, 0, 0.0)

        # Column 1: Y vector scaled by sy
        mat.SetElement(0, 1, d[1] * sy)
        mat.SetElement(1, 1, d[4] * sy)
        mat.SetElement(2, 1, d[7] * sy)
        mat.SetElement(3, 0, 0.0)

        # Column 2: Z vector scaled by sz
        mat.SetElement(0, 2, d[2] * sz)
        mat.SetElement(1, 2, d[5] * sz)
        mat.SetElement(2, 2, d[8] * sz)
        mat.SetElement(3, 0, 0.0)

        # Column 3: Translation Origin
        mat.SetElement(0, 3, ox)
        mat.SetElement(1, 3, oy)
        mat.SetElement(2, 3, oz)
        mat.SetElement(3, 3, 1.0)

        return mat

    # ------------------------------------------------------------------
    # Pure PolyData Surface Extraction (Worker-Thread Safe)
    # ------------------------------------------------------------------

    @staticmethod
    def extract_surface_polydata(
        mask: np.ndarray,
        spacing: Tuple[float, float, float],
        origin: Tuple[float, float, float],
        direction: Tuple[float, ...],
        preset: AnatomicalPreset,
    ) -> vtk.vtkPolyData:
        """
        Extract a single anatomical structure from a labeled mask as a pure vtkPolyData
        in Patient World LPS coordinates.

        Thread-safe: creates NO vtkActor, vtkRenderer, or OpenGL objects.

        Parameters
        ----------
        mask : np.ndarray
            Integer label volume (Z, Y, X), C-contiguous.
        spacing : (sx, sy, sz)
            Voxel spacing in mm.
        origin : (ox, oy, oz)
            Physical origin in mm.
        direction : tuple of 9 floats
            3x3 direction cosine matrix in row-major order.
        preset : AnatomicalPreset
            Rendering and smoothing preset for this structure.

        Returns
        -------
        vtk.vtkPolyData
            Triangulated, smoothed, decimated, and LPS-transformed surface mesh.
        """
        # ---- 1. Build Unit-Scaled Voxel-Index vtkImageData ----
        # Using unit spacing (1,1,1) and zero origin (0,0,0) ensures Marching Cubes
        # operates in pure continuous voxel index space [0, Nx-1] x [0, Ny-1] x [0, Nz-1].
        nz, ny, nx = mask.shape
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(nx, ny, nz)
        vtk_image.SetSpacing(1.0, 1.0, 1.0)
        vtk_image.SetOrigin(0.0, 0.0, 0.0)

        flat = np.ascontiguousarray(mask, dtype=np.int32).ravel()
        vtk_arr = numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_INT)
        vtk_arr.SetName("Labels")
        vtk_image.GetPointData().SetScalars(vtk_arr)

        # ---- 2. Discrete Marching Cubes (Voxel-Space Iso-Surface) ----
        marching = vtk.vtkDiscreteMarchingCubes()
        marching.SetInputData(vtk_image)
        marching.SetValue(0, float(preset.label_value))
        marching.ComputeNormalsOff()
        marching.ComputeGradientsOff()
        marching.Update()

        if marching.GetOutput().GetNumberOfPoints() == 0:
            return vtk.vtkPolyData()

        # ---- 3. Windowed-Sinc Smoothing in Voxel Space ----
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(marching.GetOutputPort())
        smoother.SetNumberOfIterations(preset.smoothing_iterations)
        smoother.SetPassBand(preset.smoothing_passband)
        smoother.BoundarySmoothingOn()
        smoother.FeatureEdgeSmoothingOff()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()

        # ---- 4. Quadric Decimation (50% Triangle Reduction) ----
        decimator = vtk.vtkQuadricDecimation()
        decimator.SetInputConnection(smoother.GetOutputPort())
        decimator.SetTargetReduction(preset.decimation_target)
        decimator.VolumePreservationOn()
        decimator.Update()

        # ---- 5. Physical LPS Affine Transformation (Rigorous Spatial Mapping) ----
        # Applies full homogeneous 4x4 matrix T_LPS:
        # p_world = D * (s * p_voxel) + origin
        t_lps = SurfaceExtractor.construct_index_to_lps_matrix(spacing, origin, direction)
        vtk_transform = vtk.vtkTransform()
        vtk_transform.SetMatrix(t_lps)

        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputConnection(decimator.GetOutputPort())
        transform_filter.SetTransform(vtk_transform)
        transform_filter.Update()

        # ---- 6. Recompute Consistent Surface Normals in World LPS Space ----
        # Normals must be computed AFTER the affine transform to guarantee outward
        # pointing normals under arbitrary direction cosine determinants (e.g. reflections).
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(transform_filter.GetOutputPort())
        normals.SetFeatureAngle(60.0)
        normals.ConsistencyOn()
        normals.SplittingOff()
        normals.AutoOrientNormalsOn()
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.Update()

        final_pd = vtk.vtkPolyData()
        final_pd.DeepCopy(normals.GetOutput())
        return final_pd

    # ------------------------------------------------------------------
    # Spatial Bounding Box & Centroid Validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_spatial_bounds(
        mesh_polydata: vtk.vtkPolyData,
        reference_volume: VolumeData,
        max_allowable_drift_mm: float = 150.0,
    ) -> bool:
        """
        Validates that the extracted mesh spatial bounding box and centroid are physically
        congruent with the active reference VolumeData.

        Raises
        ------
        CoordinateAlignmentError
            If the mesh centroid drifts beyond the allowable threshold from the volume.
        """
        if mesh_polydata.GetNumberOfPoints() == 0:
            return True

        # Mesh bounding box & centroid
        mb = mesh_polydata.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
        mesh_center = np.array([
            (mb[0] + mb[1]) * 0.5,
            (mb[2] + mb[3]) * 0.5,
            (mb[4] + mb[5]) * 0.5,
        ])

        # Reference volume bounds & centroid
        vol_center = np.array(reference_volume.get_center())
        vol_bounds = reference_volume.get_bounds()
        vol_diag = np.sqrt(
            (vol_bounds[1] - vol_bounds[0]) ** 2 +
            (vol_bounds[3] - vol_bounds[2]) ** 2 +
            (vol_bounds[5] - vol_bounds[4]) ** 2
        )

        centroid_dist = float(np.linalg.norm(mesh_center - vol_center))
        allowed_dist = max(max_allowable_drift_mm, vol_diag * 0.75)

        if centroid_dist > allowed_dist:
            err_msg = (
                f"Spatial Misalignment Detected: Mesh centroid ({mesh_center[0]:.1f}, {mesh_center[1]:.1f}, {mesh_center[2]:.1f}) mm "
                f"is drifted {centroid_dist:.1f} mm away from VolumeData center ({vol_center[0]:.1f}, {vol_center[1]:.1f}, {vol_center[2]:.1f}) mm. "
                f"Allowed threshold is {allowed_dist:.1f} mm. Check NIfTI/NRRD Direction Matrix metadata."
            )
            logger.error(err_msg)
            raise CoordinateAlignmentError(err_msg)

        return True

    # ------------------------------------------------------------------
    # Batch Extraction
    # ------------------------------------------------------------------

    @classmethod
    def extract_all_structures_polydata(
        cls,
        mask: np.ndarray,
        spacing: Tuple[float, float, float],
        origin: Tuple[float, float, float],
        direction: Tuple[float, ...],
        presets: Optional[Dict[str, AnatomicalPreset]] = None,
        reference_volume: Optional[VolumeData] = None,
    ) -> Dict[str, vtk.vtkPolyData]:
        """
        Extract all anatomical structures present in *mask* as pure vtkPolyData meshes.
        Thread-safe for background worker execution.
        """
        if presets is None:
            presets = STRUCTURE_PRESETS

        unique_labels = set(np.unique(mask).tolist())
        results: Dict[str, vtk.vtkPolyData] = {}

        for struct_id, preset in presets.items():
            if preset.label_value not in unique_labels:
                continue

            polydata = cls.extract_surface_polydata(mask, spacing, origin, direction, preset)
            if polydata.GetNumberOfPoints() > 0:
                if reference_volume is not None:
                    cls.validate_spatial_bounds(polydata, reference_volume)
                results[struct_id] = polydata

        return results

    # ------------------------------------------------------------------
    # Actor Creation (Must execute on Main GUI Thread)
    # ------------------------------------------------------------------

    @staticmethod
    def create_structure_actor(
        polydata: vtk.vtkPolyData,
        preset: AnatomicalPreset,
    ) -> vtk.vtkActor:
        """
        Constructs a fully configured vtkActor for *polydata* using *preset*.
        Must be called on the Main GUI Thread.
        """
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        prop = actor.GetProperty()
        prop.SetColor(*preset.color)
        prop.SetOpacity(preset.opacity)
        prop.SetAmbient(preset.ambient)
        prop.SetDiffuse(preset.diffuse)
        prop.SetSpecular(preset.specular)
        prop.SetSpecularPower(preset.specular_power)
        prop.SetInterpolationToPhong()
        prop.BackfaceCullingOn()

        return actor

    # ------------------------------------------------------------------
    # Legacy / Synchronous Convenience Wrappers
    # ------------------------------------------------------------------

    @classmethod
    def extract_surface(
        cls,
        mask: np.ndarray,
        spacing: Tuple[float, float, float],
        origin: Tuple[float, float, float],
        direction: Tuple[float, ...],
        preset: AnatomicalPreset,
        reference_volume: Optional[VolumeData] = None,
    ) -> Tuple[vtk.vtkActor, vtk.vtkPolyData]:
        """Synchronous wrapper extracting both actor and polydata."""
        polydata = cls.extract_surface_polydata(mask, spacing, origin, direction, preset)
        if reference_volume is not None:
            cls.validate_spatial_bounds(polydata, reference_volume)
        actor = cls.create_structure_actor(polydata, preset)
        return actor, polydata

    @classmethod
    def extract_all_structures(
        cls,
        mask: np.ndarray,
        spacing: Tuple[float, float, float],
        origin: Tuple[float, float, float],
        direction: Tuple[float, ...],
        presets: Optional[Dict[str, AnatomicalPreset]] = None,
        reference_volume: Optional[VolumeData] = None,
    ) -> Dict[str, Tuple[vtk.vtkActor, vtk.vtkPolyData]]:
        """Synchronous wrapper extracting all structures as (actor, polydata) tuples."""
        if presets is None:
            presets = STRUCTURE_PRESETS

        poly_results = cls.extract_all_structures_polydata(
            mask, spacing, origin, direction, presets, reference_volume
        )
        results: Dict[str, Tuple[vtk.vtkActor, vtk.vtkPolyData]] = {}
        for sid, pd in poly_results.items():
            preset = presets[sid]
            actor = cls.create_structure_actor(pd, preset)
            results[sid] = (actor, pd)

        return results
