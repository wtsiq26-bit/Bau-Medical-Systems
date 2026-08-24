"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Module: dental/surface_extractor.py

Multi-Label Anatomical Surface Extraction Engine.
Converts NIfTI/NRRD segmentation masks into smooth, medical-grade VTK PolyData
meshes with clinically calibrated rendering materials.

Pipeline (derived from SlicerAutomatedDentalTools):
  vtkDiscreteMarchingCubes → vtkWindowedSincPolyDataFilter
  → vtkQuadricDecimation (50 %) → vtkPolyDataNormals

Engineering Notes:
- SimpleITK direction cosines are decomposed into a 3×3 rotation matrix and
  applied via vtkTransformPolyDataFilter so that the extracted surface lives
  in the same LPS physical coordinate frame as VolumeData.vtk_image_data.
- Each anatomical structure is assigned a distinct AnatomicalPreset defining
  the Marching-Cubes label, RGB color, opacity, Phong shading coefficients,
  and smoothing intensity.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk


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
# Surface Extractor
# ---------------------------------------------------------------------------

class SurfaceExtractor:
    """
    Converts integer-labeled segmentation masks into smooth VTK PolyData
    meshes with medical-grade Phong shading materials.

    Typical usage::

        extractor = SurfaceExtractor()
        mask, spacing, origin, direction = extractor.load_segmentation_file("seg.nii.gz")
        results = extractor.extract_all_structures(mask, spacing, origin, direction)
        for name, (actor, polydata) in results.items():
            volume_view.add_segmented_mesh(name, actor)
    """

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    @staticmethod
    def load_segmentation_file(
        file_path: str,
    ) -> Tuple[np.ndarray, Tuple[float, float, float], Tuple[float, float, float], Tuple[float, ...]]:
        """
        Load a segmentation mask from NIfTI (.nii / .nii.gz) or NRRD (.nrrd).

        Returns
        -------
        mask : np.ndarray
            Integer label array in (Z, Y, X) order.
        spacing : (float, float, float)
            Voxel size in mm — (sx, sy, sz) matching VTK X, Y, Z axes.
        origin : (float, float, float)
            World-space origin — (ox, oy, oz).
        direction : tuple of 9 floats
            3×3 direction cosine matrix flattened in row-major order.
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
        mask = sitk.GetArrayFromImage(sitk_img).astype(np.int32)  # (Z, Y, X)

        spacing = tuple(float(s) for s in sitk_img.GetSpacing())      # (sx, sy, sz)
        origin  = tuple(float(o) for o in sitk_img.GetOrigin())       # (ox, oy, oz)
        direction = tuple(float(d) for d in sitk_img.GetDirection())  # 9 floats

        return mask, spacing, origin, direction

    # ------------------------------------------------------------------
    # Single-Structure Extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_surface(
        mask: np.ndarray,
        spacing: Tuple[float, float, float],
        origin: Tuple[float, float, float],
        direction: Tuple[float, ...],
        preset: AnatomicalPreset,
    ) -> Tuple[vtk.vtkActor, vtk.vtkPolyData]:
        """
        Extract a single anatomical structure from a labeled mask.

        Parameters
        ----------
        mask : np.ndarray
            Integer label volume (Z, Y, X).
        spacing : (sx, sy, sz)
            Voxel spacing in mm.
        origin : (ox, oy, oz)
            Physical origin.
        direction : tuple[float, ...]
            9-element direction cosine matrix (row-major).
        preset : AnatomicalPreset
            Rendering preset for this structure.

        Returns
        -------
        actor : vtkActor
            Fully shaded actor ready for rendering.
        polydata : vtkPolyData
            Underlying triangulated surface mesh.
        """

        # ---- 1. Pack numpy mask into vtkImageData ----
        nz, ny, nx = mask.shape
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(nx, ny, nz)
        vtk_image.SetSpacing(spacing[0], spacing[1], spacing[2])
        vtk_image.SetOrigin(origin[0], origin[1], origin[2])

        # Flatten in VTK scalar order (X-fast, then Y, then Z) — identical
        # to NumPy C-contiguous (Z, Y, X) flattened view.
        flat = np.ascontiguousarray(mask).ravel()
        vtk_arr = numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_INT)
        vtk_arr.SetName("Labels")
        vtk_image.GetPointData().SetScalars(vtk_arr)

        # ---- 2. Discrete Marching Cubes (label iso-surface) ----
        marching = vtk.vtkDiscreteMarchingCubes()
        marching.SetInputData(vtk_image)
        marching.SetValue(0, float(preset.label_value))
        marching.ComputeNormalsOff()       # Normals recomputed after smoothing
        marching.ComputeGradientsOff()
        marching.Update()

        if marching.GetOutput().GetNumberOfPoints() == 0:
            # Label not found — return empty actor
            empty_pd = vtk.vtkPolyData()
            empty_actor = vtk.vtkActor()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(empty_pd)
            empty_actor.SetMapper(mapper)
            return empty_actor, empty_pd

        # ---- 3. Windowed-Sinc Smoothing ----
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(marching.GetOutputPort())
        smoother.SetNumberOfIterations(preset.smoothing_iterations)
        smoother.SetPassBand(preset.smoothing_passband)
        smoother.BoundarySmoothingOn()
        smoother.FeatureEdgeSmoothingOff()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()

        # ---- 4. Quadric Decimation (real-time 60 FPS target) ----
        decimator = vtk.vtkQuadricDecimation()
        decimator.SetInputConnection(smoother.GetOutputPort())
        decimator.SetTargetReduction(preset.decimation_target)
        decimator.VolumePreservationOn()
        decimator.Update()

        # ---- 5. Recompute Surface Normals ----
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(decimator.GetOutputPort())
        normals.SetFeatureAngle(60.0)
        normals.ConsistencyOn()
        normals.SplittingOff()
        normals.AutoOrientNormalsOn()
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.Update()

        extracted_pd: vtk.vtkPolyData = normals.GetOutput()

        # ---- 6. Apply Direction Cosine Transform (LPS alignment) ----
        # SimpleITK direction is a 3×3 matrix stored row-major.
        # If it is not the identity, apply a vtkTransform so the mesh
        # lives in the same LPS coordinate frame as VolumeData.
        dir_mat = np.array(direction[:9], dtype=np.float64).reshape(3, 3)
        is_identity = np.allclose(dir_mat, np.eye(3), atol=1e-6)

        if not is_identity:
            vtk_transform = vtk.vtkTransform()
            mat4 = vtk.vtkMatrix4x4()
            for r in range(3):
                for c in range(3):
                    mat4.SetElement(r, c, dir_mat[r, c])
            # Translation is already baked into the origin; direction
            # rotation must pivot about the origin.
            mat4.SetElement(0, 3, 0.0)
            mat4.SetElement(1, 3, 0.0)
            mat4.SetElement(2, 3, 0.0)
            vtk_transform.SetMatrix(mat4)

            # Temporarily shift origin to world-origin, rotate, shift back
            pre_translate = vtk.vtkTransform()
            pre_translate.Translate(-origin[0], -origin[1], -origin[2])

            post_translate = vtk.vtkTransform()
            post_translate.Translate(origin[0], origin[1], origin[2])

            composite = vtk.vtkTransform()
            composite.PostMultiply()
            composite.Concatenate(pre_translate)
            composite.Concatenate(vtk_transform)
            composite.Concatenate(post_translate)

            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetInputData(extracted_pd)
            transform_filter.SetTransform(composite)
            transform_filter.Update()
            extracted_pd = transform_filter.GetOutput()

        # ---- 7. Build Rendering Actor with Medical Material ----
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(extracted_pd)
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

        # Back-face culling for watertight anatomy
        prop.BackfaceCullingOn()

        return actor, extracted_pd

    # ------------------------------------------------------------------
    # Batch Extraction (all labeled structures)
    # ------------------------------------------------------------------

    @classmethod
    def extract_all_structures(
        cls,
        mask: np.ndarray,
        spacing: Tuple[float, float, float],
        origin: Tuple[float, float, float],
        direction: Tuple[float, ...],
        presets: Optional[Dict[str, AnatomicalPreset]] = None,
    ) -> Dict[str, Tuple[vtk.vtkActor, vtk.vtkPolyData]]:
        """
        Extract all anatomical structures present in *mask*.

        Only presets whose label value is actually found in the mask are
        processed.  Returns a dictionary keyed by structure id (e.g.
        ``"mandible"``, ``"teeth"``).

        Parameters
        ----------
        mask : np.ndarray
            Integer label volume (Z, Y, X).
        spacing, origin, direction
            Physical coordinate parameters from ``load_segmentation_file``.
        presets : dict, optional
            Override default ``STRUCTURE_PRESETS``.

        Returns
        -------
        dict[str, (vtkActor, vtkPolyData)]
        """
        if presets is None:
            presets = STRUCTURE_PRESETS

        unique_labels = set(np.unique(mask).tolist())
        results: Dict[str, Tuple[vtk.vtkActor, vtk.vtkPolyData]] = {}

        for struct_id, preset in presets.items():
            if preset.label_value not in unique_labels:
                continue
            actor, polydata = cls.extract_surface(mask, spacing, origin, direction, preset)
            if polydata.GetNumberOfPoints() > 0:
                results[struct_id] = (actor, polydata)

        return results
