"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Module: core/async_workers.py

Asynchronous Background Worker Threads (QThread).
Prevents Qt Main GUI Thread freezing during heavy computational geometry algorithms:
1. SegmentationWorker: Multi-label iso-surface extraction (Marching Cubes, smoothing, decimation).
2. ICPRegistrationWorker: Rigid ICP registration for intraoral optical scans.
3. PanoramicWorker: Vectorized focal trough unrolling & curved MPR reconstruction.

Strict Thread-Safety Rules:
- Workers execute purely on CPU data structures (vtkPolyData, np.ndarray, matrices).
- Workers NEVER create or mutate vtkActor, vtkRenderer, or QWidget objects.
- All extracted meshes and results are emitted via Qt Signals to the Main GUI Thread.
"""

from __future__ import annotations

import traceback
from typing import Optional, Dict, Tuple, Any

import numpy as np
import vtk
from PySide6.QtCore import QThread, Signal, QObject

from core.volume_data import VolumeData
from dental.surface_extractor import SurfaceExtractor, AnatomicalPreset, STRUCTURE_PRESETS
from dental.mesh_registration import MeshRegistrationEngine
from dental.panoramic_mpr import DentalArchCurve, PanoramicGenerator


# ---------------------------------------------------------------------------
# 1. Segmentation Extraction Worker Thread
# ---------------------------------------------------------------------------

class SegmentationWorker(QThread):
    """
    Asynchronous Worker for loading segmentation masks (.nii / .nrrd) and
    extracting smoothed, decimated 3D anatomical surface meshes.
    """

    progress_updated = Signal(int, str)              # percentage (0-100), status message
    structure_extracted = Signal(str, object)        # struct_id, vtkPolyData (streamed per structure)
    finished_all = Signal(dict)                      # {struct_id: vtkPolyData}
    error_occurred = Signal(str)                     # error message

    def __init__(
        self,
        file_path: str,
        reference_volume: Optional[VolumeData] = None,
        presets: Optional[Dict[str, AnatomicalPreset]] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.file_path = file_path
        self.reference_volume = reference_volume
        self.presets = presets if presets is not None else STRUCTURE_PRESETS
        self._is_cancelled = False

    def cancel(self) -> None:
        """Requests cancellation of the extraction worker."""
        self._is_cancelled = True

    def run(self) -> None:
        """Executes the extraction pipeline off the GUI thread."""
        try:
            self.progress_updated.emit(5, "Reading segmentation file from disk...")

            if self._is_cancelled:
                return

            extractor = SurfaceExtractor()
            mask, spacing, origin, direction = extractor.load_segmentation_file(
                self.file_path, reference_volume=self.reference_volume
            )

            if self._is_cancelled:
                return

            self.progress_updated.emit(20, "Analyzing labeled anatomical structures...")

            unique_labels = set(np.unique(mask).tolist())
            matching_presets = {
                sid: preset for sid, preset in self.presets.items()
                if preset.label_value in unique_labels
            }

            if not matching_presets:
                self.progress_updated.emit(100, "No matching anatomical labels found in mask.")
                self.finished_all.emit({})
                return

            total_structures = len(matching_presets)
            extracted_dict: Dict[str, vtk.vtkPolyData] = {}

            for idx, (struct_id, preset) in enumerate(matching_presets.items(), start=1):
                if self._is_cancelled:
                    return

                # Calculate scaled progress from 25% to 95%
                pct_start = int(25 + (idx - 1) * (70 / total_structures))
                self.progress_updated.emit(
                    pct_start,
                    f"Extracting {preset.name} (Marching Cubes → Smoothing → Decimation)..."
                )

                polydata = extractor.extract_surface_polydata(
                    mask, spacing, origin, direction, preset
                )

                if self._is_cancelled:
                    return

                if polydata.GetNumberOfPoints() > 0:
                    if self.reference_volume is not None:
                        extractor.validate_spatial_bounds(polydata, self.reference_volume)
                    extracted_dict[struct_id] = polydata
                    self.structure_extracted.emit(struct_id, polydata)

                pct_end = int(25 + idx * (70 / total_structures))
                self.progress_updated.emit(
                    pct_end,
                    f"Extracted {preset.name}: {polydata.GetNumberOfPoints():,} vertices."
                )

            self.progress_updated.emit(100, f"Extraction complete: {len(extracted_dict)} structures ready.")
            self.finished_all.emit(extracted_dict)

        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {str(exc)}\n\n{traceback.format_exc()}"
            self.error_occurred.emit(err_msg)


# ---------------------------------------------------------------------------
# 2. ICP Registration Worker Thread
# ---------------------------------------------------------------------------

class ICPRegistrationWorker(QThread):
    """
    Asynchronous Worker for Robust Two-Stage Rigid Surface Registration:
      Stage 1: Global PCA Pre-Alignment & 4-Quadrant ambiguity search.
      Stage 2: Fine Trimmed ICP (vtkIterativeClosestPointTransform + 15% Outlier Rejection).
    """

    progress_updated = Signal(int, str)                                       # percentage (0-100), message
    registration_complete = Signal(object, object, float, int, float, str)    # aligned_poly, matrix, rms, iters, max_95th, status
    error_occurred = Signal(str)                                              # error message

    def __init__(
        self,
        source_poly: vtk.vtkPolyData,
        target_poly: vtk.vtkPolyData,
        max_iterations: int = 200,
        max_landmarks: int = 2500,
        tolerance: float = 0.001,
        trim_ratio: float = 0.15,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        # Deep copy to ensure thread isolation
        self.source_poly = vtk.vtkPolyData()
        self.source_poly.DeepCopy(source_poly)
        self.target_poly = vtk.vtkPolyData()
        self.target_poly.DeepCopy(target_poly)
        self.max_iterations = max_iterations
        self.max_landmarks = max_landmarks
        self.tolerance = tolerance
        self.trim_ratio = trim_ratio
        self._is_cancelled = False

    def cancel(self) -> None:
        """Requests cancellation of the registration worker."""
        self._is_cancelled = True

    def run(self) -> None:
        """Executes the two-stage registration pipeline off the GUI thread."""
        try:
            self.progress_updated.emit(10, "Stage 1: Computing Principal Component Analysis (PCA) & Centroids...")

            if self._is_cancelled:
                return

            self.progress_updated.emit(25, "Stage 1: Evaluating 4-Quadrant rotation candidates (Chamfer metric)...")

            if self._is_cancelled:
                return

            self.progress_updated.emit(50, f"Stage 2: Running Fine Rigid ICP ({self.max_iterations} iters, {self.max_landmarks} landmarks)...")

            engine = MeshRegistrationEngine()
            aligned_pd, transform_np, rms_error, num_iters, max_95, quality_status = engine.register_icp_transform(
                source=self.source_poly,
                target=self.target_poly,
                max_iterations=self.max_iterations,
                max_landmarks=self.max_landmarks,
                tolerance=self.tolerance,
                trim_ratio=self.trim_ratio,
            )

            if self._is_cancelled:
                return

            self.progress_updated.emit(85, "Stage 2: Applying 15% distance trimming (gingival outlier rejection)...")
            self.progress_updated.emit(95, f"Quality evaluation: {quality_status} (RMS: {rms_error:.4f} mm, 95%: {max_95:.4f} mm)...")
            self.progress_updated.emit(100, f"Registration complete ({quality_status}).")

            self.registration_complete.emit(
                aligned_pd, transform_np, rms_error, num_iters, max_95, quality_status
            )

        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {str(exc)}\n\n{traceback.format_exc()}"
            self.error_occurred.emit(err_msg)


# ---------------------------------------------------------------------------
# 3. Panoramic MPR Worker Thread
# ---------------------------------------------------------------------------

class PanoramicWorker(QThread):
    """
    Asynchronous Worker for 3D Curve Parameterization and Focal Trough
    Panoramic Volume Unrolling using trilinear spline interpolation.
    """

    progress_updated = Signal(int, str)              # percentage (0-100), message
    panoramic_ready = Signal(object)                 # np.ndarray 2D HU image (Height_Z, ArcLength_S)
    error_occurred = Signal(str)                     # error message

    def __init__(
        self,
        volume: VolumeData,
        arch_curve: DentalArchCurve,
        focal_trough_thickness_mm: float = 8.0,
        projection_mode: str = "mip",
        vertical_step_mm: float = 0.5,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.volume = volume
        self.arch_curve = arch_curve
        self.focal_trough_thickness_mm = focal_trough_thickness_mm
        self.projection_mode = projection_mode
        self.vertical_step_mm = vertical_step_mm
        self._is_cancelled = False

    def cancel(self) -> None:
        """Requests cancellation of the panoramic worker."""
        self._is_cancelled = True

    def run(self) -> None:
        """Executes the curved focal trough unrolling off the GUI thread."""
        try:
            if len(self.arch_curve.sampled_points) < 2:
                self.error_occurred.emit("Dental arch curve must contain at least 2 points.")
                return

            self.progress_updated.emit(10, "Building 3D physical sampling grid along dental arch...")

            generator = PanoramicGenerator()
            generator.focal_trough_thickness_mm = self.focal_trough_thickness_mm
            generator.projection_mode = self.projection_mode
            generator.vertical_step_mm = self.vertical_step_mm

            if self._is_cancelled:
                return

            self.progress_updated.emit(40, "Unrolling 3D volume along Darboux normal frames (map_coordinates)...")

            panoramic_img = generator.generate_panoramic_image(self.volume, self.arch_curve)

            if self._is_cancelled:
                return

            if panoramic_img is None:
                self.error_occurred.emit("Panoramic generation returned empty image.")
                return

            self.progress_updated.emit(100, "Panoramic radiograph synthesis complete.")
            self.panoramic_ready.emit(panoramic_img)

        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {str(exc)}\n\n{traceback.format_exc()}"
            self.error_occurred.emit(err_msg)


# ---------------------------------------------------------------------------
# 4. Surgical Guide Generator Worker Thread
# ---------------------------------------------------------------------------

class SurgicalGuideWorker(QThread):
    """
    Asynchronous Worker for 3D Printable Surgical Guide CAD Generation:
    - Base shell offset and solid manifold closure.
    - Parametric drill sleeve housing extrusion.
    - Boolean CSG drill channel subtraction and irrigation slot carving.
    - Mesh decimation and SLA/DLP 3D print validation.
    """

    progress_updated = Signal(int, str)              # percentage (0-100), message
    guide_generated = Signal(object, float, float)   # guide_polydata, volume_cm3, area_cm2
    error_occurred = Signal(str)                     # error message

    def __init__(
        self,
        base_surface: vtk.vtkPolyData,
        implants: List[Any],
        guide_thickness_mm: float = 3.0,
        sleeve_clearance_mm: float = 1.2,
        sleeve_outer_wall_mm: float = 2.0,
        sleeve_height_mm: float = 6.0,
        sleeve_offset_mm: float = 2.0,
        include_irrigation_windows: bool = True,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.base_surface = vtk.vtkPolyData()
        self.base_surface.DeepCopy(base_surface)
        self.implants = implants
        self.guide_thickness_mm = guide_thickness_mm
        self.sleeve_clearance_mm = sleeve_clearance_mm
        self.sleeve_outer_wall_mm = sleeve_outer_wall_mm
        self.sleeve_height_mm = sleeve_height_mm
        self.sleeve_offset_mm = sleeve_offset_mm
        self.include_irrigation_windows = include_irrigation_windows
        self._is_cancelled = False

    def cancel(self) -> None:
        """Requests cancellation of the guide generation worker."""
        self._is_cancelled = True

    def run(self) -> None:
        """Executes the CAD guide synthesis pipeline off the GUI thread."""
        try:
            from dental.surgical_guide import SurgicalGuideGenerator

            self.progress_updated.emit(10, "Generating watertight anatomical guide base shell...")

            if self._is_cancelled:
                return

            self.progress_updated.emit(35, f"Constructing {len(self.implants)} parametric metallic sleeve housings...")

            if self._is_cancelled:
                return

            self.progress_updated.emit(60, "Executing Boolean CSG drill channel subtractions & irrigation windows...")

            result = SurgicalGuideGenerator.generate_guide(
                base_surface=self.base_surface,
                implants=self.implants,
                guide_thickness_mm=self.guide_thickness_mm,
                sleeve_clearance_mm=self.sleeve_clearance_mm,
                sleeve_outer_wall_mm=self.sleeve_outer_wall_mm,
                sleeve_height_mm=self.sleeve_height_mm,
                sleeve_offset_mm=self.sleeve_offset_mm,
                include_irrigation_windows=self.include_irrigation_windows,
            )

            if self._is_cancelled:
                return

            self.progress_updated.emit(90, "Smoothing guide borders and validating SLA/DLP 3D print manifoldness...")
            self.progress_updated.emit(100, f"Surgical guide ready (Resin volume: {result.volume_cm3:.2f} cm³).")

            self.guide_generated.emit(result.guide_polydata, result.volume_cm3, result.surface_area_cm2)

        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {str(exc)}\n\n{traceback.format_exc()}"
            self.error_occurred.emit(err_msg)
