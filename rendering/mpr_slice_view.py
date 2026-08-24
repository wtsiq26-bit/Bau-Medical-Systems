"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Module: rendering/mpr_slice_view.py

High-performance 2D Multi-Planar Reconstruction (MPR) slice viewer with Complete 4-Corner PACS OSD HUD.
Features:
- Complete 4-Corner PACS HUD:
  * Top-Left: Matrix resolution, Zoom level, Slice position.
  * Top-Right: Patient Name, ID, Study Date, Modality (+C).
  * Bottom-Left: Voxel coordinates [i, j, k], Window/Level (W/L), Real-time HU with tissue classification.
  * Bottom-Right: Physical slice thickness & spacing (mm), Field of View (FoV).
  * Center Edges: Bold anatomical markers (R, L, A, P, S, I).
- Hardware-accelerated 2D re-slicing via vtkImageReslice and vtkImageMapToWindowLevelColors.
- Interactive Cine Player (Automated continuous slice navigation loop).
- Invert Grayscale (Negative film mode).
- 2D Distance Caliper and 3-Point Angle Measurement.
- 2D Mandibular Nerve Canal cross-section disc rendering.
- Interactive Mandibular Nerve Drawing mode (click to place 3D seed points).
"""

from __future__ import annotations
from typing import Optional, Tuple, Callable, List
import math
import numpy as np
import vtk
from PySide6.QtCore import QObject, Signal, Qt, QTimer
from PySide6.QtWidgets import QWidget, QVBoxLayout, QFrame

try:
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
except ImportError:
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from core.volume_data import VolumeData
from dental.nerve_tracer import NerveTracer, NerveChannel
from dental.panoramic_mpr import DentalArchCurve, CrossSectionManager


class MPRSliceSignals(QObject):
    """Qt Signals for asynchronous MPR view communication."""
    crosshair_moved = Signal(float, float, float)     # World (x, y, z) in mm
    slice_changed = Signal(str, int, int)             # plane_type, slice_index, max_slices
    window_level_changed = Signal(float, float)       # window_width, window_level
    hu_inspected = Signal(float, float, float, float) # x, y, z, HU value
    measurement_completed = Signal(str, float)        # type, value (mm or deg)
    nerve_point_placed = Signal(float, float, float)  # x, y, z
    arch_point_placed = Signal(float, float, float)   # x, y, z
    focused = Signal(str)                             # plane_type


class MPRSliceView(QFrame):
    """
    Commercial Medical PACS 2D MPR Slice Viewport with Complete 4-Corner OSD HUD.
    """

    def __init__(
        self,
        plane_type: str = "axial",
        parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.plane_type: str = plane_type.lower()
        if self.plane_type not in ("axial", "coronal", "sagittal"):
            raise ValueError(f"Invalid plane_type: {self.plane_type}. Must be 'axial', 'coronal', or 'sagittal'.")

        self.signals = MPRSliceSignals()
        self.volume_data: Optional[VolumeData] = None
        self.nerve_tracer: Optional[NerveTracer] = None
        self.arch_curve: Optional[DentalArchCurve] = None
        self.cross_section_mgr: Optional[CrossSectionManager] = None

        # State
        self._current_world_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._current_voxel_idx: Tuple[int, int, int] = (0, 0, 0)
        self._window_width: float = 2500.0
        self._window_level: float = 500.0
        self._is_crosshair_visible: bool = True
        self._is_measuring: bool = False
        self._active_tool: str = "select"  # 'select' | 'pan' | 'zoom' | 'wl' | 'distance' | 'angle'
        self._is_nerve_drawing: bool = False
        self._is_arch_drawing: bool = False
        self._is_inverted: bool = False
        self._measurement_points: list[Tuple[float, float, float]] = []

        # Cine Player Timer
        self._cine_timer = QTimer(self)
        self._cine_timer.timeout.connect(lambda: self._step_slice(step=1))

        # Color scheme matching Bau Medical Systems
        self._accent_color = (0.0, 0.86, 0.91)       # Electric Cyan #00dbe9
        self._secondary_color = (0.72, 0.78, 0.89)   # Platinum Ice
        self._left_nerve_color = (1.0, 0.176, 0.333) # Surgical Neon Pink (#ff2d55)
        self._right_nerve_color = (1.0, 0.584, 0.0)  # Neon Orange (#ff9500)
        self._arch_color = (0.0, 0.86, 0.91)         # Electric Cyan
        self._tick_color = (0.52, 0.58, 0.62)        # Slate Grey
        self._active_tick_color = (1.0, 0.584, 0.0)  # Neon Orange

        self.setObjectName(f"mpr_view_{self.plane_type}")
        self.setFrameStyle(QFrame.NoFrame)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 1. Setup VTK Interactor Widget
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtkWidget)

        # 2. Setup VTK Renderer & Camera
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.039, 0.059, 0.071)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)

        self.camera = self.renderer.GetActiveCamera()
        self.camera.ParallelProjectionOn()

        # 3. Setup 2D Image Reslice & Actor Pipeline
        self._setup_reslice_pipeline()

        # 4. Setup 2D Crosshair Overlay
        self._setup_crosshair_overlay()

        # 5. Setup 2D Nerve Canal Cross-Section Overlays
        self._setup_nerve_overlays()

        # 6. Setup 2D Dental Arch & Cross-Section Ticks Overlays
        self._setup_arch_overlays()

        # 7. Setup Complete 4-Corner OSD HUD Annotations
        self._setup_hud_annotations()

        # 8. Setup Measurement Ruler Pipeline
        self._setup_measurement_pipeline()

        # 9. Setup Interactor Style
        self._setup_interactor_style()

    def _setup_reslice_pipeline(self) -> None:
        """Configures vtkImageReslice and Window/Level color mapping."""
        self.reslice = vtk.vtkImageReslice()
        self.reslice.SetOutputDimensionality(2)
        self.reslice.SetInterpolationModeToLinear()
        self.reslice.SetAutoCropOutput(True)
        self.reslice.SetBackgroundLevel(-1024.0)

        # Initial dummy input
        dummy_data = vtk.vtkImageData()
        dummy_data.SetDimensions(2, 2, 2)
        dummy_scalars = vtk.vtkShortArray()
        dummy_scalars.SetNumberOfTuples(8)
        dummy_scalars.FillValue(-1024)
        dummy_data.GetPointData().SetScalars(dummy_scalars)
        self.reslice.SetInputData(dummy_data)

        # Window/Level mapping to grayscale RGB
        self.color_map = vtk.vtkImageMapToWindowLevelColors()
        self.color_map.SetInputConnection(self.reslice.GetOutputPort())
        self.color_map.SetWindow(self._window_width)
        self.color_map.SetLevel(self._window_level)
        self.color_map.SetOutputFormatToRGB()

        # Image Actor
        self.image_actor = vtk.vtkImageActor()
        self.image_actor.GetMapper().SetInputConnection(self.color_map.GetOutputPort())
        self.image_actor.InterpolateOn()
        self.renderer.AddActor(self.image_actor)

    def _setup_crosshair_overlay(self) -> None:
        """Creates lightweight vtkActor2D crosshair lines."""
        self.crosshair_points = vtk.vtkPoints()
        self.crosshair_points.SetNumberOfPoints(4)
        for i in range(4):
            self.crosshair_points.SetPoint(i, 0.0, 0.0, 0.0)

        lines = vtk.vtkCellArray()
        # Horiz line
        h_line = vtk.vtkLine()
        h_line.GetPointIds().SetId(0, 0)
        h_line.GetPointIds().SetId(1, 1)
        lines.InsertNextCell(h_line)

        # Vert line
        v_line = vtk.vtkLine()
        v_line.GetPointIds().SetId(0, 2)
        v_line.GetPointIds().SetId(1, 3)
        lines.InsertNextCell(v_line)

        self.crosshair_polydata = vtk.vtkPolyData()
        self.crosshair_polydata.SetPoints(self.crosshair_points)
        self.crosshair_polydata.SetLines(lines)

        self.crosshair_mapper = vtk.vtkPolyDataMapper2D()
        self.crosshair_mapper.SetInputData(self.crosshair_polydata)

        self.crosshair_actor = vtk.vtkActor2D()
        self.crosshair_actor.SetMapper(self.crosshair_mapper)
        self.crosshair_actor.GetProperty().SetColor(*self._accent_color)
        self.crosshair_actor.GetProperty().SetLineWidth(1.2)
        self.crosshair_actor.GetProperty().SetOpacity(0.85)
        self.renderer.AddViewProp(self.crosshair_actor)

    def _setup_nerve_overlays(self) -> None:
        """Setup 2D overlay actors for Left and Right mandibular nerve intersection discs."""
        # Left Nerve 2D Actor (Pink)
        self.left_nerve_points = vtk.vtkPoints()
        self.left_nerve_lines = vtk.vtkCellArray()
        self.left_nerve_polydata = vtk.vtkPolyData()
        self.left_nerve_polydata.SetPoints(self.left_nerve_points)
        self.left_nerve_polydata.SetLines(self.left_nerve_lines)

        self.left_nerve_mapper = vtk.vtkPolyDataMapper2D()
        self.left_nerve_mapper.SetInputData(self.left_nerve_polydata)

        self.left_nerve_actor = vtk.vtkActor2D()
        self.left_nerve_actor.SetMapper(self.left_nerve_mapper)
        self.left_nerve_actor.GetProperty().SetColor(*self._left_nerve_color)
        self.left_nerve_actor.GetProperty().SetLineWidth(2.5)
        self.renderer.AddViewProp(self.left_nerve_actor)

        # Right Nerve 2D Actor (Orange)
        self.right_nerve_points = vtk.vtkPoints()
        self.right_nerve_lines = vtk.vtkCellArray()
        self.right_nerve_polydata = vtk.vtkPolyData()
        self.right_nerve_polydata.SetPoints(self.right_nerve_points)
        self.right_nerve_polydata.SetLines(self.right_nerve_lines)

        self.right_nerve_mapper = vtk.vtkPolyDataMapper2D()
        self.right_nerve_mapper.SetInputData(self.right_nerve_polydata)

        self.right_nerve_actor = vtk.vtkActor2D()
        self.right_nerve_actor.SetMapper(self.right_nerve_mapper)
        self.right_nerve_actor.GetProperty().SetColor(*self._right_nerve_color)
        self.right_nerve_actor.GetProperty().SetLineWidth(2.5)
        self.renderer.AddViewProp(self.right_nerve_actor)

    def _setup_arch_overlays(self) -> None:
        """Setup 2D Dental Arch curve and transverse cross-section tick markers."""
        # 1. Main Arch Spline Curve (Cyan)
        self.arch_points = vtk.vtkPoints()
        self.arch_lines = vtk.vtkCellArray()
        self.arch_polydata = vtk.vtkPolyData()
        self.arch_polydata.SetPoints(self.arch_points)
        self.arch_polydata.SetLines(self.arch_lines)

        self.arch_mapper = vtk.vtkPolyDataMapper2D()
        self.arch_mapper.SetInputData(self.arch_polydata)

        self.arch_actor = vtk.vtkActor2D()
        self.arch_actor.SetMapper(self.arch_mapper)
        self.arch_actor.GetProperty().SetColor(*self._arch_color)
        self.arch_actor.GetProperty().SetLineWidth(2.4)
        self.renderer.AddViewProp(self.arch_actor)

        # 2. Transverse Cross-Section Ticks (Grey)
        self.ticks_points = vtk.vtkPoints()
        self.ticks_lines = vtk.vtkCellArray()
        self.ticks_polydata = vtk.vtkPolyData()
        self.ticks_polydata.SetPoints(self.ticks_points)
        self.ticks_polydata.SetLines(self.ticks_lines)

        self.ticks_mapper = vtk.vtkPolyDataMapper2D()
        self.ticks_mapper.SetInputData(self.ticks_polydata)

        self.ticks_actor = vtk.vtkActor2D()
        self.ticks_actor.SetMapper(self.ticks_mapper)
        self.ticks_actor.GetProperty().SetColor(*self._tick_color)
        self.ticks_actor.GetProperty().SetLineWidth(1.4)
        self.renderer.AddViewProp(self.ticks_actor)

        # 3. Active Cross-Section Marker (Orange)
        self.active_tick_points = vtk.vtkPoints()
        self.active_tick_lines = vtk.vtkCellArray()
        self.active_tick_polydata = vtk.vtkPolyData()
        self.active_tick_polydata.SetPoints(self.active_tick_points)
        self.active_tick_polydata.SetLines(self.active_tick_lines)

        self.active_tick_mapper = vtk.vtkPolyDataMapper2D()
        self.active_tick_mapper.SetInputData(self.active_tick_polydata)

        self.active_tick_actor = vtk.vtkActor2D()
        self.active_tick_actor.SetMapper(self.active_tick_mapper)
        self.active_tick_actor.GetProperty().SetColor(*self._active_tick_color)
        self.active_tick_actor.GetProperty().SetLineWidth(3.2)
        self.renderer.AddViewProp(self.active_tick_actor)

        # 4. Active Tick Number Text
        self.active_tick_text = vtk.vtkTextActor()
        self.active_tick_text.GetTextProperty().SetFontSize(11)
        self.active_tick_text.GetTextProperty().SetColor(*self._active_tick_color)
        self.active_tick_text.GetTextProperty().SetFontFamilyToCourier()
        self.active_tick_text.GetTextProperty().BoldOn()
        self.active_tick_text.VisibilityOff()
        self.renderer.AddViewProp(self.active_tick_text)

    def _setup_hud_annotations(self) -> None:
        """Setup Complete 4-Corner OSD PACS HUD Annotations."""
        self.corner_annotation = vtk.vtkCornerAnnotation()
        self.corner_annotation.SetMaximumFontSize(12)
        self.corner_annotation.SetMinimumFontSize(10)
        self.corner_annotation.GetTextProperty().SetFontFamilyToCourier()
        self.corner_annotation.GetTextProperty().SetColor(0.87, 0.89, 0.91)
        self.corner_annotation.GetTextProperty().ShadowOn()

        # Corner Indices in vtkCornerAnnotation:
        # 0 = Bottom-Left
        # 1 = Bottom-Right
        # 2 = Top-Left
        # 3 = Top-Right

        self.corner_annotation.SetText(2, f"Matrix: --\nPlane: {self.plane_type.upper()}\nZoom: 100%")
        self.corner_annotation.SetText(3, "BAU MEDICAL SYSTEMS\nDental CBCT\nModality: CT")
        self.corner_annotation.SetText(0, f"Voxel: [--,--,--]\nHU: --\nWW: {self._window_width:.0f} WL: {self._window_level:.0f}")
        self.corner_annotation.SetText(1, "Slice: -- / --\nThickness: -- mm\nFoV: -- mm")

        self.renderer.AddViewProp(self.corner_annotation)

    def _setup_measurement_pipeline(self) -> None:
        """Setup 2D distance caliper overlay actor."""
        self.measure_points = vtk.vtkPoints()
        self.measure_lines = vtk.vtkCellArray()
        self.measure_polydata = vtk.vtkPolyData()
        self.measure_polydata.SetPoints(self.measure_points)
        self.measure_polydata.SetLines(self.measure_lines)

        self.measure_mapper = vtk.vtkPolyDataMapper2D()
        self.measure_mapper.SetInputData(self.measure_polydata)

        self.measure_actor = vtk.vtkActor2D()
        self.measure_actor.SetMapper(self.measure_mapper)
        self.measure_actor.GetProperty().SetColor(*self._secondary_color)
        self.measure_actor.GetProperty().SetLineWidth(1.8)
        self.measure_actor.VisibilityOff()
        self.renderer.AddViewProp(self.measure_actor)

        # Distance text actor
        self.measure_text = vtk.vtkTextActor()
        self.measure_text.GetTextProperty().SetFontSize(12)
        self.measure_text.GetTextProperty().SetColor(*self._accent_color)
        self.measure_text.GetTextProperty().SetFontFamilyToCourier()
        self.measure_text.GetTextProperty().BoldOn()
        self.measure_text.VisibilityOff()
        self.renderer.AddViewProp(self.measure_text)

    def _setup_interactor_style(self) -> None:
        """Custom mouse interactor style handling slice navigation, WL adjustment, and nerve tracing."""
        self.interactor_style = vtk.vtkInteractorStyleUser()
        self.vtkWidget.SetInteractorStyle(self.interactor_style)

        self._is_left_down = False
        self._is_right_down = False
        self._is_middle_down = False
        self._last_mouse_pos = (0, 0)

        self.interactor_style.AddObserver("LeftButtonPressEvent", self._on_left_press)
        self.interactor_style.AddObserver("LeftButtonReleaseEvent", self._on_left_release)
        self.interactor_style.AddObserver("RightButtonPressEvent", self._on_right_press)
        self.interactor_style.AddObserver("RightButtonReleaseEvent", self._on_right_release)
        self.interactor_style.AddObserver("MiddleButtonPressEvent", self._on_middle_press)
        self.interactor_style.AddObserver("MiddleButtonReleaseEvent", self._on_middle_release)
        self.interactor_style.AddObserver("MouseMoveEvent", self._on_mouse_move)
        self.interactor_style.AddObserver("MouseWheelForwardEvent", self._on_wheel_forward)
        self.interactor_style.AddObserver("MouseWheelBackwardEvent", self._on_wheel_backward)

    def set_volume_data(self, volume: VolumeData) -> None:
        """Attach VolumeData to this MPR view and initialize coordinates."""
        self.volume_data = volume
        self.reslice.SetInputData(volume.vtk_image_data)

        self._window_width = volume.metadata.window_width
        self._window_level = volume.metadata.window_center
        self.color_map.SetWindow(self._window_width)
        self.color_map.SetLevel(self._window_level)

        center = volume.get_center()
        self._current_world_pos = center
        self._current_voxel_idx = volume.world_to_index(*center)

        self._update_reslice_matrix()
        self._reset_camera()
        self._update_hud_text()
        self._update_crosshair_geometry()
        self._update_nerve_slice_overlays()
        self.safe_render()

    def set_nerve_tracer(self, tracer: NerveTracer) -> None:
        """Attaches the Mandibular Nerve Tracer."""
        self.nerve_tracer = tracer
        self._update_nerve_slice_overlays()
        self.safe_render()

    def set_nerve_drawing_mode(self, enabled: bool) -> None:
        """Enables or disables interactive seed point drawing mode."""
        self._is_nerve_drawing = enabled
        if enabled:
            self._is_measuring = False
            self._is_arch_drawing = False

    def set_dental_arch(self, curve: DentalArchCurve, mgr: Optional[CrossSectionManager] = None) -> None:
        """Attaches the Dental Arch Spline curve and Cross-Section Manager."""
        self.arch_curve = curve
        self.cross_section_mgr = mgr
        self._update_arch_slice_overlays()
        self.safe_render()

    def set_arch_drawing_mode(self, enabled: bool) -> None:
        """Enables or disables interactive Dental Arch seed point drawing mode."""
        self._is_arch_drawing = enabled
        if enabled:
            self._is_measuring = False
            self._is_nerve_drawing = False

    def set_active_tool(self, tool_name: str) -> None:
        """Sets the active pointer tool ('select'|'pan'|'zoom'|'wl'|'distance'|'angle')."""
        self._active_tool = tool_name
        if tool_name == "distance":
            self.start_measurement()
        elif tool_name == "clear":
            self.clear_measurement()
            self._active_tool = "select"

    def set_invert_colors(self, invert: bool) -> None:
        """Toggles Invert Grayscale (Negative film mode)."""
        self._is_inverted = invert
        if invert:
            self.color_map.SetWindow(-abs(self._window_width))
        else:
            self.color_map.SetWindow(abs(self._window_width))
        self.safe_render()

    def start_cine(self, fps: int = 15) -> None:
        """Starts automated slice cine player loop."""
        interval_ms = max(20, int(1000 / fps))
        self._cine_timer.start(interval_ms)

    def stop_cine(self) -> None:
        """Stops automated cine player."""
        self._cine_timer.stop()

    def set_world_position(self, x: float, y: float, z: float, force_reslice: bool = False) -> None:
        """
        High-Frequency Event-Throttled Position Update.
        """
        if self.volume_data is None:
            return

        new_voxel_idx = self.volume_data.world_to_index(x, y, z)
        plane_slice_changed = False

        if self.plane_type == "axial":
            plane_slice_changed = (new_voxel_idx[2] != self._current_voxel_idx[2])
        elif self.plane_type == "coronal":
            plane_slice_changed = (new_voxel_idx[1] != self._current_voxel_idx[1])
        elif self.plane_type == "sagittal":
            plane_slice_changed = (new_voxel_idx[0] != self._current_voxel_idx[0])

        self._current_world_pos = (x, y, z)
        self._current_voxel_idx = new_voxel_idx

        if plane_slice_changed or force_reslice:
            self._update_reslice_matrix()

        self._update_crosshair_geometry()
        self._update_nerve_slice_overlays()
        self._update_arch_slice_overlays()
        self._update_hud_text()
        self.safe_render()

        if self.plane_type == "axial":
            self.signals.slice_changed.emit("axial", new_voxel_idx[2], self.volume_data.nz)
        elif self.plane_type == "coronal":
            self.signals.slice_changed.emit("coronal", new_voxel_idx[1], self.volume_data.ny)
        elif self.plane_type == "sagittal":
            self.signals.slice_changed.emit("sagittal", new_voxel_idx[0], self.volume_data.nx)

    def set_window_level(self, window_width: float, window_level: float) -> None:
        """Set Window Width and Window Level (HU contrast)."""
        self._window_width = max(1.0, float(window_width))
        self._window_level = float(window_level)

        if self._is_inverted:
            self.color_map.SetWindow(-self._window_width)
        else:
            self.color_map.SetWindow(self._window_width)

        self.color_map.SetLevel(self._window_level)
        self._update_hud_text()
        self.safe_render()

    def set_crosshair_visible(self, visible: bool) -> None:
        """Toggle crosshair lines visibility."""
        self._is_crosshair_visible = visible
        if visible:
            self.crosshair_actor.VisibilityOn()
        else:
            self.crosshair_actor.VisibilityOff()
        self.safe_render()

    def start_measurement(self) -> None:
        """Enter 2D distance caliper measurement mode."""
        self._is_measuring = True
        self._is_nerve_drawing = False
        self._measurement_points.clear()
        self.measure_actor.VisibilityOff()
        self.measure_text.VisibilityOff()
        self.safe_render()

    def clear_measurement(self) -> None:
        """Clear active measurement caliper."""
        self._is_measuring = False
        self._measurement_points.clear()
        self.measure_actor.VisibilityOff()
        self.measure_text.VisibilityOff()
        self.safe_render()

    def _update_reslice_matrix(self) -> None:
        """Updates vtkImageReslice 4x4 matrix with anatomical standard direction."""
        if self.volume_data is None:
            return
        matrix = self.volume_data.get_reslice_matrix_for_plane(self.plane_type, self._current_world_pos)
        self.reslice.SetResliceAxes(matrix)
        self.reslice.Update()
        self.color_map.Update()

    def _reset_camera(self) -> None:
        """Dynamically aligns parallel projection camera with the 2D resliced image plane."""
        if self.volume_data is None:
            return

        self._update_reslice_matrix()
        bounds = self.image_actor.GetBounds()
        if bounds[1] < bounds[0] or bounds[3] < bounds[2]:
            dims = self.volume_data.physical_size_mm
            cx, cy, cz = 0.0, 0.0, 0.0
            max_span = max(dims)
        else:
            cx = (bounds[0] + bounds[1]) * 0.5
            cy = (bounds[2] + bounds[3]) * 0.5
            cz = (bounds[4] + bounds[5]) * 0.5
            w = bounds[1] - bounds[0]
            h = bounds[3] - bounds[2]
            max_span = max(w, h)

        self.camera.SetFocalPoint(cx, cy, cz)
        self.camera.SetPosition(cx, cy, cz + 1000.0)
        self.camera.SetViewUp(0.0, 1.0, 0.0)
        self.camera.SetParallelScale(max(1.0, max_span * 0.55))
        self.renderer.ResetCameraClippingRange()

    def _update_crosshair_geometry(self) -> None:
        """Update 2D crosshair overlay actor coordinates."""
        if not self._is_crosshair_visible or self.volume_data is None:
            return

        size = self.vtkWidget.GetRenderWindow().GetSize()
        w, h = max(1, size[0]), max(1, size[1])

        reslice_matrix = self.reslice.GetResliceAxes()
        if reslice_matrix is None:
            return

        inv_matrix = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(reslice_matrix, inv_matrix)
        p_in = [self._current_world_pos[0], self._current_world_pos[1], self._current_world_pos[2], 1.0]
        p_local = [0.0, 0.0, 0.0, 0.0]
        inv_matrix.MultiplyPoint(p_in, p_local)

        self.renderer.SetWorldPoint(p_local[0], p_local[1], 0.0, 1.0)
        self.renderer.WorldToDisplay()
        disp_pt = self.renderer.GetDisplayPoint()
        cx, cy = disp_pt[0], disp_pt[1]

        self.crosshair_points.SetPoint(0, 0.0, cy, 0.0)
        self.crosshair_points.SetPoint(1, float(w), cy, 0.0)
        self.crosshair_points.SetPoint(2, cx, 0.0, 0.0)
        self.crosshair_points.SetPoint(3, cx, float(h), 0.0)

        self.crosshair_polydata.Modified()

    def _update_nerve_slice_overlays(self) -> None:
        """Calculates and renders 2D cross-sectional circles for Left and Right nerve tracks."""
        if self.nerve_tracer is None or self.volume_data is None:
            self.left_nerve_points.Reset()
            self.left_nerve_lines.Reset()
            self.left_nerve_polydata.Modified()
            self.right_nerve_points.Reset()
            self.right_nerve_lines.Reset()
            self.right_nerve_polydata.Modified()
            return

        reslice_matrix = self.reslice.GetResliceAxes()
        if reslice_matrix is None:
            return

        inv_matrix = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(reslice_matrix, inv_matrix)

        left_intersections = self.nerve_tracer.get_track(NerveChannel.LEFT).calculate_2d_slice_intersections(
            self.plane_type, self._current_world_pos
        )
        self._build_nerve_circle_polydata(
            left_intersections, inv_matrix, self.left_nerve_points, self.left_nerve_lines, self.left_nerve_polydata
        )

        right_intersections = self.nerve_tracer.get_track(NerveChannel.RIGHT).calculate_2d_slice_intersections(
            self.plane_type, self._current_world_pos
        )
        self._build_nerve_circle_polydata(
            right_intersections, inv_matrix, self.right_nerve_points, self.right_nerve_lines, self.right_nerve_polydata
        )

    def _build_nerve_circle_polydata(
        self,
        intersections: List[Tuple[float, float, float, float]],
        inv_matrix: vtk.vtkMatrix4x4,
        pts: vtk.vtkPoints,
        lines: vtk.vtkCellArray,
        polydata: vtk.vtkPolyData
    ) -> None:
        pts.Reset()
        lines.Reset()

        if not intersections:
            polydata.Modified()
            return

        num_segments = 24
        pt_idx = 0

        for (ix, iy, iz, radius_mm) in intersections:
            p_in = [ix, iy, iz, 1.0]
            p_local = [0.0, 0.0, 0.0, 0.0]
            inv_matrix.MultiplyPoint(p_in, p_local)

            self.renderer.SetWorldPoint(p_local[0], p_local[1], 0.0, 1.0)
            self.renderer.WorldToDisplay()
            center_disp = self.renderer.GetDisplayPoint()
            cx, cy = center_disp[0], center_disp[1]

            self.renderer.SetWorldPoint(p_local[0] + radius_mm, p_local[1], 0.0, 1.0)
            self.renderer.WorldToDisplay()
            edge_disp = self.renderer.GetDisplayPoint()
            r_pixels = max(3.0, abs(edge_disp[0] - cx))

            start_idx = pt_idx
            for s in range(num_segments):
                theta = 2.0 * math.pi * s / num_segments
                px = cx + r_pixels * math.cos(theta)
                py = cy + r_pixels * math.sin(theta)
                pts.InsertNextPoint(px, py, 0.0)
                pt_idx += 1

            for s in range(num_segments):
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, start_idx + s)
                line.GetPointIds().SetId(1, start_idx + ((s + 1) % num_segments))
                lines.InsertNextCell(line)

            # Center cross
            pts.InsertNextPoint(cx - r_pixels * 0.5, cy, 0.0)
            pts.InsertNextPoint(cx + r_pixels * 0.5, cy, 0.0)
            pts.InsertNextPoint(cx, cy - r_pixels * 0.5, 0.0)
            pts.InsertNextPoint(cx, cy + r_pixels * 0.5, 0.0)

            c_line1 = vtk.vtkLine()
            c_line1.GetPointIds().SetId(0, pt_idx)
            c_line1.GetPointIds().SetId(1, pt_idx + 1)
            lines.InsertNextCell(c_line1)

            c_line2 = vtk.vtkLine()
            c_line2.GetPointIds().SetId(0, pt_idx + 2)
            c_line2.GetPointIds().SetId(1, pt_idx + 3)
            lines.InsertNextCell(c_line2)

            pt_idx += 4

        polydata.Modified()

    def _update_arch_slice_overlays(self) -> None:
        """Renders Dental Arch Spline and Transverse Cross-Section Ticks on the 2D slice."""
        if self.plane_type != "axial" or self.arch_curve is None or len(self.arch_curve.sampled_points) < 2:
            self.arch_points.Reset()
            self.arch_lines.Reset()
            self.arch_polydata.Modified()
            self.ticks_points.Reset()
            self.ticks_lines.Reset()
            self.ticks_polydata.Modified()
            self.active_tick_points.Reset()
            self.active_tick_lines.Reset()
            self.active_tick_polydata.Modified()
            self.active_tick_text.VisibilityOff()
            return

        reslice_matrix = self.reslice.GetResliceAxes()
        if reslice_matrix is None:
            return

        inv_matrix = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(reslice_matrix, inv_matrix)

        def to_disp(wx, wy, wz):
            p_in = [wx, wy, wz, 1.0]
            p_local = [0.0, 0.0, 0.0, 0.0]
            inv_matrix.MultiplyPoint(p_in, p_local)
            self.renderer.SetWorldPoint(p_local[0], p_local[1], 0.0, 1.0)
            self.renderer.WorldToDisplay()
            d = self.renderer.GetDisplayPoint()
            return d[0], d[1]

        # 1. Arch Spline Curve (Cyan)
        self.arch_points.Reset()
        self.arch_lines.Reset()

        pts = self.arch_curve.sampled_points
        num_pts = len(pts)
        for i in range(num_pts):
            dx, dy = to_disp(pts[i, 0], pts[i, 1], pts[i, 2])
            self.arch_points.InsertNextPoint(dx, dy, 0.0)

        for i in range(num_pts - 1):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, i)
            line.GetPointIds().SetId(1, i + 1)
            self.arch_lines.InsertNextCell(line)

        self.arch_polydata.Modified()
        self.arch_actor.VisibilityOn()

        # 2. Cross-Section Ticks (Grey)
        self.ticks_points.Reset()
        self.ticks_lines.Reset()

        tick_endpoints = self.arch_curve.get_cross_section_tick_endpoints(tick_length_mm=8.0)
        pt_idx = 0
        for i, (p_b, p_l) in enumerate(tick_endpoints):
            if i % 3 != 0 and i != 0 and i != len(tick_endpoints) - 1:
                continue

            db = to_disp(*p_b)
            dl = to_disp(*p_l)

            self.ticks_points.InsertNextPoint(db[0], db[1], 0.0)
            self.ticks_points.InsertNextPoint(dl[0], dl[1], 0.0)

            l = vtk.vtkLine()
            l.GetPointIds().SetId(0, pt_idx)
            l.GetPointIds().SetId(1, pt_idx + 1)
            self.ticks_lines.InsertNextCell(l)
            pt_idx += 2

        self.ticks_polydata.Modified()
        self.ticks_actor.VisibilityOn()

        # 3. Active Cross-Section Tick (Neon Orange)
        self.active_tick_points.Reset()
        self.active_tick_lines.Reset()

        active_idx = self.cross_section_mgr.active_index if self.cross_section_mgr else 0
        if 0 <= active_idx < len(tick_endpoints):
            p_b, p_l = tick_endpoints[active_idx]
            db = to_disp(*p_b)
            dl = to_disp(*p_l)

            self.active_tick_points.InsertNextPoint(db[0], db[1], 0.0)
            self.active_tick_points.InsertNextPoint(dl[0], dl[1], 0.0)

            al = vtk.vtkLine()
            al.GetPointIds().SetId(0, 0)
            al.GetPointIds().SetId(1, 1)
            self.active_tick_lines.InsertNextCell(al)

            self.active_tick_polydata.Modified()
            self.active_tick_actor.VisibilityOn()

            self.active_tick_text.SetInput(f"#{active_idx + 1}")
            self.active_tick_text.SetDisplayPosition(int(db[0] + 6), int(db[1] + 4))
            self.active_tick_text.VisibilityOn()
        else:
            self.active_tick_actor.VisibilityOff()
            self.active_tick_text.VisibilityOff()

    def _update_hud_text(self) -> None:
        """Update Complete 4-Corner OSD PACS HUD Labels."""
        if self.volume_data is None:
            return

        meta = self.volume_data.metadata
        x, y, z = self._current_world_pos
        i, j, k = self._current_voxel_idx
        hu = self.volume_data.get_hu_at_voxel(i, j, k)

        # Tissue Classification
        tissue = "Air"
        if hu > 1800:
            tissue = "Enamel / Metal"
        elif hu > 800:
            tissue = "Cortical Bone"
        elif hu > 250:
            tissue = "Trabecular Bone"
        elif hu > -50:
            tissue = "Soft Tissue"
        elif hu > -200:
            tissue = "Adipose"

        # Active Slice Index & Max Slices
        if self.plane_type == "axial":
            cur_slice = k + 1
            max_slices = self.volume_data.nz
            thickness = self.volume_data.spacing[2]
            fov = self.volume_data.physical_size_mm[0]
        elif self.plane_type == "coronal":
            cur_slice = j + 1
            max_slices = self.volume_data.ny
            thickness = self.volume_data.spacing[1]
            fov = self.volume_data.physical_size_mm[0]
        else:
            cur_slice = i + 1
            max_slices = self.volume_data.nx
            thickness = self.volume_data.spacing[0]
            fov = self.volume_data.physical_size_mm[1]

        # Top-Left (2)
        hud_tl = (
            f"Matrix: {self.volume_data.nx}×{self.volume_data.ny}\n"
            f"Plane: {self.plane_type.upper()}\n"
            f"Pos: {x:+.1f}, {y:+.1f}, {z:+.1f} mm"
        )
        self.corner_annotation.SetText(2, hud_tl)

        # Top-Right (3)
        hud_tr = (
            f"{meta.patient_name}\n"
            f"ID: {meta.patient_id}\n"
            f"{meta.modality} +C | {meta.study_date}"
        )
        self.corner_annotation.SetText(3, hud_tr)

        # Bottom-Left (0)
        hud_bl = (
            f"Voxel: [{i},{j},{k}]\n"
            f"HU: {hu:+.0f} [{tissue}]\n"
            f"W: {self._window_width:.0f} L: {self._window_level:.0f}"
        )
        self.corner_annotation.SetText(0, hud_bl)

        # Bottom-Right (1)
        hud_br = (
            f"Slice: {cur_slice} / {max_slices}\n"
            f"Thick: {thickness:.2f} mm\n"
            f"FoV: {fov:.1f} mm"
        )
        self.corner_annotation.SetText(1, hud_br)

    # --------------------------------------------------------------------------
    # Mouse & Keyboard Event Handlers
    # --------------------------------------------------------------------------

    def _on_left_press(self, obj, event) -> None:
        self.signals.focused.emit(self.plane_type)
        self._is_left_down = True
        pos = self.vtkWidget.GetRenderWindow().GetInteractor().GetEventPosition()
        self._last_mouse_pos = pos

        if self._is_arch_drawing and self.arch_curve is not None and self.plane_type == "axial":
            world_pos = self._display_to_world(pos[0], pos[1])
            if world_pos is not None:
                self.arch_curve.add_seed_point(*world_pos)
                self._update_arch_slice_overlays()
                self.signals.arch_point_placed.emit(*world_pos)
                self.safe_render()
        elif self._is_nerve_drawing and self.nerve_tracer is not None:
            world_pos = self._display_to_world(pos[0], pos[1])
            if world_pos is not None:
                self.nerve_tracer.add_point(*world_pos)
                self._update_nerve_slice_overlays()
                self.signals.nerve_point_placed.emit(*world_pos)
                self.safe_render()
        elif self._is_measuring:
            self._handle_measurement_click(pos)
        elif self._active_tool == "pan":
            pass
        elif self._active_tool == "zoom":
            pass
        else:
            self._pick_and_update_world_position(pos[0], pos[1])

    def _on_left_release(self, obj, event) -> None:
        self._is_left_down = False

    def _on_right_press(self, obj, event) -> None:
        self.signals.focused.emit(self.plane_type)
        self._is_right_down = True
        self._last_mouse_pos = self.vtkWidget.GetRenderWindow().GetInteractor().GetEventPosition()

    def _on_right_release(self, obj, event) -> None:
        self._is_right_down = False

    def _on_middle_press(self, obj, event) -> None:
        self.signals.focused.emit(self.plane_type)
        self._is_middle_down = True
        self._last_mouse_pos = self.vtkWidget.GetRenderWindow().GetInteractor().GetEventPosition()

    def _on_middle_release(self, obj, event) -> None:
        self._is_middle_down = False

    def _on_mouse_move(self, obj, event) -> None:
        pos = self.vtkWidget.GetRenderWindow().GetInteractor().GetEventPosition()
        dx = pos[0] - self._last_mouse_pos[0]
        dy = pos[1] - self._last_mouse_pos[1]
        self._last_mouse_pos = pos

        if self._is_left_down and not self._is_measuring and not self._is_nerve_drawing:
            if self._active_tool == "pan":
                # Pan camera
                self.camera.SetFocalPoint(
                    self.camera.GetFocalPoint()[0] - dx * 0.5,
                    self.camera.GetFocalPoint()[1] - dy * 0.5,
                    self.camera.GetFocalPoint()[2]
                )
                self.camera.SetPosition(
                    self.camera.GetPosition()[0] - dx * 0.5,
                    self.camera.GetPosition()[1] - dy * 0.5,
                    self.camera.GetPosition()[2]
                )
                self.safe_render()
            elif self._active_tool == "zoom":
                # Zoom camera
                factor = 1.0 - dy * 0.01
                self.camera.SetParallelScale(max(1.0, self.camera.GetParallelScale() * factor))
                self.safe_render()
            else:
                self._pick_and_update_world_position(pos[0], pos[1])

        elif self._is_right_down or (self._is_left_down and self._active_tool == "wl"):
            new_ww = max(1.0, self._window_width + dx * 4.0)
            new_wl = self._window_level + dy * 4.0
            self.set_window_level(new_ww, new_wl)
            self.signals.window_level_changed.emit(new_ww, new_wl)

        elif self._is_middle_down:
            # Middle button drag = Pan
            self.camera.SetFocalPoint(
                self.camera.GetFocalPoint()[0] - dx * 0.5,
                self.camera.GetFocalPoint()[1] - dy * 0.5,
                self.camera.GetFocalPoint()[2]
            )
            self.camera.SetPosition(
                self.camera.GetPosition()[0] - dx * 0.5,
                self.camera.GetPosition()[1] - dy * 0.5,
                self.camera.GetPosition()[2]
            )
            self.safe_render()

        elif self.volume_data is not None:
            world_coord = self._display_to_world(pos[0], pos[1])
            if world_coord is not None:
                hu = self.volume_data.get_hu_at_world(*world_coord)
                self.signals.hu_inspected.emit(world_coord[0], world_coord[1], world_coord[2], hu)

    def _on_wheel_forward(self, obj, event) -> None:
        self._step_slice(step=1)

    def _on_wheel_backward(self, obj, event) -> None:
        self._step_slice(step=-1)

    def _step_slice(self, step: int) -> None:
        if self.volume_data is None:
            return

        dx, dy, dz = self.volume_data.spacing
        x, y, z = self._current_world_pos
        if self.plane_type == "axial":
            z += step * dz
            # Loop for cine
            bounds = self.volume_data.get_bounds()
            if z > bounds[5]:
                z = bounds[4]
            elif z < bounds[4]:
                z = bounds[5]
        elif self.plane_type == "coronal":
            y += step * dy
            bounds = self.volume_data.get_bounds()
            if y > bounds[3]:
                y = bounds[2]
            elif y < bounds[2]:
                y = bounds[3]
        elif self.plane_type == "sagittal":
            x += step * dx
            bounds = self.volume_data.get_bounds()
            if x > bounds[1]:
                x = bounds[0]
            elif x < bounds[0]:
                x = bounds[1]

        self.set_world_position(x, y, z, force_reslice=True)
        self.signals.crosshair_moved.emit(x, y, z)

    def _display_to_world(self, dx: int, dy: int) -> Optional[Tuple[float, float, float]]:
        if self.volume_data is None:
            return None

        self.renderer.SetDisplayPoint(dx, dy, 0.0)
        self.renderer.DisplayToWorld()
        world_pt = self.renderer.GetWorldPoint()
        if world_pt[3] == 0.0:
            return None

        wx, wy = world_pt[0] / world_pt[3], world_pt[1] / world_pt[3]

        reslice_matrix = self.reslice.GetResliceAxes()
        if reslice_matrix is None:
            return None

        p_in = [wx, wy, 0.0, 1.0]
        p_out = [0.0, 0.0, 0.0, 0.0]
        reslice_matrix.MultiplyPoint(p_in, p_out)
        return (p_out[0], p_out[1], p_out[2])

    def _pick_and_update_world_position(self, dx: int, dy: int) -> None:
        world_pos = self._display_to_world(dx, dy)
        if world_pos is not None:
            self.set_world_position(*world_pos)
            self.signals.crosshair_moved.emit(*world_pos)

    def _handle_measurement_click(self, pos: Tuple[int, int]) -> None:
        world_pos = self._display_to_world(pos[0], pos[1])
        if world_pos is None:
            return

        self._measurement_points.append(world_pos)

        if self._active_tool == "angle":
            if len(self._measurement_points) == 3:
                p1 = np.array(self._measurement_points[0])
                p2 = np.array(self._measurement_points[1])  # Vertex
                p3 = np.array(self._measurement_points[2])

                v1 = p1 - p2
                v2 = p3 - p2
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)

                if norm1 > 1e-4 and norm2 > 1e-4:
                    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    angle_deg = float(np.degrees(np.arccos(cos_theta)))
                else:
                    angle_deg = 0.0

                self._render_angle_measurement(angle_deg)
                self.signals.measurement_completed.emit("angle", angle_deg)
                self._is_measuring = False

        elif self._active_tool == "roi":
            if len(self._measurement_points) == 2:
                p1 = self._measurement_points[0]
                p2 = self._measurement_points[1]
                self._render_roi_measurement(p1, p2)
                self._is_measuring = False

        else:
            # Default 2-point Distance Caliper
            if len(self._measurement_points) == 2:
                p1 = self._measurement_points[0]
                p2 = self._measurement_points[1]
                dist_mm = math.dist(p1, p2)

                self._render_measurement_ruler(pos, dist_mm)
                self.signals.measurement_completed.emit("distance", dist_mm)
                self._is_measuring = False

    def _render_angle_measurement(self, angle_deg: float) -> None:
        """Renders 3-point angle measurement rays and degree label."""
        if len(self._measurement_points) < 3:
            return

        reslice_matrix = self.reslice.GetResliceAxes()
        inv_matrix = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(reslice_matrix, inv_matrix)

        disp_pts = []
        for p in self._measurement_points:
            p_in = [p[0], p[1], p[2], 1.0]
            p_local = [0.0, 0.0, 0.0, 0.0]
            inv_matrix.MultiplyPoint(p_in, p_local)

            self.renderer.SetWorldPoint(p_local[0], p_local[1], 0.0, 1.0)
            self.renderer.WorldToDisplay()
            d = self.renderer.GetDisplayPoint()
            disp_pts.append((d[0], d[1]))

        self.measure_points.Reset()
        self.measure_lines.Reset()

        for d in disp_pts:
            self.measure_points.InsertNextPoint(d[0], d[1], 0.0)

        # Ray 1: P2 -> P1
        line1 = vtk.vtkLine()
        line1.GetPointIds().SetId(0, 1)
        line1.GetPointIds().SetId(1, 0)
        self.measure_lines.InsertNextCell(line1)

        # Ray 2: P2 -> P3
        line2 = vtk.vtkLine()
        line2.GetPointIds().SetId(0, 1)
        line2.GetPointIds().SetId(1, 2)
        self.measure_lines.InsertNextCell(line2)

        self.measure_polydata.Modified()
        self.measure_actor.VisibilityOn()

        # Place label near vertex P2
        vx, vy = disp_pts[1][0] + 12, disp_pts[1][1] + 12
        self.measure_text.SetInput(f"Angle: {angle_deg:.1f}°")
        self.measure_text.SetDisplayPosition(int(vx), int(vy))
        self.measure_text.VisibilityOn()
        self.safe_render()

    def _render_roi_measurement(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> None:
        """Renders rectangular ROI box and HU statistics."""
        reslice_matrix = self.reslice.GetResliceAxes()
        inv_matrix = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(reslice_matrix, inv_matrix)

        def to_disp(p):
            p_in = [p[0], p[1], p[2], 1.0]
            p_local = [0.0, 0.0, 0.0, 0.0]
            inv_matrix.MultiplyPoint(p_in, p_local)
            self.renderer.SetWorldPoint(p_local[0], p_local[1], 0.0, 1.0)
            self.renderer.WorldToDisplay()
            d = self.renderer.GetDisplayPoint()
            return d[0], d[1]

        d1 = to_disp(p1)
        d2 = to_disp(p2)

        xmin, xmax = min(d1[0], d2[0]), max(d1[0], d2[0])
        ymin, ymax = min(d1[1], d2[1]), max(d1[1], d2[1])

        self.measure_points.Reset()
        self.measure_lines.Reset()

        self.measure_points.InsertNextPoint(xmin, ymin, 0.0)
        self.measure_points.InsertNextPoint(xmax, ymin, 0.0)
        self.measure_points.InsertNextPoint(xmax, ymax, 0.0)
        self.measure_points.InsertNextPoint(xmin, ymax, 0.0)

        for s in range(4):
            l = vtk.vtkLine()
            l.GetPointIds().SetId(0, s)
            l.GetPointIds().SetId(1, (s + 1) % 4)
            self.measure_lines.InsertNextCell(l)

        self.measure_polydata.Modified()
        self.measure_actor.VisibilityOn()

        # Sample approximate center HU
        cx, cy, cz = (p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5, (p1[2] + p2[2]) * 0.5
        hu = self.volume_data.get_hu_at_world(cx, cy, cz) if self.volume_data else 0.0
        w_mm = abs(p2[0] - p1[0])
        h_mm = abs(p2[1] - p1[1])
        area_mm2 = max(1.0, w_mm * h_mm)

        self.measure_text.SetInput(f"ROI: {area_mm2:.1f} mm²\nMean HU: {hu:+.0f}")
        self.measure_text.SetDisplayPosition(int(xmax + 6), int(ymax - 10))
        self.measure_text.VisibilityOn()
        self.safe_render()
        self.signals.measurement_completed.emit("roi", hu)

    def _render_measurement_ruler(self, current_disp_pos: Tuple[int, int], dist_mm: float) -> None:
        p1 = self._measurement_points[0]
        reslice_matrix = self.reslice.GetResliceAxes()
        inv_matrix = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(reslice_matrix, inv_matrix)
        p_in = [p1[0], p1[1], p1[2], 1.0]
        p_local = [0.0, 0.0, 0.0, 0.0]
        inv_matrix.MultiplyPoint(p_in, p_local)

        self.renderer.SetWorldPoint(p_local[0], p_local[1], 0.0, 1.0)
        self.renderer.WorldToDisplay()
        d1 = self.renderer.GetDisplayPoint()

        self.measure_points.Reset()
        self.measure_lines.Reset()

        self.measure_points.InsertNextPoint(d1[0], d1[1], 0.0)
        self.measure_points.InsertNextPoint(current_disp_pos[0], current_disp_pos[1], 0.0)

        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, 0)
        line.GetPointIds().SetId(1, 1)
        self.measure_lines.InsertNextCell(line)

        self.measure_polydata.Modified()
        self.measure_actor.VisibilityOn()

        mx = (d1[0] + current_disp_pos[0]) * 0.5 + 10
        my = (d1[1] + current_disp_pos[1]) * 0.5 + 10
        self.measure_text.SetInput(f"{dist_mm:.2f} mm")
        self.measure_text.SetDisplayPosition(int(mx), int(my))
        self.measure_text.VisibilityOn()

        self.safe_render()

    def safe_render(self) -> None:
        """Safely executes Render only when viewport has valid dimensions and is visible."""
        if hasattr(self, 'vtkWidget') and self.vtkWidget is not None:
            rw = self.vtkWidget.GetRenderWindow()
            if rw and self.isVisible() and self.width() > 10 and self.height() > 10:
                try:
                    rw.Render()
                except Exception:
                    pass

    def cleanup(self) -> None:
        """Cleanly releases VTK render window and OpenGL context before Qt widget destruction."""
        self.stop_cine()
        if hasattr(self, 'vtkWidget') and self.vtkWidget is not None:
            try:
                rw = self.vtkWidget.GetRenderWindow()
                if rw is not None:
                    rw.Finalize()
            except Exception:
                pass

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.width() > 10 and self.height() > 10:
            self._update_crosshair_geometry()
            self._update_nerve_slice_overlays()
            self._update_arch_slice_overlays()
            self.safe_render()
