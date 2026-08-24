"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Module: rendering/panoramic_view.py

Interactive 2D Panoramic Radiograph (Curved MPR Focal Trough) Viewport.
Features:
- Hardware-accelerated 2D display of unrolled synthetic panoramic dental radiograph.
- Synchronized Window Width / Window Level (WW/WL) contrast adjustment.
- Interactive vertical reference indicator line (vtkActor2D) showing the active cross-section slice.
- Click & Drag on panoramic image to scrub through cross-section slices along the arch.
- Complete 4-Corner OSD PACS HUD (Focal Trough thickness, Arc Length mm, WW/WL).
- Distance measurement caliper for vertical bone height and anatomical landmarks.
"""

from __future__ import annotations
from typing import Optional, Tuple
import math
import numpy as np
import vtk
from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QFrame

try:
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
except ImportError:
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from core.volume_data import VolumeData
from dental.panoramic_mpr import DentalArchCurve, PanoramicGenerator


class PanoramicViewSignals(QObject):
    """Qt Signals emitted from the Panoramic Viewport."""
    cross_section_selected = Signal(int)             # Active cross-section index
    window_level_changed = Signal(float, float)      # WW, WL
    measurement_completed = Signal(float)            # Caliper distance mm
    focused = Signal(str)                            # 'panoramic'


class PanoramicView(QFrame):
    """
    Interactive 2D Panoramic (Curved MPR) Viewport.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.signals = PanoramicViewSignals()
        self.volume_data: Optional[VolumeData] = None
        self.arch_curve: Optional[DentalArchCurve] = None
        self.panoramic_generator = PanoramicGenerator()

        self._window_width: float = 2500.0
        self._window_level: float = 500.0
        self._active_cross_section_idx: int = 0
        self._is_measuring: bool = False
        self._measurement_points: list[Tuple[float, float]] = []

        self._accent_color = (0.0, 0.86, 0.91)        # Electric Cyan
        self._cursor_color = (1.0, 0.584, 0.0)        # Neon Orange (#ff9500)
        self._secondary_color = (0.72, 0.78, 0.89)

        self.setObjectName("panoramic_view")
        self.setFrameStyle(QFrame.NoFrame)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 1. VTK Interactor Widget
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtkWidget)

        # 2. Renderer & Camera
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.039, 0.059, 0.071)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)

        self.camera = self.renderer.GetActiveCamera()
        self.camera.ParallelProjectionOn()

        # 3. Image Actor Pipeline
        self._setup_image_pipeline()

        # 4. Vertical Cross-Section Indicator Line
        self._setup_cursor_line()

        # 5. Corner HUD Annotations
        self._setup_hud()

        # 6. Measurement Pipeline
        self._setup_measurement_pipeline()

        # 7. Interactor Style
        self._setup_interactor_style()

    def _setup_image_pipeline(self) -> None:
        """Configures 2D Panoramic vtkImageActor and Window/Level mapper."""
        self.color_map = vtk.vtkImageMapToWindowLevelColors()
        self.color_map.SetWindow(self._window_width)
        self.color_map.SetLevel(self._window_level)
        self.color_map.SetOutputFormatToRGB()

        # Dummy initial 2D image
        dummy_data = vtk.vtkImageData()
        dummy_data.SetDimensions(2, 2, 1)
        dummy_scalars = vtk.vtkShortArray()
        dummy_scalars.SetNumberOfTuples(4)
        dummy_scalars.FillValue(-1024)
        dummy_data.GetPointData().SetScalars(dummy_scalars)
        self.color_map.SetInputData(dummy_data)

        self.image_actor = vtk.vtkImageActor()
        self.image_actor.GetMapper().SetInputConnection(self.color_map.GetOutputPort())
        self.image_actor.InterpolateOn()
        self.renderer.AddActor(self.image_actor)

    def _setup_cursor_line(self) -> None:
        """Creates vertical orange reference line indicating the active cross-section."""
        self.cursor_points = vtk.vtkPoints()
        self.cursor_points.SetNumberOfPoints(2)
        self.cursor_points.SetPoint(0, 0.0, 0.0, 0.0)
        self.cursor_points.SetPoint(1, 0.0, 100.0, 0.0)

        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, 0)
        line.GetPointIds().SetId(1, 1)
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(line)

        self.cursor_polydata = vtk.vtkPolyData()
        self.cursor_polydata.SetPoints(self.cursor_points)
        self.cursor_polydata.SetLines(lines)

        self.cursor_mapper = vtk.vtkPolyDataMapper2D()
        self.cursor_mapper.SetInputData(self.cursor_polydata)

        self.cursor_actor = vtk.vtkActor2D()
        self.cursor_actor.SetMapper(self.cursor_mapper)
        self.cursor_actor.GetProperty().SetColor(*self._cursor_color)
        self.cursor_actor.GetProperty().SetLineWidth(2.0)
        self.cursor_actor.VisibilityOff()
        self.renderer.AddViewProp(self.cursor_actor)

    def _setup_hud(self) -> None:
        """Configures 4-Corner OSD PACS HUD."""
        self.corner_annotation = vtk.vtkCornerAnnotation()
        self.corner_annotation.SetMaximumFontSize(12)
        self.corner_annotation.SetMinimumFontSize(10)
        self.corner_annotation.GetTextProperty().SetFontFamilyToCourier()
        self.corner_annotation.GetTextProperty().SetColor(0.87, 0.89, 0.91)
        self.corner_annotation.GetTextProperty().ShadowOn()

        self.corner_annotation.SetText(2, "PANORAMIC (Curved MPR)\nMode: MIP Focal Trough\nTrough: 8.0 mm")
        self.corner_annotation.SetText(3, "BAU MEDICAL SYSTEMS\nDental Panoramic View")
        self.corner_annotation.SetText(0, f"W: {self._window_width:.0f} L: {self._window_level:.0f}\nCross-Section: #0")
        self.corner_annotation.SetText(1, "Arc Length: 0.0 mm\nZoom: 100%")

        self.renderer.AddViewProp(self.corner_annotation)

    def _setup_measurement_pipeline(self) -> None:
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

        self.measure_text = vtk.vtkTextActor()
        self.measure_text.GetTextProperty().SetFontSize(12)
        self.measure_text.GetTextProperty().SetColor(*self._accent_color)
        self.measure_text.GetTextProperty().SetFontFamilyToCourier()
        self.measure_text.GetTextProperty().BoldOn()
        self.measure_text.VisibilityOff()
        self.renderer.AddViewProp(self.measure_text)

    def _setup_interactor_style(self) -> None:
        self.interactor_style = vtk.vtkInteractorStyleUser()
        self.vtkWidget.SetInteractorStyle(self.interactor_style)

        self._is_left_down = False
        self._is_right_down = False
        self._last_mouse_pos = (0, 0)

        self.interactor_style.AddObserver("LeftButtonPressEvent", self._on_left_press)
        self.interactor_style.AddObserver("LeftButtonReleaseEvent", self._on_left_release)
        self.interactor_style.AddObserver("RightButtonPressEvent", self._on_right_press)
        self.interactor_style.AddObserver("RightButtonReleaseEvent", self._on_right_release)
        self.interactor_style.AddObserver("MouseMoveEvent", self._on_mouse_move)

    def update_panoramic(self, volume: VolumeData, arch_curve: DentalArchCurve) -> None:
        """Regenerates the unrolled panoramic image and fits camera."""
        self.volume_data = volume
        self.arch_curve = arch_curve

        vtk_img = self.panoramic_generator.generate_panoramic_vtk_image(volume, arch_curve)
        if vtk_img is None:
            return

        self.color_map.SetInputData(vtk_img)
        self.color_map.Update()

        self._reset_camera()
        self.cursor_actor.VisibilityOn()
        self._update_cursor_line()
        self._update_hud_text()
        self.safe_render()

    def set_active_cross_section_index(self, index: int) -> None:
        """Updates the active cross-section index and indicator position."""
        self._active_cross_section_idx = index
        self._update_cursor_line()
        self._update_hud_text()
        self.safe_render()

    def set_window_level(self, window_width: float, window_level: float) -> None:
        """Set Window Width / Window Level."""
        self._window_width = max(1.0, float(window_width))
        self._window_level = float(window_level)
        self.color_map.SetWindow(self._window_width)
        self.color_map.SetLevel(self._window_level)
        self._update_hud_text()
        self.safe_render()

    def set_focal_trough_thickness(self, thickness_mm: float) -> None:
        """Sets focal trough slab thickness and regenerates image."""
        self.panoramic_generator.focal_trough_thickness_mm = max(1.0, float(thickness_mm))
        if self.volume_data and self.arch_curve:
            self.update_panoramic(self.volume_data, self.arch_curve)

    def _reset_camera(self) -> None:
        """Centers camera on the unrolled panoramic image."""
        bounds = self.image_actor.GetBounds()
        if bounds[1] < bounds[0]:
            return
        cx = (bounds[0] + bounds[1]) * 0.5
        cy = (bounds[2] + bounds[3]) * 0.5
        cz = (bounds[4] + bounds[5]) * 0.5
        w = bounds[1] - bounds[0]
        h = bounds[3] - bounds[2]

        self.camera.SetFocalPoint(cx, cy, cz)
        self.camera.SetPosition(cx, cy, cz + 1000.0)
        self.camera.SetViewUp(0.0, 1.0, 0.0)
        self.camera.SetParallelScale(max(1.0, max(w, h) * 0.52))
        self.renderer.ResetCameraClippingRange()

    def _update_cursor_line(self) -> None:
        """Positions vertical orange line at active cross-section column."""
        if not self.arch_curve or len(self.arch_curve.sampled_points) == 0:
            self.cursor_actor.VisibilityOff()
            return

        size = self.vtkWidget.GetRenderWindow().GetSize()
        h_disp = float(max(1, size[1]))

        # Calculate world X coordinate on unrolled image: col_idx * step_size_mm
        col_x = self._active_cross_section_idx * self.arch_curve.step_size_mm
        self.renderer.SetWorldPoint(col_x, 0.0, 0.0, 1.0)
        self.renderer.WorldToDisplay()
        disp_pt = self.renderer.GetDisplayPoint()
        cx = disp_pt[0]

        self.cursor_points.SetPoint(0, cx, 0.0, 0.0)
        self.cursor_points.SetPoint(1, cx, h_disp, 0.0)
        self.cursor_polydata.Modified()

    def _update_hud_text(self) -> None:
        if not self.arch_curve:
            return

        tot_len = self.arch_curve.total_length_mm
        tot_cross = len(self.arch_curve.sampled_points)
        thick = self.panoramic_generator.focal_trough_thickness_mm

        self.corner_annotation.SetText(2, f"PANORAMIC (Curved MPR)\nMode: {self.panoramic_generator.projection_mode.upper()} Trough\nTrough: {thick:.1f} mm")
        self.corner_annotation.SetText(0, f"W: {self._window_width:.0f} L: {self._window_level:.0f}\nCross-Section: #{self._active_cross_section_idx + 1} / {tot_cross}")
        self.corner_annotation.SetText(1, f"Arch Length: {tot_len:.1f} mm\nResolution: {self.arch_curve.step_size_mm:.1f} mm")

    def _on_left_press(self, obj, event) -> None:
        self.signals.focused.emit("panoramic")
        self._is_left_down = True
        pos = self.vtkWidget.GetRenderWindow().GetInteractor().GetEventPosition()
        self._last_mouse_pos = pos

        if self._is_measuring:
            self._handle_measurement_click(pos)
        else:
            self._scrub_cross_section(pos[0])

    def _on_left_release(self, obj, event) -> None:
        self._is_left_down = False

    def _on_right_press(self, obj, event) -> None:
        self.signals.focused.emit("panoramic")
        self._is_right_down = True
        self._last_mouse_pos = self.vtkWidget.GetRenderWindow().GetInteractor().GetEventPosition()

    def _on_right_release(self, obj, event) -> None:
        self._is_right_down = False

    def _on_mouse_move(self, obj, event) -> None:
        pos = self.vtkWidget.GetRenderWindow().GetInteractor().GetEventPosition()
        dx = pos[0] - self._last_mouse_pos[0]
        dy = pos[1] - self._last_mouse_pos[1]
        self._last_mouse_pos = pos

        if self._is_left_down and not self._is_measuring:
            self._scrub_cross_section(pos[0])

        elif self._is_right_down:
            new_ww = max(1.0, self._window_width + dx * 4.0)
            new_wl = self._window_level + dy * 4.0
            self.set_window_level(new_ww, new_wl)
            self.signals.window_level_changed.emit(new_ww, new_wl)

    def _scrub_cross_section(self, disp_x: int) -> None:
        """Converts display X click coordinate into active cross-section index."""
        if not self.arch_curve or len(self.arch_curve.sampled_points) == 0:
            return

        self.renderer.SetDisplayPoint(disp_x, 0.0, 0.0)
        self.renderer.DisplayToWorld()
        world_pt = self.renderer.GetWorldPoint()
        if world_pt[3] == 0.0:
            return

        wx = world_pt[0] / world_pt[3]
        idx = int(round(wx / self.arch_curve.step_size_mm))
        tot_cross = len(self.arch_curve.sampled_points)
        clamped_idx = max(0, min(idx, tot_cross - 1))

        self.set_active_cross_section_index(clamped_idx)
        self.signals.cross_section_selected.emit(clamped_idx)

    def _handle_measurement_click(self, pos: Tuple[int, int]) -> None:
        self.renderer.SetDisplayPoint(pos[0], pos[1], 0.0)
        self.renderer.DisplayToWorld()
        world_pt = self.renderer.GetWorldPoint()
        if world_pt[3] == 0.0:
            return
        wx, wy = world_pt[0] / world_pt[3], world_pt[1] / world_pt[3]
        self._measurement_points.append((wx, wy))

        if len(self._measurement_points) == 2:
            p1, p2 = self._measurement_points[0], self._measurement_points[1]
            dist_mm = math.dist(p1, p2)

            self.measure_points.Reset()
            self.measure_lines.Reset()

            self.renderer.SetWorldPoint(p1[0], p1[1], 0.0, 1.0)
            self.renderer.WorldToDisplay()
            d1 = self.renderer.GetDisplayPoint()

            self.measure_points.InsertNextPoint(d1[0], d1[1], 0.0)
            self.measure_points.InsertNextPoint(pos[0], pos[1], 0.0)

            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, 0)
            line.GetPointIds().SetId(1, 1)
            self.measure_lines.InsertNextCell(line)

            self.measure_polydata.Modified()
            self.measure_actor.VisibilityOn()

            mx = (d1[0] + pos[0]) * 0.5 + 10
            my = (d1[1] + pos[1]) * 0.5 + 10
            self.measure_text.SetInput(f"{dist_mm:.2f} mm")
            self.measure_text.SetDisplayPosition(int(mx), int(my))
            self.measure_text.VisibilityOn()

            self.safe_render()
            self.signals.measurement_completed.emit(dist_mm)
            self._is_measuring = False

    def safe_render(self) -> None:
        if hasattr(self, 'vtkWidget') and self.vtkWidget is not None:
            rw = self.vtkWidget.GetRenderWindow()
            if rw and self.isVisible() and self.width() > 10 and self.height() > 10:
                try:
                    rw.Render()
                except Exception:
                    pass

    def cleanup(self) -> None:
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
            self._update_cursor_line()
            self.safe_render()
