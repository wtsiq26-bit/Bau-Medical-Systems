"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Module: rendering/cross_section_view.py

Interactive Transverse Bucco-Lingual Cross-Sectional Viewport for Dental Implant Planning.
Features:
- Real-time 60 FPS reslicing along the normal plane of the dental arch (vtkImageReslice).
- Anatomical Buccal (B) vs. Lingual (L) and Superior (S) vs. Inferior (I) edge markers.
- High-precision Caliper for measuring cortical bone ridge width, bone height, and nerve safety margin.
- 4-Corner OSD PACS HUD (Slice #, Arc position mm, Matrix size, WW/WL).
- Synchronized Window/Level contrast adjustment.
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
from dental.panoramic_mpr import CrossSectionManager, DentalArchCurve
from dental.implant_simulator import ImplantManager, DentalImplant


class CrossSectionSignals(QObject):
    """Signals emitted from the Cross-Section Viewport."""
    window_level_changed = Signal(float, float)
    measurement_completed = Signal(float)
    focused = Signal(str)  # 'cross_section'


class CrossSectionView(QFrame):
    """
    Transverse Bucco-Lingual Cross-Sectional Viewport with Virtual Implant Projection.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.signals = CrossSectionSignals()
        self.volume_data: Optional[VolumeData] = None
        self.cross_section_mgr: Optional[CrossSectionManager] = None
        self.implant_manager: Optional[ImplantManager] = None

        self._window_width: float = 2500.0
        self._window_level: float = 500.0
        self._is_measuring: bool = False
        self._measurement_points: list[Tuple[float, float, float]] = []

        self._accent_color = (0.0, 0.86, 0.91)
        self._secondary_color = (0.72, 0.78, 0.89)

        self.setObjectName("cross_section_view")
        self.setFrameStyle(QFrame.NoFrame)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 1. VTK Interactor
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtkWidget)

        # 2. Renderer & Parallel Camera
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.039, 0.059, 0.071)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)

        self.camera = self.renderer.GetActiveCamera()
        self.camera.ParallelProjectionOn()

        # 3. Reslice Pipeline
        self._setup_reslice_pipeline()

        # 4. Implant Projection Overlays
        self._setup_implant_overlays()

        # 5. Corner HUD
        self._setup_hud()

        # 6. Measurement Pipeline
        self._setup_measurement_pipeline()

        # 7. Interactor Style
        self._setup_interactor_style()

    def _setup_reslice_pipeline(self) -> None:
        self.reslice = vtk.vtkImageReslice()
        self.reslice.SetOutputDimensionality(2)
        self.reslice.SetInterpolationModeToLinear()
        self.reslice.SetAutoCropOutput(True)
        self.reslice.SetBackgroundLevel(-1024.0)

        dummy_data = vtk.vtkImageData()
        dummy_data.SetDimensions(2, 2, 2)
        dummy_scalars = vtk.vtkShortArray()
        dummy_scalars.SetNumberOfTuples(8)
        dummy_scalars.FillValue(-1024)
        dummy_data.GetPointData().SetScalars(dummy_scalars)
        self.reslice.SetInputData(dummy_data)

        self.color_map = vtk.vtkImageMapToWindowLevelColors()
        self.color_map.SetInputConnection(self.reslice.GetOutputPort())
        self.color_map.SetWindow(self._window_width)
        self.color_map.SetLevel(self._window_level)
        self.color_map.SetOutputFormatToRGB()

        self.image_actor = vtk.vtkImageActor()
        self.image_actor.GetMapper().SetInputConnection(self.color_map.GetOutputPort())
        self.image_actor.InterpolateOn()
        self.renderer.AddActor(self.image_actor)

    def _setup_implant_overlays(self) -> None:
        """Configures 2D projected implant outline and safety sleeve actors."""
        self.implant_polydata = vtk.vtkPolyData()
        self.implant_mapper_2d = vtk.vtkPolyDataMapper2D()
        self.implant_mapper_2d.SetInputData(self.implant_polydata)

        self.implant_actor_2d = vtk.vtkActor2D()
        self.implant_actor_2d.SetMapper(self.implant_mapper_2d)
        self.implant_actor_2d.GetProperty().SetColor(0.0, 0.95, 0.5)
        self.implant_actor_2d.GetProperty().SetLineWidth(2.5)
        self.renderer.AddViewProp(self.implant_actor_2d)

        # Safety Sleeve
        self.sleeve_polydata_2d = vtk.vtkPolyData()
        self.sleeve_mapper_2d = vtk.vtkPolyDataMapper2D()
        self.sleeve_mapper_2d.SetInputData(self.sleeve_polydata_2d)

        self.sleeve_actor_2d = vtk.vtkActor2D()
        self.sleeve_actor_2d.SetMapper(self.sleeve_mapper_2d)
        self.sleeve_actor_2d.GetProperty().SetColor(0.0, 0.86, 0.91)
        self.sleeve_actor_2d.GetProperty().SetLineWidth(1.2)
        self.renderer.AddViewProp(self.sleeve_actor_2d)

    def _setup_hud(self) -> None:
        self.corner_annotation = vtk.vtkCornerAnnotation()
        self.corner_annotation.SetMaximumFontSize(12)
        self.corner_annotation.SetMinimumFontSize(10)
        self.corner_annotation.GetTextProperty().SetFontFamilyToCourier()
        self.corner_annotation.GetTextProperty().SetColor(0.87, 0.89, 0.91)
        self.corner_annotation.GetTextProperty().ShadowOn()

        # Anatomical Edge Markers
        self.corner_annotation.SetText(4, "B")  # Left: Buccal
        self.corner_annotation.SetText(5, "L")  # Right: Lingual
        self.corner_annotation.SetText(6, "S")  # Top: Superior
        self.corner_annotation.SetText(7, "I")  # Bottom: Inferior

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

    def set_volume_and_manager(self, volume: VolumeData, manager: CrossSectionManager) -> None:
        """Attaches volume and cross-section manager."""
        self.volume_data = volume
        self.cross_section_mgr = manager
        self.reslice.SetInputData(volume.vtk_image_data)
        self.update_slice()

    def set_implant_manager(self, mgr: ImplantManager) -> None:
        """Attaches Dental Implant Manager and connects signals."""
        self.implant_manager = mgr
        mgr.signals.implant_modified.connect(lambda *_: self._update_implant_overlays())
        mgr.signals.implant_selected.connect(lambda *_: self._update_implant_overlays())
        mgr.signals.safety_status_changed.connect(lambda *_: self._update_implant_overlays())
        self._update_implant_overlays()

    def update_slice(self) -> None:
        """Reslices the volume at active cross-section plane and resets camera."""
        if not self.volume_data or not self.cross_section_mgr:
            return

        matrix = self.cross_section_mgr.get_reslice_matrix_for_index(self.cross_section_mgr.active_index)
        if matrix is None:
            return

        self.reslice.SetResliceAxes(matrix)
        self.reslice.Update()
        self.color_map.Update()

        self._reset_camera()
        self._update_implant_overlays()
        self._update_hud_text()
        self.safe_render()

    def _update_implant_overlays(self) -> None:
        """Projects active implant outline and central axis onto the active bucco-lingual slice."""
        if not self.implant_manager or not self.cross_section_mgr or not self.volume_data:
            return

        implant = self.implant_manager.get_active_implant()
        if not implant:
            self.implant_polydata.Reset()
            self.implant_polydata.Modified()
            self.safe_render()
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
            return d[0], d[1], p_local[2]

        top = implant.get_platform_center_world()
        apex = implant.get_apical_tip_world()

        d_top = to_disp(*top)
        d_apex = to_disp(*apex)

        # Draw projected outline if within 15mm of this cross-section plane
        if abs(d_top[2]) < 15.0:
            pts = vtk.vtkPoints()
            lines = vtk.vtkCellArray()

            r_top_pix = (implant.diameter_mm * 0.5) / self.volume_data.spacing[0] * 1.5
            r_bot_pix = r_top_pix * 0.7

            pts.InsertNextPoint(d_top[0] - r_top_pix, d_top[1], 0.0)
            pts.InsertNextPoint(d_top[0] + r_top_pix, d_top[1], 0.0)
            pts.InsertNextPoint(d_apex[0] + r_bot_pix, d_apex[1], 0.0)
            pts.InsertNextPoint(d_apex[0] - r_bot_pix, d_apex[1], 0.0)

            # Central axis
            pts.InsertNextPoint(d_top[0], d_top[1], 0.0)
            pts.InsertNextPoint(d_apex[0], d_apex[1], 0.0)

            for i in range(4):
                l = vtk.vtkLine()
                l.GetPointIds().SetId(0, i)
                l.GetPointIds().SetId(1, (i + 1) % 4)
                lines.InsertNextCell(l)

            al = vtk.vtkLine()
            al.GetPointIds().SetId(0, 4)
            al.GetPointIds().SetId(1, 5)
            lines.InsertNextCell(al)

            self.implant_polydata.SetPoints(pts)
            self.implant_polydata.SetLines(lines)
            self.implant_polydata.Modified()

            # Set color based on safety state
            if implant.safety_state.value == "safe":
                self.implant_actor_2d.GetProperty().SetColor(0.0, 0.95, 0.5)
            elif implant.safety_state.value == "warning":
                self.implant_actor_2d.GetProperty().SetColor(1.0, 0.84, 0.0)
            else:
                self.implant_actor_2d.GetProperty().SetColor(1.0, 0.05, 0.15)
        else:
            self.implant_polydata.Reset()
            self.implant_polydata.Modified()

        self.safe_render()

    def set_cross_section_index(self, index: int) -> None:
        """Switches active cross-section slice index."""
        if self.cross_section_mgr:
            self.cross_section_mgr.set_active_index(index)
            self.update_slice()

    def set_window_level(self, window_width: float, window_level: float) -> None:
        self._window_width = max(1.0, float(window_width))
        self._window_level = float(window_level)
        self.color_map.SetWindow(self._window_width)
        self.color_map.SetLevel(self._window_level)
        self._update_hud_text()
        self.safe_render()

    def start_measurement(self) -> None:
        self._is_measuring = True
        self._measurement_points.clear()
        self.measure_actor.VisibilityOff()
        self.measure_text.VisibilityOff()
        self.safe_render()

    def clear_measurement(self) -> None:
        self._is_measuring = False
        self._measurement_points.clear()
        self.measure_actor.VisibilityOff()
        self.measure_text.VisibilityOff()
        self.safe_render()

    def _reset_camera(self) -> None:
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
        self.camera.SetParallelScale(max(1.0, max(w, h) * 0.55))
        self.renderer.ResetCameraClippingRange()

    def _update_hud_text(self) -> None:
        if not self.cross_section_mgr:
            return

        cur = self.cross_section_mgr.active_index + 1
        tot = self.cross_section_mgr.total_cross_sections
        step = self.cross_section_mgr.arch_curve.step_size_mm
        arc_pos = (cur - 1) * step

        self.corner_annotation.SetText(2, f"CROSS-SECTION #{cur} / {tot}\nPlane: BUCCO-LINGUAL\nArc Pos: {arc_pos:.1f} mm")
        self.corner_annotation.SetText(3, "BAU MEDICAL SYSTEMS\nImplant Planning View")
        self.corner_annotation.SetText(0, f"W: {self._window_width:.0f} L: {self._window_level:.0f}\nStep: {step:.1f} mm")
        self.corner_annotation.SetText(1, f"Slice Spacing: {step:.1f} mm\nZoom: 100%")

    def _on_left_press(self, obj, event) -> None:
        self.signals.focused.emit("cross_section")
        self._is_left_down = True
        pos = self.vtkWidget.GetRenderWindow().GetInteractor().GetEventPosition()
        self._last_mouse_pos = pos

        if self._is_measuring:
            self._handle_measurement_click(pos)

    def _on_left_release(self, obj, event) -> None:
        self._is_left_down = False

    def _on_right_press(self, obj, event) -> None:
        self.signals.focused.emit("cross_section")
        self._is_right_down = True
        self._last_mouse_pos = self.vtkWidget.GetRenderWindow().GetInteractor().GetEventPosition()

    def _on_right_release(self, obj, event) -> None:
        self._is_right_down = False

    def _on_mouse_move(self, obj, event) -> None:
        pos = self.vtkWidget.GetRenderWindow().GetInteractor().GetEventPosition()
        dx = pos[0] - self._last_mouse_pos[0]
        dy = pos[1] - self._last_mouse_pos[1]
        self._last_mouse_pos = pos

        if self._is_right_down:
            new_ww = max(1.0, self._window_width + dx * 4.0)
            new_wl = self._window_level + dy * 4.0
            self.set_window_level(new_ww, new_wl)
            self.signals.window_level_changed.emit(new_ww, new_wl)

    def _handle_measurement_click(self, pos: Tuple[int, int]) -> None:
        self.renderer.SetDisplayPoint(pos[0], pos[1], 0.0)
        self.renderer.DisplayToWorld()
        world_pt = self.renderer.GetWorldPoint()
        if world_pt[3] == 0.0:
            return

        wx, wy = world_pt[0] / world_pt[3], world_pt[1] / world_pt[3]

        reslice_matrix = self.reslice.GetResliceAxes()
        if reslice_matrix is None:
            return

        p_in = [wx, wy, 0.0, 1.0]
        p_out = [0.0, 0.0, 0.0, 0.0]
        reslice_matrix.MultiplyPoint(p_in, p_out)
        self._measurement_points.append((p_out[0], p_out[1], p_out[2]))

        if len(self._measurement_points) == 2:
            p1, p2 = self._measurement_points[0], self._measurement_points[1]
            dist_mm = math.dist(p1, p2)

            inv_matrix = vtk.vtkMatrix4x4()
            vtk.vtkMatrix4x4.Invert(reslice_matrix, inv_matrix)

            p1_in = [p1[0], p1[1], p1[2], 1.0]
            p1_local = [0.0, 0.0, 0.0, 0.0]
            inv_matrix.MultiplyPoint(p1_in, p1_local)
            self.renderer.SetWorldPoint(p1_local[0], p1_local[1], 0.0, 1.0)
            self.renderer.WorldToDisplay()
            d1 = self.renderer.GetDisplayPoint()

            self.measure_points.Reset()
            self.measure_lines.Reset()

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
