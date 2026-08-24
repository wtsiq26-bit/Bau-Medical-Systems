"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Module: ui/viewport_grid.py

Commercial PACS Multi-Viewport Grid Manager with Active Viewport Highlighting,
Multiplanar Orthogonal MPR, Curved Panoramic MPR, and Bucco-Lingual Cross-Sectional Views.
Features:
- Dynamic Layout Modes:
  * `2x2`: Standard 4-viewport PACS grid (Axial, Coronal, Sagittal, 3D).
  * `1x1`: Single maximized active viewport.
  * `1+3`: Master Viewport (left/top 2x2 span) + 3 thumbnail slice insets.
  * `3d`: Fullscreen 3D Volume raycaster.
  * `implant`: Specialized Dental Implant Planning Layout (Axial Arch + Panoramic + Cross-Section + 3D Volume).
- Active Focused Viewport Highlighting with Electric Cyan border (#00dbe9, 2px).
- Bidirectional synchronized 3D crosshair and cross-section navigation.
- Synchronized Window Width / Window Level (WW/WL) contrast adjustments.
- Mandibular Nerve Tracing & Dental Arch Curved MPR integration.
"""

from __future__ import annotations
from typing import Optional, Dict, Tuple, List
from PySide6.QtCore import QObject, Signal, Qt, QSize
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QSizePolicy
)
from PySide6.QtGui import QIcon, QFont, QColor

from core.volume_data import VolumeData
from rendering.mpr_slice_view import MPRSliceView
from rendering.volume_view import VolumeView
from rendering.panoramic_view import PanoramicView
from rendering.cross_section_view import CrossSectionView
from dental.nerve_tracer import NerveTracer, NerveChannel
from dental.panoramic_mpr import DentalArchCurve, CrossSectionManager
from dental.implant_simulator import ImplantManager, DentalImplant, ImplantSafetyState, STANDARD_IMPLANT_PRESETS


class ViewportGridSignals(QObject):
    """Signals aggregated from all viewports."""
    crosshair_moved = Signal(float, float, float)
    slice_changed = Signal(str, int, int)
    window_level_changed = Signal(float, float)
    hu_inspected = Signal(float, float, float, float)
    measurement_completed = Signal(str, float)
    nerve_updated = Signal(str, int, float)
    arch_updated = Signal(int, float)            # num_points, total_length_mm
    cross_section_changed = Signal(int, int)     # active_index, total_slices
    implant_safety_changed = Signal(str, float, str, str) # id, dist, state, nerve
    active_viewport_changed = Signal(str)        # 'axial' | 'coronal' | 'sagittal' | '3d' | 'panoramic' | 'cross_section'


class ViewportContainer(QFrame):
    """Container frame wrapping an individual viewport with a modern header bar and focus border."""

    def __init__(
        self,
        viewport_id: str,
        title: str,
        color_tag: str,
        content_widget: QWidget,
        parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.viewport_id = viewport_id
        self.title = title
        self.color_tag = color_tag
        self.content_widget = content_widget
        self._is_active = False

        self.setObjectName("viewport_card")
        self._apply_border_style()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)

        # Header bar
        self.header = QFrame(self)
        self.header.setObjectName("viewport_header")
        self.header.setFixedHeight(26)
        self.header.setStyleSheet("""
            QFrame#viewport_header {
                background-color: #171c1f;
                border-bottom: 1px solid #262b2e;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
            }
        """)

        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(8, 0, 6, 0)
        header_layout.setSpacing(6)

        self.color_badge = QLabel()
        self.color_badge.setFixedSize(8, 8)
        self.color_badge.setStyleSheet(f"background-color: {color_tag}; border-radius: 4px;")
        header_layout.addWidget(self.color_badge)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("color: #dfe3e7; font-family: Inter; font-size: 11px; font-weight: 600; letter-spacing: 0.05em;")
        header_layout.addWidget(self.title_label)

        header_layout.addStretch()

        self.btn_maximize = QPushButton("⛶")
        self.btn_maximize.setToolTip("Maximize Viewport")
        self.btn_maximize.setFixedSize(20, 18)
        self.btn_maximize.setCursor(Qt.PointingHandCursor)
        self.btn_maximize.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #849495;
                border: none;
                font-size: 12px;
                font-weight: bold;
                padding: 0;
            }
            QPushButton:hover {
                color: #00dbe9;
                background-color: #262b2e;
                border-radius: 2px;
            }
        """)
        header_layout.addWidget(self.btn_maximize)

        layout.addWidget(self.header)
        layout.addWidget(self.content_widget, 1)

    def set_active(self, active: bool) -> None:
        self._is_active = active
        self._apply_border_style()

    def _apply_border_style(self) -> None:
        if self._is_active:
            self.setStyleSheet("""
                QFrame#viewport_card {
                    background-color: #0a0f12;
                    border: 2px solid #00dbe9;
                    border-radius: 4px;
                }
            """)
        else:
            self.setStyleSheet("""
                QFrame#viewport_card {
                    background-color: #0a0f12;
                    border: 1px solid #1b2023;
                    border-radius: 4px;
                }
                QFrame#viewport_card:hover {
                    border: 1px solid #3b494b;
                }
            """)


class ViewportGrid(QWidget):
    """
    Commercial Medical PACS Multi-Viewport Grid.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.signals = ViewportGridSignals()
        self.volume_data: Optional[VolumeData] = None
        self.nerve_tracer = NerveTracer()
        self.dental_arch = DentalArchCurve(step_size_mm=0.8)
        self.cross_section_mgr = CrossSectionManager(self.dental_arch)
        self.implant_manager = ImplantManager()

        self.active_viewport_id = "axial"
        self._current_layout_mode = "2x2"

        # Main Grid Layout
        self.grid_layout = QGridLayout(self)
        self.grid_layout.setContentsMargins(4, 4, 4, 4)
        self.grid_layout.setSpacing(4)

        # 1. Instantiate Viewports
        self.axial_view = MPRSliceView(plane_type="axial", parent=self)
        self.coronal_view = MPRSliceView(plane_type="coronal", parent=self)
        self.sagittal_view = MPRSliceView(plane_type="sagittal", parent=self)
        self.volume_view = VolumeView(parent=self)
        self.panoramic_view = PanoramicView(parent=self)
        self.cross_section_view = CrossSectionView(parent=self)

        # 2. Attach NerveTracer, DentalArch, and ImplantManager
        self.axial_view.set_nerve_tracer(self.nerve_tracer)
        self.coronal_view.set_nerve_tracer(self.nerve_tracer)
        self.sagittal_view.set_nerve_tracer(self.nerve_tracer)
        self.volume_view.set_nerve_tracer(self.nerve_tracer)
        self.volume_view.set_implant_manager(self.implant_manager)

        self.axial_view.set_dental_arch(self.dental_arch, self.cross_section_mgr)
        self.cross_section_view.set_implant_manager(self.implant_manager)

        # 3. Wrap in ViewportContainers
        self.axial_container = ViewportContainer("axial", "AXIAL (Z)", "#00dbe9", self.axial_view, self)
        self.coronal_container = ViewportContainer("coronal", "CORONAL (Y)", "#4ade80", self.coronal_view, self)
        self.sagittal_container = ViewportContainer("sagittal", "SAGITTAL (X)", "#f59e0b", self.sagittal_view, self)
        self.volume_container = ViewportContainer("3d", "3D VOLUME (GPU)", "#a855f7", self.volume_view, self)
        self.panoramic_container = ViewportContainer("panoramic", "PANORAMIC (Curved MPR)", "#00dbe9", self.panoramic_view, self)
        self.cross_section_container = ViewportContainer("cross_section", "CROSS-SECTION (Bucco-Lingual)", "#ff9500", self.cross_section_view, self)

        self.containers: Dict[str, ViewportContainer] = {
            "axial": self.axial_container,
            "coronal": self.coronal_container,
            "sagittal": self.sagittal_container,
            "3d": self.volume_container,
            "panoramic": self.panoramic_container,
            "cross_section": self.cross_section_container,
        }

        # 4. Connect Maximize Buttons
        self.axial_container.btn_maximize.clicked.connect(lambda: self.toggle_maximize("axial"))
        self.coronal_container.btn_maximize.clicked.connect(lambda: self.toggle_maximize("coronal"))
        self.sagittal_container.btn_maximize.clicked.connect(lambda: self.toggle_maximize("sagittal"))
        self.volume_container.btn_maximize.clicked.connect(lambda: self.toggle_maximize("3d"))
        self.panoramic_container.btn_maximize.clicked.connect(lambda: self.toggle_maximize("panoramic"))
        self.cross_section_container.btn_maximize.clicked.connect(lambda: self.toggle_maximize("cross_section"))

        # 5. Connect Focus Signals
        self.axial_view.signals.focused.connect(lambda: self.set_active_viewport("axial"))
        self.coronal_view.signals.focused.connect(lambda: self.set_active_viewport("coronal"))
        self.sagittal_view.signals.focused.connect(lambda: self.set_active_viewport("sagittal"))
        self.panoramic_view.signals.focused.connect(lambda: self.set_active_viewport("panoramic"))
        self.cross_section_view.signals.focused.connect(lambda: self.set_active_viewport("cross_section"))

        # 6. Apply Default 2x2 Layout
        self.set_layout_mode("2x2")
        self.set_active_viewport("axial")

        # 7. Connect Synchronized Signals
        self._connect_viewport_signals()

    def _connect_viewport_signals(self) -> None:
        """Wires real-time cross-viewport synchronization."""
        # Crosshair movement
        self.axial_view.signals.crosshair_moved.connect(self._on_crosshair_moved)
        self.coronal_view.signals.crosshair_moved.connect(self._on_crosshair_moved)
        self.sagittal_view.signals.crosshair_moved.connect(self._on_crosshair_moved)

        # Slice index broadcast
        self.axial_view.signals.slice_changed.connect(self.signals.slice_changed.emit)
        self.coronal_view.signals.slice_changed.connect(self.signals.slice_changed.emit)
        self.sagittal_view.signals.slice_changed.connect(self.signals.slice_changed.emit)

        # Window/Level synchronization
        self.axial_view.signals.window_level_changed.connect(self._on_window_level_changed)
        self.coronal_view.signals.window_level_changed.connect(self._on_window_level_changed)
        self.sagittal_view.signals.window_level_changed.connect(self._on_window_level_changed)
        self.panoramic_view.signals.window_level_changed.connect(self._on_window_level_changed)
        self.cross_section_view.signals.window_level_changed.connect(self._on_window_level_changed)

        # HU inspection
        self.axial_view.signals.hu_inspected.connect(self.signals.hu_inspected.emit)
        self.coronal_view.signals.hu_inspected.connect(self.signals.hu_inspected.emit)
        self.sagittal_view.signals.hu_inspected.connect(self.signals.hu_inspected.emit)

        # Measurements
        self.axial_view.signals.measurement_completed.connect(self.signals.measurement_completed.emit)
        self.coronal_view.signals.measurement_completed.connect(self.signals.measurement_completed.emit)
        self.sagittal_view.signals.measurement_completed.connect(self.signals.measurement_completed.emit)
        self.panoramic_view.signals.measurement_completed.connect(lambda v: self.signals.measurement_completed.emit("distance", v))
        self.cross_section_view.signals.measurement_completed.connect(lambda v: self.signals.measurement_completed.emit("distance", v))

        # Nerve tracing
        self.nerve_tracer.signals.nerve_updated.connect(self._on_nerve_updated)
        self.axial_view.signals.nerve_point_placed.connect(lambda *_: self._refresh_all_nerve_views())
        self.coronal_view.signals.nerve_point_placed.connect(lambda *_: self._refresh_all_nerve_views())
        self.sagittal_view.signals.nerve_point_placed.connect(lambda *_: self._refresh_all_nerve_views())

        # Dental Arch & Panoramic Navigation
        self.axial_view.signals.arch_point_placed.connect(self._on_arch_point_placed)
        self.panoramic_view.signals.cross_section_selected.connect(self.set_cross_section_index)

        # Implant Signals
        self.implant_manager.signals.safety_status_changed.connect(self.signals.implant_safety_changed.emit)

    def set_active_viewport(self, viewport_id: str) -> None:
        """Sets active focus highlight on selected viewport."""
        self.active_viewport_id = viewport_id
        for vid, container in self.containers.items():
            container.set_active(vid == viewport_id)
        self.signals.active_viewport_changed.emit(viewport_id)

    def set_layout_mode(self, mode: str) -> None:
        """
        Dynamically changes multi-viewport layout:
        - `2x2`: Standard 4-grid (Axial, Coronal, Sagittal, 3D).
        - `1x1`: Active viewport full screen.
        - `1+3`: Master view on left + 3 stacked views on right.
        - `3d`: 3D Volume full screen.
        - `implant`: Specialized Dental Implant Planning Layout (Axial, Panoramic, Cross-Section, 3D).
        """
        self._current_layout_mode = mode

        # Clear existing layout
        for container in self.containers.values():
            self.grid_layout.removeWidget(container)
            container.hide()

        if mode == "2x2":
            self.grid_layout.addWidget(self.axial_container, 0, 0)
            self.grid_layout.addWidget(self.coronal_container, 0, 1)
            self.grid_layout.addWidget(self.sagittal_container, 1, 0)
            self.grid_layout.addWidget(self.volume_container, 1, 1)
            for c in [self.axial_container, self.coronal_container, self.sagittal_container, self.volume_container]:
                c.show()
                c.btn_maximize.setText("⛶")

        elif mode == "implant":
            # 1. Top-Left: Axial Arch View
            self.grid_layout.addWidget(self.axial_container, 0, 0)
            # 2. Top-Right: Panoramic View
            self.grid_layout.addWidget(self.panoramic_container, 0, 1)
            # 3. Bottom-Left: Cross-Section View
            self.grid_layout.addWidget(self.cross_section_container, 1, 0)
            # 4. Bottom-Right: 3D Volume Raycaster
            self.grid_layout.addWidget(self.volume_container, 1, 1)

            for c in [self.axial_container, self.panoramic_container, self.cross_section_container, self.volume_container]:
                c.show()
                c.btn_maximize.setText("⛶")

            # Regenerate panoramic & cross-section if needed
            if self.volume_data:
                if len(self.dental_arch.sampled_points) < 2:
                    self.auto_fit_dental_arch()
                else:
                    self._update_panoramic_and_cross_sections()

        elif mode == "1x1":
            target = self.containers.get(self.active_viewport_id, self.axial_container)
            self.grid_layout.addWidget(target, 0, 0, 2, 2)
            target.show()
            target.btn_maximize.setText("❐")

        elif mode == "1+3":
            master = self.containers.get(self.active_viewport_id, self.axial_container)
            self.grid_layout.addWidget(master, 0, 0, 3, 2)
            master.show()

            standard_set = [self.axial_container, self.coronal_container, self.sagittal_container, self.volume_container]
            others = [c for c in standard_set if c.viewport_id != self.active_viewport_id]
            for idx, c in enumerate(others):
                self.grid_layout.addWidget(c, idx, 2, 1, 1)
                c.show()

        elif mode == "3d":
            self.grid_layout.addWidget(self.volume_container, 0, 0, 2, 2)
            self.volume_container.show()
            self.volume_container.btn_maximize.setText("❐")

        self._refresh_all_views()

    def toggle_maximize(self, viewport_id: str) -> None:
        """Toggles maximize/restore state for specific viewport."""
        if self._current_layout_mode == "1x1" and self.active_viewport_id == viewport_id:
            self.set_layout_mode("2x2")
        else:
            self.set_active_viewport(viewport_id)
            self.set_layout_mode("1x1")

    def set_volume_data(self, volume: VolumeData) -> None:
        """Distribute VolumeData to all viewports and initialize dental arch."""
        self.volume_data = volume
        self.axial_view.set_volume_data(volume)
        self.coronal_view.set_volume_data(volume)
        self.sagittal_view.set_volume_data(volume)
        self.volume_view.set_volume_data(volume)

        # Auto-fit mandibular arch parabola
        self.auto_fit_dental_arch()

    def set_active_tool(self, tool_name: str) -> None:
        """Propagates active navigation/measurement tool across all slice views."""
        self.axial_view.set_active_tool(tool_name)
        self.coronal_view.set_active_tool(tool_name)
        self.sagittal_view.set_active_tool(tool_name)

    def set_invert_colors(self, invert: bool) -> None:
        """Invert grayscale HU mapping on all 2D slices."""
        self.axial_view.set_invert_colors(invert)
        self.coronal_view.set_invert_colors(invert)
        self.sagittal_view.set_invert_colors(invert)

    def start_cine(self, fps: int = 15) -> None:
        """Starts cine loop on the active viewport."""
        if self.active_viewport_id == "axial":
            self.axial_view.start_cine(fps)
        elif self.active_viewport_id == "coronal":
            self.coronal_view.start_cine(fps)
        elif self.active_viewport_id == "sagittal":
            self.sagittal_view.start_cine(fps)

    def stop_cine(self) -> None:
        """Stops all running cine loops."""
        self.axial_view.stop_cine()
        self.coronal_view.stop_cine()
        self.sagittal_view.stop_cine()

    # --------------------------------------------------------------------------
    # Dental Arch & Panoramic Methods
    # --------------------------------------------------------------------------

    def draw_dental_arch(self, enabled: bool) -> None:
        """Toggle manual dental arch curve drawing mode on Axial view."""
        self.axial_view.set_arch_drawing_mode(enabled)

    def auto_fit_dental_arch(self) -> None:
        """Auto-fits parabolic dental arch to mandibular jaw bounds."""
        if self.volume_data is None:
            return
        cz = self.volume_data.get_center()[2]
        self.dental_arch.auto_fit_parabola(self.volume_data, z_world=cz - 10.0)
        self.axial_view._update_arch_slice_overlays()
        self._update_panoramic_and_cross_sections()
        self.signals.arch_updated.emit(len(self.dental_arch.seed_points), self.dental_arch.total_length_mm)

    def clear_dental_arch(self) -> None:
        """Clears the dental arch curve."""
        self.dental_arch.clear()
        self.axial_view._update_arch_slice_overlays()
        self.axial_view.safe_render()
        self.signals.arch_updated.emit(0, 0.0)

    def set_focal_trough_thickness(self, thickness_mm: float) -> None:
        """Updates focal trough thickness and updates panoramic view."""
        self.panoramic_view.set_focal_trough_thickness(thickness_mm)

    def set_cross_section_index(self, index: int) -> None:
        """Sets the active cross-section index and updates slice views."""
        self.cross_section_mgr.set_active_index(index)
        self.cross_section_view.set_cross_section_index(index)
        self.panoramic_view.set_active_cross_section_index(index)
        self.axial_view._update_arch_slice_overlays()
        self.axial_view.safe_render()
        self.signals.cross_section_changed.emit(index, self.cross_section_mgr.total_cross_sections)

    def _on_arch_point_placed(self, x: float, y: float, z: float) -> None:
        """Triggered when a seed point is placed on the axial slice."""
        self._update_panoramic_and_cross_sections()
        self.signals.arch_updated.emit(len(self.dental_arch.seed_points), self.dental_arch.total_length_mm)

    def _update_panoramic_and_cross_sections(self) -> None:
        """Updates Panoramic and Cross-Section viewports from current arch curve."""
        if self.volume_data is None or len(self.dental_arch.sampled_points) < 2:
            return

        self.panoramic_view.update_panoramic(self.volume_data, self.dental_arch)
        self.cross_section_view.set_volume_and_manager(self.volume_data, self.cross_section_mgr)
        mid_idx = len(self.dental_arch.sampled_points) // 2
        self.set_cross_section_index(mid_idx)

    # --------------------------------------------------------------------------
    # Synchronization & Event Handlers
    # --------------------------------------------------------------------------

    def _on_crosshair_moved(self, x: float, y: float, z: float) -> None:
        sender = self.sender()
        if sender != self.axial_view.signals:
            self.axial_view.set_world_position(x, y, z)
        if sender != self.coronal_view.signals:
            self.coronal_view.set_world_position(x, y, z)
        if sender != self.sagittal_view.signals:
            self.sagittal_view.set_world_position(x, y, z)
        self.signals.crosshair_moved.emit(x, y, z)

    def _on_window_level_changed(self, ww: float, wl: float) -> None:
        sender = self.sender()
        if sender != self.axial_view.signals:
            self.axial_view.set_window_level(ww, wl)
        if sender != self.coronal_view.signals:
            self.coronal_view.set_window_level(ww, wl)
        if sender != self.sagittal_view.signals:
            self.sagittal_view.set_window_level(ww, wl)
        if sender != self.panoramic_view.signals:
            self.panoramic_view.set_window_level(ww, wl)
        if sender != self.cross_section_view.signals:
            self.cross_section_view.set_window_level(ww, wl)
        self.signals.window_level_changed.emit(ww, wl)

    def _on_nerve_updated(self, channel: str, count: int, length_mm: float) -> None:
        self._refresh_all_nerve_views()
        self.signals.nerve_updated.emit(channel, count, length_mm)

    def _refresh_all_nerve_views(self) -> None:
        self.axial_view._update_nerve_slice_overlays()
        self.axial_view.safe_render()
        self.coronal_view._update_nerve_slice_overlays()
        self.coronal_view.safe_render()
        self.sagittal_view._update_nerve_slice_overlays()
        self.sagittal_view.safe_render()
        self.volume_view.safe_render()

    def _refresh_all_views(self) -> None:
        self.axial_view.safe_render()
        self.coronal_view.safe_render()
        self.sagittal_view.safe_render()
        self.volume_view.safe_render()
        self.panoramic_view.safe_render()
        self.cross_section_view.safe_render()

    def set_nerve_drawing_mode(self, enabled: bool, channel_name: str = "left") -> None:
        ch = NerveChannel.LEFT if channel_name.lower() == "left" else NerveChannel.RIGHT
        self.nerve_tracer.set_active_channel(ch)
        self.nerve_tracer.set_drawing_mode(enabled)
        self.axial_view.set_nerve_drawing_mode(enabled)
        self.coronal_view.set_nerve_drawing_mode(enabled)
        self.sagittal_view.set_nerve_drawing_mode(enabled)

    def undo_nerve_point(self) -> None:
        self.nerve_tracer.undo_last_point()

    def clear_nerve(self) -> None:
        self.nerve_tracer.clear_nerve()

    def set_nerve_diameter(self, diameter_mm: float) -> None:
        radius = diameter_mm * 0.5
        self.nerve_tracer.set_nerve_radius(radius)

    def set_global_window_level(self, ww: float, wl: float) -> None:
        self.axial_view.set_window_level(ww, wl)
        self.coronal_view.set_window_level(ww, wl)
        self.sagittal_view.set_window_level(ww, wl)
        self.panoramic_view.set_window_level(ww, wl)
        self.cross_section_view.set_window_level(ww, wl)

    def set_crosshair_visible(self, visible: bool) -> None:
        self.axial_view.set_crosshair_visible(visible)
        self.coronal_view.set_crosshair_visible(visible)
        self.sagittal_view.set_crosshair_visible(visible)

    def reset_all_views(self) -> None:
        if self.volume_data is None:
            return
        center = self.volume_data.get_center()
        self.axial_view.set_world_position(*center, force_reslice=True)
        self.coronal_view.set_world_position(*center, force_reslice=True)
        self.sagittal_view.set_world_position(*center, force_reslice=True)
        self.axial_view._reset_camera()
        self.coronal_view._reset_camera()
        self.sagittal_view._reset_camera()
        self.volume_view.reset_camera()
        if len(self.dental_arch.sampled_points) >= 2:
            self._update_panoramic_and_cross_sections()

    # --------------------------------------------------------------------------
    # Virtual Dental Implant Planning & Safety Methods
    # --------------------------------------------------------------------------

    def add_implant(
        self,
        tooth_number: int = 19,
        diameter_mm: float = 4.0,
        length_mm: float = 11.5
    ) -> str:
        """Adds a new virtual implant placed at the active cross-section bone ridge."""
        pos = self.cross_section_mgr.get_active_world_position()
        if pos is None and self.volume_data is not None:
            pos = self.volume_data.get_center()
        elif pos is None:
            pos = (0.0, 0.0, 0.0)

        implant = self.implant_manager.add_implant(
            tooth_number=tooth_number,
            diameter_mm=diameter_mm,
            length_mm=length_mm,
            position=pos
        )
        self.evaluate_implant_clearance()
        self._refresh_all_implant_views()
        return implant.implant_id

    def remove_active_implant(self) -> None:
        """Deletes the active virtual implant from the case."""
        if self.implant_manager.active_implant_id:
            self.implant_manager.remove_implant(self.implant_manager.active_implant_id)
            self._refresh_all_implant_views()

    def set_implant_preset(self, preset_idx: int) -> None:
        """Applies a clinical preset dimension to the active implant."""
        if 0 <= preset_idx < len(STANDARD_IMPLANT_PRESETS):
            p = STANDARD_IMPLANT_PRESETS[preset_idx]
            implant = self.implant_manager.get_active_implant()
            if implant:
                implant.set_dimensions(p.diameter_mm, p.length_mm)
                self.evaluate_implant_clearance()
                self._refresh_all_implant_views()

    def set_implant_dimensions(self, diameter_mm: float, length_mm: float) -> None:
        """Modifies diameter and length of active implant."""
        implant = self.implant_manager.get_active_implant()
        if implant:
            implant.set_dimensions(diameter_mm, length_mm)
            self.evaluate_implant_clearance()
            self._refresh_all_implant_views()

    def set_implant_angulation(self, bl_deg: float, md_deg: float) -> None:
        """Adjusts Bucco-Lingual and Mesio-Distal tilt angles."""
        implant = self.implant_manager.get_active_implant()
        if implant:
            implant.bl_angle_deg = bl_deg
            implant.md_angle_deg = md_deg
            implant.update_transform()
            self.evaluate_implant_clearance()
            self._refresh_all_implant_views()

    def set_implant_depth(self, z_world_mm: float) -> None:
        """Adjusts vertical Z depth of active implant."""
        implant = self.implant_manager.get_active_implant()
        if implant:
            implant.position[2] = z_world_mm
            implant.update_transform()
            self.evaluate_implant_clearance()
            self._refresh_all_implant_views()

    def toggle_implant_safety_sleeve(self, visible: bool) -> None:
        """Toggles visibility of the 2.0mm safety envelope wireframe."""
        implant = self.implant_manager.get_active_implant()
        if implant:
            implant.show_safety_sleeve = visible
            if visible:
                implant.sleeve_actor.VisibilityOn()
            else:
                implant.sleeve_actor.VisibilityOff()
            self._refresh_all_implant_views()

    def evaluate_implant_clearance(self) -> None:
        """Evaluates live Euclidean clearance from implant to mandibular canal."""
        self.implant_manager.evaluate_nerve_clearance(self.nerve_tracer)

    def _refresh_all_implant_views(self) -> None:
        """Redraws all viewport overlays for active implants."""
        self.volume_view.safe_render()
        self.cross_section_view._update_implant_overlays()

    def cleanup(self) -> None:
        self.stop_cine()
        self.axial_view.cleanup()
        self.coronal_view.cleanup()
        self.sagittal_view.cleanup()
        self.volume_view.cleanup()
        self.panoramic_view.cleanup()
        self.cross_section_view.cleanup()
