"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Module: ui/ribbon_toolbar.py

Commercial Medical PACS Clinical Action Ribbon Toolbar.
Features:
- Divided into high-density medical tool groups:
  1. Navigation & Cine Loop Player.
  2. Diagnostic Measurements (Caliper, 3-Point Angle, Cobb Angle, ROI).
  3. Viewport Layout Manager (2x2, 1x1, 1+3, 3D Only).
  4. Display & Filter Enhancement (Invert HU, Sharpen, MIP, Linked Crosshairs).
  5. Export & Sharing (Slice PNG, 3D STL Mesh, Anonymized DICOM).
"""

from __future__ import annotations
from typing import Optional
from PySide6.QtCore import QObject, Signal, Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QToolButton, QComboBox, QSlider, QFrame, QButtonGroup, QMenu
)
from PySide6.QtGui import QIcon, QAction


class RibbonSignals(QObject):
    """Signals emitted from the PACS Ribbon."""
    tool_changed = Signal(str)            # 'select' | 'pan' | 'zoom' | 'wl'
    cine_toggled = Signal(bool, int)      # is_playing, fps
    measurement_tool_selected = Signal(str) # 'distance' | 'angle' | 'cobb' | 'roi' | 'clear'
    layout_changed = Signal(str)          # '2x2' | '1x1' | '1+3' | '3d'
    invert_colors_toggled = Signal(bool)
    sharpen_filter_toggled = Signal(bool)
    mip_mode_toggled = Signal(bool)
    crosshair_toggled = Signal(bool)
    reset_views_clicked = Signal()
    export_slice_clicked = Signal()
    export_stl_clicked = Signal()
    export_dicom_clicked = Signal()
    report_clicked = Signal()


class RibbonGroup(QFrame):
    """Visual grouping container for PACS ribbon tool blocks."""

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("ribbon_group")
        self.setStyleSheet("""
            QFrame#ribbon_group {
                background-color: #171c1f;
                border: 1px solid #262b2e;
                border-radius: 4px;
                margin: 2px 3px;
            }
        """)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(6, 4, 6, 3)
        self.main_layout.setSpacing(2)

        # Tools Row Container
        self.tools_row = QHBoxLayout()
        self.tools_row.setContentsMargins(0, 0, 0, 0)
        self.tools_row.setSpacing(4)
        self.main_layout.addLayout(self.tools_row, 1)

        # Bottom Caption Label
        self.lbl_caption = QLabel(title.upper())
        self.lbl_caption.setAlignment(Qt.AlignCenter)
        self.lbl_caption.setStyleSheet("color: #849495; font-family: Inter; font-size: 9px; font-weight: 700; letter-spacing: 0.05em;")
        self.main_layout.addWidget(self.lbl_caption)


class RibbonButton(QPushButton):
    """PACS Style Ribbon Action Button with icon, text, and active highlight state."""

    def __init__(self, text: str, icon_symbol: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setCheckable(False)

        if icon_symbol:
            self.setText(f"{icon_symbol}\n{text}")
        else:
            self.setText(text)

        self.setStyleSheet("""
            QPushButton {
                background-color: #212629;
                color: #dfe3e7;
                border: 1px solid #2e3538;
                border-radius: 3px;
                padding: 4px 6px;
                font-family: Inter;
                font-size: 10px;
                font-weight: 600;
                min-width: 48px;
                min-height: 40px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #283034;
                border: 1px solid #00dbe9;
                color: #00dbe9;
            }
            QPushButton:checked {
                background-color: #00dbe9;
                color: #090e11;
                border: 1px solid #00dbe9;
                font-weight: 700;
            }
        """)


class RibbonToolbar(QWidget):
    """
    Commercial Medical PACS Action Ribbon.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.signals = RibbonSignals()
        self.setObjectName("ribbon_toolbar")
        self.setFixedHeight(74)

        self.setStyleSheet("""
            QWidget#ribbon_toolbar {
                background-color: #121619;
                border-bottom: 1px solid #262b2e;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(4)

        # 1. Navigation & Cine Loop Group
        self._build_navigation_group(layout)

        # 2. Diagnostic Measurements Group
        self._build_measurements_group(layout)

        # 3. Viewport Layout Manager Group
        self._build_layout_group(layout)

        # 4. Display & Filter Tools Group
        self._build_display_group(layout)

        # 5. Export & Sharing Group
        self._build_export_group(layout)

        layout.addStretch()

    def _build_navigation_group(self, parent_layout: QHBoxLayout) -> None:
        group = RibbonGroup("Navigation & Cine", self)

        self.btn_select = RibbonButton("Select", "↖")
        self.btn_select.setCheckable(True)
        self.btn_select.setChecked(True)

        self.btn_pan = RibbonButton("Pan", "✋")
        self.btn_pan.setCheckable(True)

        self.btn_zoom = RibbonButton("Zoom", "🔍")
        self.btn_zoom.setCheckable(True)

        self.btn_wl = RibbonButton("W / L", "🌓")
        self.btn_wl.setCheckable(True)

        # Tool button group (mutually exclusive)
        self.tool_button_group = QButtonGroup(self)
        self.tool_button_group.addButton(self.btn_select)
        self.tool_button_group.addButton(self.btn_pan)
        self.tool_button_group.addButton(self.btn_zoom)
        self.tool_button_group.addButton(self.btn_wl)

        self.btn_select.clicked.connect(lambda: self.signals.tool_changed.emit("select"))
        self.btn_pan.clicked.connect(lambda: self.signals.tool_changed.emit("pan"))
        self.btn_zoom.clicked.connect(lambda: self.signals.tool_changed.emit("zoom"))
        self.btn_wl.clicked.connect(lambda: self.signals.tool_changed.emit("wl"))

        group.tools_row.addWidget(self.btn_select)
        group.tools_row.addWidget(self.btn_pan)
        group.tools_row.addWidget(self.btn_zoom)
        group.tools_row.addWidget(self.btn_wl)

        # Cine Player Toggle
        self.btn_cine = RibbonButton("Cine", "▶")
        self.btn_cine.setCheckable(True)
        self.btn_cine.setToolTip("Start/Stop Automated Cine Loop (Slice Navigation)")
        self.btn_cine.clicked.connect(self._on_cine_toggled)
        group.tools_row.addWidget(self.btn_cine)

        parent_layout.addWidget(group)

    def _build_measurements_group(self, parent_layout: QHBoxLayout) -> None:
        group = RibbonGroup("Measurements", self)

        self.btn_dist = RibbonButton("Caliper", "📏")
        self.btn_dist.setToolTip("Measure 2-point physical distance in millimeters")
        self.btn_dist.clicked.connect(lambda: self.signals.measurement_tool_selected.emit("distance"))

        self.btn_angle = RibbonButton("Angle", "📐")
        self.btn_angle.setToolTip("Measure 3-point clinical angle (degrees)")
        self.btn_angle.clicked.connect(lambda: self.signals.measurement_tool_selected.emit("angle"))

        self.btn_roi = RibbonButton("ROI Box", "⬚")
        self.btn_roi.setToolTip("Sample Hounsfield Unit statistics inside rectangular ROI")
        self.btn_roi.clicked.connect(lambda: self.signals.measurement_tool_selected.emit("roi"))

        self.btn_clear_meas = RibbonButton("Clear", "🗑")
        self.btn_clear_meas.setToolTip("Clear all caliper overlays")
        self.btn_clear_meas.clicked.connect(lambda: self.signals.measurement_tool_selected.emit("clear"))

        group.tools_row.addWidget(self.btn_dist)
        group.tools_row.addWidget(self.btn_angle)
        group.tools_row.addWidget(self.btn_roi)
        group.tools_row.addWidget(self.btn_clear_meas)

        parent_layout.addWidget(group)

    def _build_layout_group(self, parent_layout: QHBoxLayout) -> None:
        group = RibbonGroup("Grid Layout", self)

        self.btn_layout_2x2 = RibbonButton("2 × 2", "⊞")
        self.btn_layout_2x2.setCheckable(True)
        self.btn_layout_2x2.setChecked(True)
        self.btn_layout_2x2.setToolTip("Standard 2x2 Multiplanar + 3D Grid")

        self.btn_layout_1x1 = RibbonButton("1 × 1", "🗖")
        self.btn_layout_1x1.setCheckable(True)
        self.btn_layout_1x1.setToolTip("Single Viewport Maximized")

        self.btn_layout_1p3 = RibbonButton("1 + 3", "⚏")
        self.btn_layout_1p3.setCheckable(True)
        self.btn_layout_1p3.setToolTip("Master Panoramic View + 3 Inset Slices")

        self.btn_layout_implant = RibbonButton("Implant", "🦷")
        self.btn_layout_implant.setCheckable(True)
        self.btn_layout_implant.setToolTip("Dental Implant Planning Layout (Axial Arch, Panoramic, Cross-Section, 3D)")

        self.btn_layout_3d = RibbonButton("3D Full", "🧊")
        self.btn_layout_3d.setCheckable(True)
        self.btn_layout_3d.setToolTip("Fullscreen 3D GPU Volume View")

        self.layout_button_group = QButtonGroup(self)
        self.layout_button_group.addButton(self.btn_layout_2x2)
        self.layout_button_group.addButton(self.btn_layout_1x1)
        self.layout_button_group.addButton(self.btn_layout_1p3)
        self.layout_button_group.addButton(self.btn_layout_implant)
        self.layout_button_group.addButton(self.btn_layout_3d)

        self.btn_layout_2x2.clicked.connect(lambda: self.signals.layout_changed.emit("2x2"))
        self.btn_layout_1x1.clicked.connect(lambda: self.signals.layout_changed.emit("1x1"))
        self.btn_layout_1p3.clicked.connect(lambda: self.signals.layout_changed.emit("1+3"))
        self.btn_layout_implant.clicked.connect(lambda: self.signals.layout_changed.emit("implant"))
        self.btn_layout_3d.clicked.connect(lambda: self.signals.layout_changed.emit("3d"))

        group.tools_row.addWidget(self.btn_layout_2x2)
        group.tools_row.addWidget(self.btn_layout_1x1)
        group.tools_row.addWidget(self.btn_layout_1p3)
        group.tools_row.addWidget(self.btn_layout_implant)
        group.tools_row.addWidget(self.btn_layout_3d)

        parent_layout.addWidget(group)

    def _build_display_group(self, parent_layout: QHBoxLayout) -> None:
        group = RibbonGroup("Display & Filters", self)

        self.btn_invert = RibbonButton("Invert", "☯")
        self.btn_invert.setCheckable(True)
        self.btn_invert.setToolTip("Invert grayscale HU color mapping (Negative film mode)")
        self.btn_invert.toggled.connect(self.signals.invert_colors_toggled.emit)

        self.btn_crosshairs = RibbonButton("Crosshair", "✛")
        self.btn_crosshairs.setCheckable(True)
        self.btn_crosshairs.setChecked(True)
        self.btn_crosshairs.setToolTip("Toggle synchronized 3D crosshair lines")
        self.btn_crosshairs.toggled.connect(self.signals.crosshair_toggled.emit)

        self.btn_reset = RibbonButton("Reset", "↺")
        self.btn_reset.setToolTip("Reset all cameras and centering")
        self.btn_reset.clicked.connect(self.signals.reset_views_clicked.emit)

        group.tools_row.addWidget(self.btn_invert)
        group.tools_row.addWidget(self.btn_crosshairs)
        group.tools_row.addWidget(self.btn_reset)

        parent_layout.addWidget(group)

    def _build_export_group(self, parent_layout: QHBoxLayout) -> None:
        group = RibbonGroup("PACS & Reports", self)

        self.btn_report = RibbonButton("Report", "📑")
        self.btn_report.setToolTip("Generate comprehensive Dental Implant Surgical Planning PDF Report")
        self.btn_report.clicked.connect(self.signals.report_clicked.emit)

        self.btn_export_png = RibbonButton("Slice PNG", "🖼")
        self.btn_export_png.setToolTip("Export active 2D slice as high-res PNG image")
        self.btn_export_png.clicked.connect(self.signals.export_slice_clicked.emit)

        self.btn_export_stl = RibbonButton("3D Mesh", "🦷")
        self.btn_export_stl.setToolTip("Export segmented 3D bone surface mesh (STL)")
        self.btn_export_stl.clicked.connect(self.signals.export_stl_clicked.emit)

        self.btn_export_dcm = RibbonButton("DICOM", "💾")
        self.btn_export_dcm.setToolTip("Export anonymized DICOM file series")
        self.btn_export_dcm.clicked.connect(self.signals.export_dicom_clicked.emit)

        group.tools_row.addWidget(self.btn_report)
        group.tools_row.addWidget(self.btn_export_png)
        group.tools_row.addWidget(self.btn_export_stl)
        group.tools_row.addWidget(self.btn_export_dcm)

        parent_layout.addWidget(group)

    def _on_cine_toggled(self, checked: bool) -> None:
        if checked:
            self.btn_cine.setText("⏸\nPause")
            self.signals.cine_toggled.emit(True, 15)  # 15 FPS
        else:
            self.btn_cine.setText("▶\nCine")
            self.signals.cine_toggled.emit(False, 15)
