"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Module: ui/control_panel.py

Right-hand clinical control dock.
Features:
- Patient & Series acquisition metadata card with text wrapping.
- Synchronized multi-axis slice navigators (Axial, Coronal, Sagittal) in voxels and millimeters.
- 2D Window/Level contrast sliders with 1-click clinical presets (Bone, Teeth, Soft Tissue, etc.).
- Dedicated Mandibular Inferior Alveolar Nerve Tracing Section:
  * Left / Right nerve channel drawing toggles.
  * Real-time nerve canal diameter slider (1.0mm to 4.0mm).
  * Live anatomical length telemetry (mm).
  * Undo point, Clear canal, and Export coordinates (JSON/CSV).
- 3D GPU volume rendering controls (Transfer Function presets, Opacity, Shading).
- Calibrated 2D Caliper measurement and crosshair toggles.
"""

from __future__ import annotations
from typing import Optional, Dict
from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QSpinBox, QDoubleSpinBox, QPushButton, QComboBox,
    QCheckBox, QGroupBox, QFrame, QScrollArea, QSizePolicy
)

from core.volume_data import VolumeData
from core.presets import WL_PRESETS, VOLUME_3D_PRESETS


class ControlPanelSignals(QObject):
    """Qt signals emitted from the control panel."""
    slice_navigated = Signal(str, int)                # plane ('axial'|'coronal'|'sagittal'), slice_idx
    window_level_changed = Signal(float, float)       # ww, wl
    preset_selected = Signal(str)                     # preset_key
    volume_preset_changed = Signal(str)               # 3d preset id
    volume_opacity_changed = Signal(float)            # opacity multiplier
    volume_shading_toggled = Signal(bool)             # shading on/off
    crosshair_toggled = Signal(bool)                  # crosshair visible
    start_measurement_clicked = Signal()              # start caliper
    clear_measurement_clicked = Signal()              # clear caliper
    reset_views_clicked = Signal()                    # reset all viewports

    # Mandibular Nerve Signals
    nerve_draw_toggled = Signal(bool, str)            # is_drawing, channel ("left"|"right")
    nerve_undo_clicked = Signal()                     # undo last point
    nerve_clear_clicked = Signal()                    # clear active canal
    nerve_diameter_changed = Signal(float)            # diameter in mm
    nerve_export_clicked = Signal()                   # export coordinates

    # Panoramic & Cross-Section Signals
    arch_draw_toggled = Signal(bool)                  # is_drawing_arch
    arch_autofit_clicked = Signal()                   # auto-fit parabola
    arch_clear_clicked = Signal()                     # clear arch
    trough_thickness_changed = Signal(float)          # slab thickness in mm
    cross_section_navigated = Signal(int)             # cross-section slice index

    # Virtual Dental Implant Signals
    implant_add_clicked = Signal()
    implant_delete_clicked = Signal()
    implant_preset_selected = Signal(int)
    implant_dimensions_changed = Signal(float, float) # diameter, length
    implant_angulation_changed = Signal(float, float) # bl_deg, md_deg
    implant_depth_changed = Signal(float)             # z_offset
    implant_sleeve_toggled = Signal(bool)             # show_sleeve

    # 3D AI Segmentation & IOS Mesh Alignment Signals
    load_segmentation_clicked = Signal()              # load .nii / .nrrd mask
    import_ios_clicked = Signal()                     # import .stl / .ply scan
    run_icp_clicked = Signal()                        # auto-align IOS via ICP
    mesh_visibility_changed = Signal(str, bool)       # structure_id, visible
    mesh_opacity_changed = Signal(str, float)         # structure_id, opacity
    ios_visibility_changed = Signal(bool)             # show/hide IOS scan
    ios_opacity_changed = Signal(float)               # IOS opacity [0..1]


class SectionCard(QFrame):
    """Styled collapsible-style clinical card container."""

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("section_card")
        self.setStyleSheet("""
            QFrame#section_card {
                background-color: #171c1f;
                border: 1px solid #262b2e;
                border-radius: 4px;
                margin-bottom: 6px;
            }
        """)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 8, 10, 8)
        self.layout.setSpacing(6)

        # Title Label
        title_label = QLabel(title.upper())
        title_label.setStyleSheet("color: #00dbe9; font-family: Inter; font-size: 10px; font-weight: 700; letter-spacing: 0.08em;")
        self.layout.addWidget(title_label)


class ControlPanel(QWidget):
    """Right-side control dock for the Dental CBCT viewer."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.signals = ControlPanelSignals()
        self.volume_data: Optional[VolumeData] = None

        self.setFixedWidth(280)
        self.setObjectName("control_panel")
        self.setStyleSheet("""
            QWidget#control_panel {
                background-color: #111618;
                border-left: 1px solid #262b2e;
            }
        """)

        # Main Layout
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Scroll Area for dense UI
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #111618;
                width: 6px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background-color: #262b2e;
                border-radius: 3px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #00dbe9;
            }
        """)

        self.scroll_content = QWidget()
        self.content_layout = QVBoxLayout(self.scroll_content)
        self.content_layout.setContentsMargins(6, 6, 6, 6)
        self.content_layout.setSpacing(4)

        # 1. Patient & Metadata Card
        self._build_metadata_section()

        # 2. Panoramic & Bucco-Lingual Cross-Section Card
        self._build_panoramic_section()

        # 3. Virtual Dental Implant Planning Card
        self._build_implant_section()

        # 4. Mandibular Nerve Canal Tracing Card
        self._build_nerve_tracing_section()

        # 5. Multi-Axis Slice Navigation Card
        self._build_slice_navigation_section()

        # 6. Window / Level (Contrast) Card
        self._build_window_level_section()

        # 7. 3D Volume Rendering Card
        self._build_volume_3d_section()

        # 8. 3D AI Segmentation & Mesh Alignment Card
        self._build_segmentation_section()

        # 9. Diagnostic Tools & Measurement Card
        self._build_tools_section()

        self.content_layout.addStretch()

    def _build_metadata_section(self) -> None:
        """Builds Patient and Scan Information Card with word wrapping."""
        card = SectionCard("Patient & Acquisition Info", self)

        self.lbl_patient_name = QLabel("Name: --")
        self.lbl_patient_id = QLabel("ID: --")
        self.lbl_dimensions = QLabel("Matrix: --")
        self.lbl_spacing = QLabel("Spacing: --")
        self.lbl_fov = QLabel("FoV: --")

        for lbl in (self.lbl_patient_name, self.lbl_patient_id, self.lbl_dimensions, self.lbl_spacing, self.lbl_fov):
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color: #dfe3e7; font-family: 'JetBrains Mono', monospace; font-size: 11px;")
            card.layout.addWidget(lbl)

        self.content_layout.addWidget(card)

    def _build_panoramic_section(self) -> None:
        """Builds Panoramic Curved MPR & Bucco-Lingual Cross-Section Card."""
        card = SectionCard("Panoramic & Cross-Sections", self)

        # Draw / Auto-Fit Buttons Row
        row_arch = QHBoxLayout()
        row_arch.setSpacing(4)

        self.btn_draw_arch = QPushButton("✏ Draw Arch")
        self.btn_draw_arch.setCheckable(True)
        self.btn_draw_arch.setCursor(Qt.PointingHandCursor)
        self.btn_draw_arch.setStyleSheet("""
            QPushButton {
                background-color: #212629;
                color: #00dbe9;
                border: 1px solid #3b494b;
                border-radius: 3px;
                padding: 4px 6px;
                font-weight: 600;
                font-size: 10px;
            }
            QPushButton:checked {
                background-color: #00dbe9;
                color: #090e11;
                border: 1px solid #00dbe9;
            }
        """)
        self.btn_draw_arch.toggled.connect(self._on_draw_arch_toggled)
        row_arch.addWidget(self.btn_draw_arch)

        self.btn_autofit_arch = QPushButton("✨ Auto-Fit")
        self.btn_autofit_arch.setCursor(Qt.PointingHandCursor)
        self.btn_autofit_arch.setStyleSheet("""
            QPushButton {
                background-color: #212629;
                color: #dfe3e7;
                border: 1px solid #3b494b;
                border-radius: 3px;
                padding: 4px 6px;
                font-size: 10px;
                font-weight: 600;
            }
            QPushButton:hover {
                border: 1px solid #00dbe9;
                color: #00dbe9;
            }
        """)
        self.btn_autofit_arch.clicked.connect(self.signals.arch_autofit_clicked.emit)
        row_arch.addWidget(self.btn_autofit_arch)

        self.btn_clear_arch = QPushButton("🗑 Clear")
        self.btn_clear_arch.setCursor(Qt.PointingHandCursor)
        self.btn_clear_arch.setStyleSheet("""
            QPushButton {
                background-color: #212629;
                color: #dfe3e7;
                border: 1px solid #3b494b;
                border-radius: 3px;
                padding: 4px 6px;
                font-size: 10px;
            }
            QPushButton:hover {
                border: 1px solid #ff2d55;
                color: #ff2d55;
            }
        """)
        self.btn_clear_arch.clicked.connect(self.signals.arch_clear_clicked.emit)
        row_arch.addWidget(self.btn_clear_arch)
        card.layout.addLayout(row_arch)

        # Focal Trough Thickness Slider
        lbl_trough = QLabel("Focal Trough Slab:")
        lbl_trough.setStyleSheet("color: #849495; font-size: 10px;")
        card.layout.addWidget(lbl_trough)

        row_trough = QHBoxLayout()
        self.slider_trough = QSlider(Qt.Horizontal)
        self.slider_trough.setRange(2, 25)
        self.slider_trough.setValue(8)
        row_trough.addWidget(self.slider_trough, 1)

        self.lbl_trough_val = QLabel("8.0 mm")
        self.lbl_trough_val.setStyleSheet("color: #00dbe9; font-family: 'JetBrains Mono'; font-size: 10px; font-weight: bold;")
        row_trough.addWidget(self.lbl_trough_val)
        card.layout.addLayout(row_trough)
        self.slider_trough.valueChanged.connect(self._on_trough_slider_changed)

        # Cross-Section Slice Navigator
        lbl_cross = QLabel("Cross-Section Navigator:")
        lbl_cross.setStyleSheet("color: #849495; font-size: 10px;")
        card.layout.addWidget(lbl_cross)

        row_cross = QHBoxLayout()
        self.slider_cross = QSlider(Qt.Horizontal)
        self.slider_cross.setRange(0, 100)
        self.slider_cross.setValue(0)
        row_cross.addWidget(self.slider_cross, 1)

        self.lbl_cross_val = QLabel("#0 / 0")
        self.lbl_cross_val.setStyleSheet("color: #ff9500; font-family: 'JetBrains Mono'; font-size: 10px; font-weight: bold;")
        row_cross.addWidget(self.lbl_cross_val)
        card.layout.addLayout(row_cross)
        self.slider_cross.valueChanged.connect(self._on_cross_slider_changed)

        # Arch Length Readout
        self.lbl_arch_telemetry = QLabel("Dental Arch: 0.0 mm (0 slices)")
        self.lbl_arch_telemetry.setStyleSheet("color: #849495; font-family: 'JetBrains Mono'; font-size: 10px;")
        card.layout.addWidget(self.lbl_arch_telemetry)

        self.content_layout.addWidget(card)

    def _build_implant_section(self) -> None:
        """Builds Virtual Dental Implant Planning & Safety Clearance Card."""
        card = SectionCard("Virtual Implant Planning", self)

        # Action Buttons (Add, Delete, Toggle Sleeve)
        row_acts = QHBoxLayout()
        row_acts.setSpacing(4)

        self.btn_add_implant = QPushButton("➕ Add")
        self.btn_add_implant.setCursor(Qt.PointingHandCursor)
        self.btn_add_implant.setStyleSheet("""
            QPushButton {
                background-color: #212629;
                color: #00dbe9;
                border: 1px solid #3b494b;
                border-radius: 3px;
                padding: 4px 6px;
                font-weight: 600;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #00dbe9;
                color: #090e11;
            }
        """)
        self.btn_add_implant.clicked.connect(self.signals.implant_add_clicked.emit)
        row_acts.addWidget(self.btn_add_implant)

        self.btn_del_implant = QPushButton("🗑 Delete")
        self.btn_del_implant.setCursor(Qt.PointingHandCursor)
        self.btn_del_implant.setStyleSheet("""
            QPushButton {
                background-color: #212629;
                color: #dfe3e7;
                border: 1px solid #3b494b;
                border-radius: 3px;
                padding: 4px 6px;
                font-size: 10px;
            }
            QPushButton:hover {
                border: 1px solid #ff2d55;
                color: #ff2d55;
            }
        """)
        self.btn_del_implant.clicked.connect(self.signals.implant_delete_clicked.emit)
        row_acts.addWidget(self.btn_del_implant)

        self.chk_sleeve = QCheckBox("2mm Sleeve")
        self.chk_sleeve.setChecked(True)
        self.chk_sleeve.setStyleSheet("color: #849495; font-size: 10px;")
        self.chk_sleeve.toggled.connect(self.signals.implant_sleeve_toggled.emit)
        row_acts.addWidget(self.chk_sleeve)

        card.layout.addLayout(row_acts)

        # Clinical Preset Combo
        from dental.implant_simulator import STANDARD_IMPLANT_PRESETS
        self.combo_implant_preset = QComboBox()
        self.combo_implant_preset.setStyleSheet("""
            QComboBox {
                background-color: #212629;
                color: #dfe3e7;
                border: 1px solid #3b494b;
                border-radius: 3px;
                padding: 3px 6px;
                font-size: 10px;
            }
        """)
        for idx, preset in enumerate(STANDARD_IMPLANT_PRESETS):
            self.combo_implant_preset.addItem(preset.name, idx)
        self.combo_implant_preset.currentIndexChanged.connect(self.signals.implant_preset_selected.emit)
        card.layout.addWidget(self.combo_implant_preset)

        # Diameter & Length Sliders
        lbl_diam = QLabel("Diameter (Ø):")
        lbl_diam.setStyleSheet("color: #849495; font-size: 10px;")
        card.layout.addWidget(lbl_diam)

        row_diam = QHBoxLayout()
        self.slider_implant_diam = QSlider(Qt.Horizontal)
        self.slider_implant_diam.setRange(30, 60)
        self.slider_implant_diam.setValue(40) # 4.0 mm
        row_diam.addWidget(self.slider_implant_diam, 1)

        self.lbl_implant_diam_val = QLabel("4.0 mm")
        self.lbl_implant_diam_val.setStyleSheet("color: #00dbe9; font-family: 'JetBrains Mono'; font-size: 10px; font-weight: bold;")
        row_diam.addWidget(self.lbl_implant_diam_val)
        card.layout.addLayout(row_diam)

        lbl_len = QLabel("Length (L):")
        lbl_len.setStyleSheet("color: #849495; font-size: 10px;")
        card.layout.addWidget(lbl_len)

        row_len = QHBoxLayout()
        self.slider_implant_len = QSlider(Qt.Horizontal)
        self.slider_implant_len.setRange(80, 160)
        self.slider_implant_len.setValue(115) # 11.5 mm
        row_len.addWidget(self.slider_implant_len, 1)

        self.lbl_implant_len_val = QLabel("11.5 mm")
        self.lbl_implant_len_val.setStyleSheet("color: #00dbe9; font-family: 'JetBrains Mono'; font-size: 10px; font-weight: bold;")
        row_len.addWidget(self.lbl_implant_len_val)
        card.layout.addLayout(row_len)

        self.slider_implant_diam.valueChanged.connect(self._on_implant_dim_slider_changed)
        self.slider_implant_len.valueChanged.connect(self._on_implant_dim_slider_changed)

        # Bucco-Lingual (BL) & Mesio-Distal (MD) Angulations
        lbl_ang = QLabel("BL Tilt Angulation:")
        lbl_ang.setStyleSheet("color: #849495; font-size: 10px;")
        card.layout.addWidget(lbl_ang)

        row_ang = QHBoxLayout()
        self.slider_bl_ang = QSlider(Qt.Horizontal)
        self.slider_bl_ang.setRange(-45, 45)
        self.slider_bl_ang.setValue(0)
        row_ang.addWidget(self.slider_bl_ang, 1)

        self.lbl_bl_ang_val = QLabel("0°")
        self.lbl_bl_ang_val.setStyleSheet("color: #ff9500; font-family: 'JetBrains Mono'; font-size: 10px; font-weight: bold;")
        row_ang.addWidget(self.lbl_bl_ang_val)
        card.layout.addLayout(row_ang)
        self.slider_bl_ang.valueChanged.connect(self._on_implant_ang_changed)

        # LIVE NERVE SAFETY CLEARANCE HUD BADGE
        self.frame_safety_badge = QFrame()
        self.frame_safety_badge.setObjectName("safety_badge")
        self.frame_safety_badge.setStyleSheet("""
            QFrame#safety_badge {
                background-color: #0d2818;
                border: 1px solid #00FF7F;
                border-radius: 4px;
                padding: 4px;
            }
        """)
        badge_layout = QVBoxLayout(self.frame_safety_badge)
        badge_layout.setContentsMargins(6, 4, 6, 4)
        badge_layout.setSpacing(2)

        self.lbl_safety_status = QLabel("🛡 SAFE: >= 2.0 mm")
        self.lbl_safety_status.setStyleSheet("color: #00FF7F; font-size: 10px; font-weight: bold;")
        badge_layout.addWidget(self.lbl_safety_status)

        self.lbl_safety_dist = QLabel("Clearance: -- mm")
        self.lbl_safety_dist.setStyleSheet("color: #dfe3e7; font-family: 'JetBrains Mono'; font-size: 10px;")
        badge_layout.addWidget(self.lbl_safety_dist)

        card.layout.addWidget(self.frame_safety_badge)

        self.content_layout.addWidget(card)

    def _on_implant_dim_slider_changed(self) -> None:
        d = self.slider_implant_diam.value() / 10.0
        l = self.slider_implant_len.value() / 10.0
        self.lbl_implant_diam_val.setText(f"{d:.1f} mm")
        self.lbl_implant_len_val.setText(f"{l:.1f} mm")
        self.signals.implant_dimensions_changed.emit(d, l)

    def _on_implant_ang_changed(self, val: int) -> None:
        self.lbl_bl_ang_val.setText(f"{val}°")
        self.signals.implant_angulation_changed.emit(float(val), 0.0)

    def update_implant_safety_hud(self, implant_id: str, min_dist_mm: float, state_str: str, nerve_name: str) -> None:
        """Updates live color-coded clearance badge in the control panel."""
        if min_dist_mm == float('inf'):
            self.lbl_safety_dist.setText("Clearance: No Nerve Traced")
            self.lbl_safety_status.setText("🛡 CLEAR: No Obstacle")
            self.frame_safety_badge.setStyleSheet("""
                QFrame#safety_badge {
                    background-color: #0d2818;
                    border: 1px solid #00FF7F;
                    border-radius: 4px;
                }
            """)
            self.lbl_safety_status.setStyleSheet("color: #00FF7F; font-size: 10px; font-weight: bold;")
            return

        self.lbl_safety_dist.setText(f"Clearance: {min_dist_mm:.2f} mm ({nerve_name})")
        if state_str == "safe":
            self.lbl_safety_status.setText(f"🛡 SAFE: {min_dist_mm:.2f} mm")
            self.frame_safety_badge.setStyleSheet("""
                QFrame#safety_badge {
                    background-color: #0d2818;
                    border: 1px solid #00FF7F;
                    border-radius: 4px;
                }
            """)
            self.lbl_safety_status.setStyleSheet("color: #00FF7F; font-size: 10px; font-weight: bold;")
        elif state_str == "warning":
            self.lbl_safety_status.setText(f"⚠ WARNING: {min_dist_mm:.2f} mm")
            self.frame_safety_badge.setStyleSheet("""
                QFrame#safety_badge {
                    background-color: #2b2206;
                    border: 1px solid #FFD700;
                    border-radius: 4px;
                }
            """)
            self.lbl_safety_status.setStyleSheet("color: #FFD700; font-size: 10px; font-weight: bold;")
        else:
            self.lbl_safety_status.setText(f"🚨 CRITICAL BREACH: {min_dist_mm:.2f} mm")
            self.frame_safety_badge.setStyleSheet("""
                QFrame#safety_badge {
                    background-color: #3b0a0a;
                    border: 1px solid #FF0033;
                    border-radius: 4px;
                }
            """)
            self.lbl_safety_status.setStyleSheet("color: #FF0033; font-size: 10px; font-weight: bold;")

    def _on_draw_arch_toggled(self, checked: bool) -> None:
        if checked:
            self.btn_draw_left.setChecked(False)
            self.btn_draw_right.setChecked(False)
        self.signals.arch_draw_toggled.emit(checked)

    def _on_trough_slider_changed(self, val: int) -> None:
        t_mm = float(val)
        self.lbl_trough_val.setText(f"{t_mm:.1f} mm")
        self.signals.trough_thickness_changed.emit(t_mm)

    def _on_cross_slider_changed(self, val: int) -> None:
        self.signals.cross_section_navigated.emit(val)

    def update_arch_telemetry(self, count: int, length_mm: float) -> None:
        """Updates dental arch length and slice count readout."""
        self.lbl_arch_telemetry.setText(f"Dental Arch: {length_mm:.1f} mm ({count} pts)")
        self.slider_cross.setRange(0, max(0, count - 1))

    def update_cross_section_index(self, active_idx: int, total_slices: int) -> None:
        """Updates cross-section navigator slider value and label."""
        self.slider_cross.blockSignals(True)
        self.slider_cross.setRange(0, max(0, total_slices - 1))
        self.slider_cross.setValue(active_idx)
        self.slider_cross.blockSignals(False)
        self.lbl_cross_val.setText(f"#{active_idx + 1} / {total_slices}")

    def _build_nerve_tracing_section(self) -> None:
        """Builds Mandibular Nerve Canal Tracing Card."""
        card = SectionCard("Mandibular Nerve Tracing", self)

        # Left / Right Nerve Draw Toggles
        row_toggles = QHBoxLayout()
        row_toggles.setSpacing(4)

        self.btn_draw_left = QPushButton("🔴 Draw Left")
        self.btn_draw_left.setCheckable(True)
        self.btn_draw_left.setCursor(Qt.PointingHandCursor)
        self.btn_draw_left.setStyleSheet("""
            QPushButton {
                background-color: #212629;
                color: #ff2d55;
                border: 1px solid #3b494b;
                border-radius: 3px;
                padding: 5px 8px;
                font-weight: 600;
                font-size: 11px;
            }
            QPushButton:checked {
                background-color: #ff2d55;
                color: #ffffff;
                border: 1px solid #ff2d55;
            }
        """)
        self.btn_draw_left.toggled.connect(self._on_draw_left_toggled)
        row_toggles.addWidget(self.btn_draw_left)

        self.btn_draw_right = QPushButton("🟠 Draw Right")
        self.btn_draw_right.setCheckable(True)
        self.btn_draw_right.setCursor(Qt.PointingHandCursor)
        self.btn_draw_right.setStyleSheet("""
            QPushButton {
                background-color: #212629;
                color: #ff9500;
                border: 1px solid #3b494b;
                border-radius: 3px;
                padding: 5px 8px;
                font-weight: 600;
                font-size: 11px;
            }
            QPushButton:checked {
                background-color: #ff9500;
                color: #ffffff;
                border: 1px solid #ff9500;
            }
        """)
        self.btn_draw_right.toggled.connect(self._on_draw_right_toggled)
        row_toggles.addWidget(self.btn_draw_right)

        card.layout.addLayout(row_toggles)

        # Diameter Slider (1.0mm to 4.0mm)
        lbl_diam = QLabel("Canal Diameter:")
        lbl_diam.setStyleSheet("color: #b9cacb; font-size: 11px;")
        card.layout.addWidget(lbl_diam)

        row_diam = QHBoxLayout()
        self.slider_diameter = QSlider(Qt.Horizontal)
        self.slider_diameter.setRange(8, 40)  # 0.8mm to 4.0mm
        self.slider_diameter.setValue(20)      # 2.0mm default
        row_diam.addWidget(self.slider_diameter, 1)

        self.lbl_diam_val = QLabel("2.0 mm")
        self.lbl_diam_val.setStyleSheet("color: #00dbe9; font-family: 'JetBrains Mono'; font-size: 11px;")
        row_diam.addWidget(self.lbl_diam_val)
        card.layout.addLayout(row_diam)

        self.slider_diameter.valueChanged.connect(self._on_diameter_changed)

        # Telemetry Labels
        self.lbl_nerve_left_len = QLabel("Left Canal:  0.0 mm (0 pts)")
        self.lbl_nerve_left_len.setStyleSheet("color: #ff2d55; font-family: 'JetBrains Mono'; font-size: 10px;")
        card.layout.addWidget(self.lbl_nerve_left_len)

        self.lbl_nerve_right_len = QLabel("Right Canal: 0.0 mm (0 pts)")
        self.lbl_nerve_right_len.setStyleSheet("color: #ff9500; font-family: 'JetBrains Mono'; font-size: 10px;")
        card.layout.addWidget(self.lbl_nerve_right_len)

        # Actions (Undo, Clear, Export)
        row_acts = QHBoxLayout()
        row_acts.setSpacing(4)

        self.btn_undo_nerve = QPushButton("Undo Point")
        self.btn_undo_nerve.clicked.connect(self.signals.nerve_undo_clicked.emit)
        row_acts.addWidget(self.btn_undo_nerve)

        self.btn_clear_nerve = QPushButton("Clear Canal")
        self.btn_clear_nerve.clicked.connect(self.signals.nerve_clear_clicked.emit)
        row_acts.addWidget(self.btn_clear_nerve)

        card.layout.addLayout(row_acts)

        self.btn_export_nerve = QPushButton("💾 Export Coordinates")
        self.btn_export_nerve.setToolTip("Export nerve canal coordinates to JSON or CSV file.")
        self.btn_export_nerve.clicked.connect(self.signals.nerve_export_clicked.emit)
        card.layout.addWidget(self.btn_export_nerve)

        self.content_layout.addWidget(card)

    def _on_draw_left_toggled(self, checked: bool) -> None:
        if checked:
            self.btn_draw_right.blockSignals(True)
            self.btn_draw_right.setChecked(False)
            self.btn_draw_right.blockSignals(False)
            self.signals.nerve_draw_toggled.emit(True, "left")
        else:
            self.signals.nerve_draw_toggled.emit(False, "left")

    def _on_draw_right_toggled(self, checked: bool) -> None:
        if checked:
            self.btn_draw_left.blockSignals(True)
            self.btn_draw_left.setChecked(False)
            self.btn_draw_left.blockSignals(False)
            self.signals.nerve_draw_toggled.emit(True, "right")
        else:
            self.signals.nerve_draw_toggled.emit(False, "right")

    def _on_diameter_changed(self, value: int) -> None:
        diam_mm = value / 10.0
        self.lbl_diam_val.setText(f"{diam_mm:.1f} mm")
        self.signals.nerve_diameter_changed.emit(diam_mm)

    def update_nerve_telemetry(self, channel: str, count: int, length_mm: float) -> None:
        """Updates live nerve telemetry displays."""
        if channel.lower() == "left":
            self.lbl_nerve_left_len.setText(f"Left Canal:  {length_mm:.1f} mm ({count} pts)")
        else:
            self.lbl_nerve_right_len.setText(f"Right Canal: {length_mm:.1f} mm ({count} pts)")

    def _build_slice_navigation_section(self) -> None:
        """Builds Axial, Coronal, and Sagittal slice navigators."""
        card = SectionCard("Slice Navigation", self)

        # Axial (Z)
        self.slider_axial, self.spin_axial, self.lbl_axial_mm = self._create_axis_control(
            card, "Axial (Z)", "#00dbe9", lambda val: self.signals.slice_navigated.emit("axial", val)
        )
        # Coronal (Y)
        self.slider_coronal, self.spin_coronal, self.lbl_coronal_mm = self._create_axis_control(
            card, "Coronal (Y)", "#4ade80", lambda val: self.signals.slice_navigated.emit("coronal", val)
        )
        # Sagittal (X)
        self.slider_sagittal, self.spin_sagittal, self.lbl_sagittal_mm = self._create_axis_control(
            card, "Sagittal (X)", "#f59e0b", lambda val: self.signals.slice_navigated.emit("sagittal", val)
        )

        self.content_layout.addWidget(card)

    def _create_axis_control(self, card: SectionCard, label_text: str, color: str, callback):
        """Helper to create a unified Slider + SpinBox + Millimeter label."""
        row_hdr = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: 600;")
        row_hdr.addWidget(lbl)

        lbl_mm = QLabel("0.0 mm")
        lbl_mm.setStyleSheet("color: #849495; font-family: 'JetBrains Mono'; font-size: 10px;")
        row_hdr.addStretch()
        row_hdr.addWidget(lbl_mm)
        card.layout.addLayout(row_hdr)

        row_ctrl = QHBoxLayout()
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(50)
        row_ctrl.addWidget(slider, 1)

        spin = QSpinBox()
        spin.setRange(0, 100)
        spin.setValue(50)
        spin.setFixedWidth(54)
        row_ctrl.addWidget(spin)
        card.layout.addLayout(row_ctrl)

        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)
        slider.valueChanged.connect(callback)

        return slider, spin, lbl_mm

    def _build_window_level_section(self) -> None:
        """Builds Window Width & Window Level sliders and preset buttons."""
        card = SectionCard("2D Window / Level (Contrast)", self)

        # Presets Buttons Grid
        grid_presets = QHBoxLayout()
        grid_presets.setSpacing(4)

        for key, preset in list(WL_PRESETS.items())[:3]:
            btn = QPushButton(preset.name.split()[0])
            btn.setProperty("class", "preset-btn")
            btn.setToolTip(f"{preset.name}\nWW: {preset.window_width:.0f}, WL: {preset.window_level:.0f}\n{preset.description}")
            btn.clicked.connect(lambda _, k=key: self._on_preset_clicked(k))
            grid_presets.addWidget(btn)
        card.layout.addLayout(grid_presets)

        grid_presets_2 = QHBoxLayout()
        grid_presets_2.setSpacing(4)
        for key, preset in list(WL_PRESETS.items())[3:]:
            btn = QPushButton(preset.name.split()[0])
            btn.setProperty("class", "preset-btn")
            btn.setToolTip(f"{preset.name}\nWW: {preset.window_width:.0f}, WL: {preset.window_level:.0f}\n{preset.description}")
            btn.clicked.connect(lambda _, k=key: self._on_preset_clicked(k))
            grid_presets_2.addWidget(btn)
        card.layout.addLayout(grid_presets_2)

        # Window Width (WW) Slider
        lbl_ww = QLabel("Window Width (WW):")
        lbl_ww.setStyleSheet("color: #b9cacb; font-size: 11px;")
        card.layout.addWidget(lbl_ww)

        row_ww = QHBoxLayout()
        self.slider_ww = QSlider(Qt.Horizontal)
        self.slider_ww.setRange(1, 4500)
        self.slider_ww.setValue(2500)
        row_ww.addWidget(self.slider_ww, 1)

        self.spin_ww = QDoubleSpinBox()
        self.spin_ww.setRange(1, 4500)
        self.spin_ww.setValue(2500)
        self.spin_ww.setDecimals(0)
        self.spin_ww.setFixedWidth(64)
        row_ww.addWidget(self.spin_ww)
        card.layout.addLayout(row_ww)

        # Window Level (WL) Slider
        lbl_wl = QLabel("Window Level (WL):")
        lbl_wl.setStyleSheet("color: #b9cacb; font-size: 11px;")
        card.layout.addWidget(lbl_wl)

        row_wl = QHBoxLayout()
        self.slider_wl = QSlider(Qt.Horizontal)
        self.slider_wl.setRange(-1024, 2500)
        self.slider_wl.setValue(500)
        row_wl.addWidget(self.slider_wl, 1)

        self.spin_wl = QDoubleSpinBox()
        self.spin_wl.setRange(-1024, 2500)
        self.spin_wl.setValue(500)
        self.spin_wl.setDecimals(0)
        self.spin_wl.setFixedWidth(64)
        row_wl.addWidget(self.spin_wl)
        card.layout.addLayout(row_wl)

        self.slider_ww.valueChanged.connect(self.spin_ww.setValue)
        self.spin_ww.valueChanged.connect(self.slider_ww.setValue)
        self.slider_wl.valueChanged.connect(self.spin_wl.setValue)
        self.spin_wl.valueChanged.connect(self.slider_wl.setValue)

        self.slider_ww.valueChanged.connect(self._emit_window_level)
        self.slider_wl.valueChanged.connect(self._emit_window_level)

        self.content_layout.addWidget(card)

    def _build_volume_3d_section(self) -> None:
        """Builds 3D Volume Rendering Controls."""
        card = SectionCard("3D GPU Volume Rendering", self)

        # Preset Dropdown
        self.combo_3d = QComboBox()
        for p in VOLUME_3D_PRESETS:
            self.combo_3d.addItem(p.name, p.id)
        self.combo_3d.currentIndexChanged.connect(self._on_3d_preset_changed)
        card.layout.addWidget(self.combo_3d)

        # Opacity Slider
        lbl_op = QLabel("3D Opacity Multiplier:")
        lbl_op.setStyleSheet("color: #b9cacb; font-size: 11px;")
        card.layout.addWidget(lbl_op)

        row_op = QHBoxLayout()
        self.slider_op = QSlider(Qt.Horizontal)
        self.slider_op.setRange(10, 300)
        self.slider_op.setValue(100)
        row_op.addWidget(self.slider_op, 1)

        self.lbl_op_val = QLabel("1.0x")
        self.lbl_op_val.setStyleSheet("color: #00dbe9; font-family: 'JetBrains Mono'; font-size: 11px;")
        row_op.addWidget(self.lbl_op_val)
        card.layout.addLayout(row_op)

        self.slider_op.valueChanged.connect(self._on_opacity_changed)

        # Shading Checkbox
        self.chk_shading = QCheckBox("Enable Raycast Shading")
        self.chk_shading.setChecked(True)
        self.chk_shading.setStyleSheet("color: #dfe3e7; font-size: 11px;")
        self.chk_shading.toggled.connect(self.signals.volume_shading_toggled.emit)
        card.layout.addWidget(self.chk_shading)

        self.content_layout.addWidget(card)

    def _build_segmentation_section(self) -> None:
        """Builds 3D AI Segmentation & Intraoral Scan Alignment Card."""
        card = SectionCard("3D AI Segmentation & Mesh Alignment", self)

        # ---- Load / Import Action Buttons ----
        row_load = QHBoxLayout()
        row_load.setSpacing(4)

        self.btn_load_seg = QPushButton("🧠 Load Seg. Mask")
        self.btn_load_seg.setCursor(Qt.PointingHandCursor)
        self.btn_load_seg.setToolTip("Load AI segmentation mask (.nii / .nii.gz / .nrrd)")
        self.btn_load_seg.setStyleSheet("""
            QPushButton {
                background-color: #212629;
                color: #00dbe9;
                border: 1px solid #3b494b;
                border-radius: 3px;
                padding: 5px 6px;
                font-weight: 600;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #00dbe9;
                color: #090e11;
                border: 1px solid #00dbe9;
            }
        """)
        self.btn_load_seg.clicked.connect(self.signals.load_segmentation_clicked.emit)
        row_load.addWidget(self.btn_load_seg)

        self.btn_import_ios = QPushButton("🦷 Import IOS")
        self.btn_import_ios.setCursor(Qt.PointingHandCursor)
        self.btn_import_ios.setToolTip("Import Intraoral Scan (.stl / .ply)")
        self.btn_import_ios.setStyleSheet("""
            QPushButton {
                background-color: #212629;
                color: #dfe3e7;
                border: 1px solid #3b494b;
                border-radius: 3px;
                padding: 5px 6px;
                font-weight: 600;
                font-size: 10px;
            }
            QPushButton:hover {
                border: 1px solid #4ce5ef;
                color: #4ce5ef;
            }
        """)
        self.btn_import_ios.clicked.connect(self.signals.import_ios_clicked.emit)
        row_load.addWidget(self.btn_import_ios)
        card.layout.addLayout(row_load)

        # ---- ICP Auto-Align Button ----
        self.btn_icp_align = QPushButton("⚡ Auto-Align Scan (ICP)")
        self.btn_icp_align.setCursor(Qt.PointingHandCursor)
        self.btn_icp_align.setToolTip("Register IOS scan onto CBCT teeth via Iterative Closest Point")
        self.btn_icp_align.setStyleSheet("""
            QPushButton {
                background-color: #1a2733;
                color: #4ade80;
                border: 1px solid #2a5a3a;
                border-radius: 3px;
                padding: 5px 8px;
                font-weight: 700;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #4ade80;
                color: #090e11;
                border: 1px solid #4ade80;
            }
            QPushButton:disabled {
                background-color: #171c1f;
                color: #555;
                border: 1px solid #262b2e;
            }
        """)
        self.btn_icp_align.setEnabled(False)
        self.btn_icp_align.clicked.connect(self.signals.run_icp_clicked.emit)
        card.layout.addWidget(self.btn_icp_align)

        # ---- ICP Status Label ----
        self.lbl_icp_status = QLabel("Alignment: Not Run")
        self.lbl_icp_status.setStyleSheet(
            "color: #849495; font-family: 'JetBrains Mono'; font-size: 10px;"
        )
        card.layout.addWidget(self.lbl_icp_status)

        # ---- Structure Visibility Toggles ----
        lbl_vis = QLabel("Structure Visibility:")
        lbl_vis.setStyleSheet("color: #b9cacb; font-size: 10px; margin-top: 4px;")
        card.layout.addWidget(lbl_vis)

        self._seg_checkboxes: dict = {}
        seg_structures = [
            ("mandible",  "Mandible Bone",       "#ebe0c8"),
            ("canal",     "Mandibular Canal",     "#ff4040"),
            ("teeth",     "Teeth / Enamel",       "#f8faff"),
            ("soft",      "Soft Tissue",          "#d9a68c"),
        ]

        for struct_id, label, color_hex in seg_structures:
            chk = QCheckBox(label)
            chk.setChecked(True)
            chk.setStyleSheet(f"""
                QCheckBox {{
                    color: {color_hex};
                    font-size: 10px;
                    spacing: 4px;
                }}
                QCheckBox::indicator {{
                    width: 10px;
                    height: 10px;
                    border: 1px solid {color_hex};
                    border-radius: 2px;
                }}
                QCheckBox::indicator:checked {{
                    background-color: {color_hex};
                }}
            """)
            chk.toggled.connect(
                lambda checked, sid=struct_id: self.signals.mesh_visibility_changed.emit(sid, checked)
            )
            card.layout.addWidget(chk)
            self._seg_checkboxes[struct_id] = chk

        # IOS Scan Visibility
        self.chk_ios_visible = QCheckBox("IOS Scan Overlay")
        self.chk_ios_visible.setChecked(True)
        self.chk_ios_visible.setStyleSheet("""
            QCheckBox {
                color: #4dbfe6;
                font-size: 10px;
                spacing: 4px;
            }
            QCheckBox::indicator {
                width: 10px;
                height: 10px;
                border: 1px solid #4dbfe6;
                border-radius: 2px;
            }
            QCheckBox::indicator:checked {
                background-color: #4dbfe6;
            }
        """)
        self.chk_ios_visible.toggled.connect(self.signals.ios_visibility_changed.emit)
        card.layout.addWidget(self.chk_ios_visible)

        # ---- Opacity Sliders ----
        lbl_mesh_op = QLabel("Mesh Opacity:")
        lbl_mesh_op.setStyleSheet("color: #849495; font-size: 10px; margin-top: 2px;")
        card.layout.addWidget(lbl_mesh_op)

        row_mesh_op = QHBoxLayout()
        self.slider_mesh_opacity = QSlider(Qt.Horizontal)
        self.slider_mesh_opacity.setRange(0, 100)
        self.slider_mesh_opacity.setValue(85)
        row_mesh_op.addWidget(self.slider_mesh_opacity, 1)

        self.lbl_mesh_op_val = QLabel("85%")
        self.lbl_mesh_op_val.setStyleSheet(
            "color: #00dbe9; font-family: 'JetBrains Mono'; font-size: 10px; font-weight: bold;"
        )
        row_mesh_op.addWidget(self.lbl_mesh_op_val)
        card.layout.addLayout(row_mesh_op)

        self.slider_mesh_opacity.valueChanged.connect(self._on_mesh_opacity_changed)

        lbl_ios_op = QLabel("IOS Scan Opacity:")
        lbl_ios_op.setStyleSheet("color: #849495; font-size: 10px;")
        card.layout.addWidget(lbl_ios_op)

        row_ios_op = QHBoxLayout()
        self.slider_ios_opacity = QSlider(Qt.Horizontal)
        self.slider_ios_opacity.setRange(0, 100)
        self.slider_ios_opacity.setValue(65)
        row_ios_op.addWidget(self.slider_ios_opacity, 1)

        self.lbl_ios_op_val = QLabel("65%")
        self.lbl_ios_op_val.setStyleSheet(
            "color: #4dbfe6; font-family: 'JetBrains Mono'; font-size: 10px; font-weight: bold;"
        )
        row_ios_op.addWidget(self.lbl_ios_op_val)
        card.layout.addLayout(row_ios_op)

        self.slider_ios_opacity.valueChanged.connect(self._on_ios_opacity_changed)

        self.content_layout.addWidget(card)

    def _on_mesh_opacity_changed(self, value: int) -> None:
        """Broadcasts mesh opacity change for all structures."""
        opacity = value / 100.0
        self.lbl_mesh_op_val.setText(f"{value}%")
        for struct_id in ("mandible", "canal", "teeth", "soft"):
            self.signals.mesh_opacity_changed.emit(struct_id, opacity)

    def _on_ios_opacity_changed(self, value: int) -> None:
        """Broadcasts IOS scan opacity change."""
        opacity = value / 100.0
        self.lbl_ios_op_val.setText(f"{value}%")
        self.signals.ios_opacity_changed.emit(opacity)

    def update_icp_status(
        self,
        rms: float,
        iterations: int,
        quality_status: str = "EXCELLENT",
        max_95th: Optional[float] = None,
    ) -> None:
        """Update the ICP alignment status label with clinical quality metrics."""
        color_map = {
            "EXCELLENT": "#4ade80",   # Green
            "ACCEPTABLE": "#a3e635",  # Lime
            "WARNING": "#fbbf24",     # Amber
            "FAILED": "#f87171",      # Red
        }
        badge_color = color_map.get(quality_status, "#4ade80")
        icon = "✅" if quality_status in ("EXCELLENT", "ACCEPTABLE") else ("⚠️" if quality_status == "WARNING" else "❌")

        status_text = f"{icon} RMS: {rms:.3f} mm [{quality_status}]"
        if max_95th is not None:
            status_text += f"\n   95%: {max_95th:.3f} mm | Iters: {iterations}"
        else:
            status_text += f" | {iterations} iters"

        self.lbl_icp_status.setText(status_text)
        self.lbl_icp_status.setStyleSheet(
            f"color: {badge_color}; font-family: 'JetBrains Mono'; font-size: 10px; font-weight: bold;"
        )

    def set_icp_button_enabled(self, enabled: bool) -> None:
        """Enable/disable ICP button (requires both teeth and IOS loaded)."""
        self.btn_icp_align.setEnabled(enabled)

    def _build_tools_section(self) -> None:
        """Builds Diagnostic Tools & Caliper buttons."""
        card = SectionCard("Diagnostic Tools", self)

        self.chk_crosshair = QCheckBox("Show Linked Crosshairs")
        self.chk_crosshair.setChecked(True)
        self.chk_crosshair.setStyleSheet("color: #dfe3e7; font-size: 11px;")
        self.chk_crosshair.toggled.connect(self.signals.crosshair_toggled.emit)
        card.layout.addWidget(self.chk_crosshair)

        # Caliper Measurement
        row_meas = QHBoxLayout()
        self.btn_measure = QPushButton("📏 Measure (mm)")
        self.btn_measure.setToolTip("Click two points on any 2D slice to measure physical distance in millimeters.")
        self.btn_measure.clicked.connect(self.signals.start_measurement_clicked.emit)
        row_meas.addWidget(self.btn_measure)

        self.btn_clear_meas = QPushButton("Clear")
        self.btn_clear_meas.clicked.connect(self.signals.clear_measurement_clicked.emit)
        row_meas.addWidget(self.btn_clear_meas)
        card.layout.addLayout(row_meas)

        # Reset All Views
        self.btn_reset_views = QPushButton("↺ Reset All Views")
        self.btn_reset_views.clicked.connect(self.signals.reset_views_clicked.emit)
        card.layout.addWidget(self.btn_reset_views)

        self.content_layout.addWidget(card)

    def set_volume_data(self, volume: VolumeData) -> None:
        """Populate metadata and configure navigator bounds."""
        self.volume_data = volume
        meta = volume.metadata

        self.lbl_patient_name.setText(f"Name: {meta.patient_name}")
        self.lbl_patient_id.setText(f"ID: {meta.patient_id}")
        self.lbl_dimensions.setText(f"Matrix: {volume.nx} × {volume.ny} × {volume.nz}")
        self.lbl_spacing.setText(f"Voxel: {volume.spacing[0]:.2f} × {volume.spacing[1]:.2f} × {volume.spacing[2]:.2f} mm")
        self.lbl_fov.setText(f"FoV: {volume.physical_size_mm[0]:.1f} × {volume.physical_size_mm[1]:.1f} × {volume.physical_size_mm[2]:.1f} mm")

        # Reset slider ranges
        self.slider_axial.blockSignals(True)
        self.slider_coronal.blockSignals(True)
        self.slider_sagittal.blockSignals(True)

        self.slider_axial.setRange(0, volume.nz - 1)
        self.spin_axial.setRange(0, volume.nz - 1)
        self.slider_axial.setValue(volume.nz // 2)
        self.spin_axial.setValue(volume.nz // 2)
        self.lbl_axial_mm.setText(f"{(volume.nz // 2) * volume.spacing[2]:.1f} mm")

        self.slider_coronal.setRange(0, volume.ny - 1)
        self.spin_coronal.setRange(0, volume.ny - 1)
        self.slider_coronal.setValue(volume.ny // 2)
        self.spin_coronal.setValue(volume.ny // 2)
        self.lbl_coronal_mm.setText(f"{(volume.ny // 2) * volume.spacing[1]:.1f} mm")

        self.slider_sagittal.setRange(0, volume.nx - 1)
        self.spin_sagittal.setRange(0, volume.nx - 1)
        self.slider_sagittal.setValue(volume.nx // 2)
        self.spin_sagittal.setValue(volume.nx // 2)
        self.lbl_sagittal_mm.setText(f"{(volume.nx // 2) * volume.spacing[0]:.1f} mm")

        self.slider_axial.blockSignals(False)
        self.slider_coronal.blockSignals(False)
        self.slider_sagittal.blockSignals(False)

        self.update_window_level_values(meta.window_width, meta.window_center)

    def update_slice_index(self, plane: str, slice_idx: int) -> None:
        """Update slider position from external slice navigation event without re-triggering."""
        if self.volume_data is None:
            return

        if plane == "axial":
            self.slider_axial.blockSignals(True)
            self.spin_axial.blockSignals(True)
            self.slider_axial.setValue(slice_idx)
            self.spin_axial.setValue(slice_idx)
            self.lbl_axial_mm.setText(f"{slice_idx * self.volume_data.spacing[2]:.1f} mm")
            self.slider_axial.blockSignals(False)
            self.spin_axial.blockSignals(False)

        elif plane == "coronal":
            self.slider_coronal.blockSignals(True)
            self.spin_coronal.blockSignals(True)
            self.slider_coronal.setValue(slice_idx)
            self.spin_coronal.setValue(slice_idx)
            self.lbl_coronal_mm.setText(f"{slice_idx * self.volume_data.spacing[1]:.1f} mm")
            self.slider_coronal.blockSignals(False)
            self.spin_coronal.blockSignals(False)

        elif plane == "sagittal":
            self.slider_sagittal.blockSignals(True)
            self.spin_sagittal.blockSignals(True)
            self.slider_sagittal.setValue(slice_idx)
            self.spin_sagittal.setValue(slice_idx)
            self.lbl_sagittal_mm.setText(f"{slice_idx * self.volume_data.spacing[0]:.1f} mm")
            self.slider_sagittal.blockSignals(False)
            self.spin_sagittal.blockSignals(False)

    def update_window_level_values(self, ww: float, wl: float) -> None:
        """Update WW/WL controls from viewport drag event."""
        self.slider_ww.blockSignals(True)
        self.spin_ww.blockSignals(True)
        self.slider_wl.blockSignals(True)
        self.spin_wl.blockSignals(True)

        self.slider_ww.setValue(int(ww))
        self.spin_ww.setValue(ww)
        self.slider_wl.setValue(int(wl))
        self.spin_wl.setValue(wl)

        self.slider_ww.blockSignals(False)
        self.spin_ww.blockSignals(False)
        self.slider_wl.blockSignals(False)
        self.spin_wl.blockSignals(False)

    def _emit_window_level(self) -> None:
        ww = float(self.slider_ww.value())
        wl = float(self.slider_wl.value())
        self.signals.window_level_changed.emit(ww, wl)

    def _on_preset_clicked(self, key: str) -> None:
        if key in WL_PRESETS:
            p = WL_PRESETS[key]
            self.update_window_level_values(p.window_width, p.window_level)
            self.signals.window_level_changed.emit(p.window_width, p.window_level)

    def _on_3d_preset_changed(self, index: int) -> None:
        preset_id = self.combo_3d.itemData(index)
        if preset_id:
            self.signals.volume_preset_changed.emit(str(preset_id))

    def _on_opacity_changed(self, value: int) -> None:
        mult = value / 100.0
        self.lbl_op_val.setText(f"{mult:.1f}x")
        self.signals.volume_opacity_changed.emit(mult)
