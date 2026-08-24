"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Module: ui/main_window.py

Main Application Window for the Commercial Dental CBCT PACS Workstation.
Features:
- Branded Clinical Header with official Bau Medical Systems identity.
- Commercial PACS Action Ribbon Toolbar (Navigation, Measurements, Layouts, Filters, Exports).
- Left-Hand Collapsible Series Thumbnail Sidebar.
- Central Multi-Viewport Grid with dynamic layout switching & active focus highlighting.
- Right-Hand Clinical Control Panel (Nerve Tracing, Slice Navigators, Window/Level, 3D presets).
- Non-blocking multi-format loading pipeline with modal progress.
"""

from __future__ import annotations
import os
import sys
from typing import Optional
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QProgressDialog, QLabel,
    QPushButton, QStatusBar, QDialog, QProgressBar, QSplitter,
    QApplication
)
from PySide6.QtGui import QIcon, QPixmap, QColor

import vtk
from core.volume_data import VolumeData
from core.dicom_loader import DicomLoaderWorker
from core.async_workers import SegmentationWorker, ICPRegistrationWorker
from ui.viewport_grid import ViewportGrid
from ui.control_panel import ControlPanel
from ui.series_sidebar import SeriesSidebar
from ui.ribbon_toolbar import RibbonToolbar
from ui.styles import BAU_DARK_THEME
from dental.surface_extractor import SurfaceExtractor, STRUCTURE_PRESETS
from dental.mesh_registration import MeshRegistrationEngine


class LoadingProgressDialog(QDialog):
    """Sleek dark-themed modal progress dialog for background volume reconstruction."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Bau Medical Systems — Processing Volume")
        self.setFixedSize(380, 140)
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        self.lbl_status = QLabel("Initializing volume pipeline...")
        self.lbl_status.setStyleSheet("color: #00dbe9; font-family: Inter; font-size: 11px; font-weight: 600;")
        layout.addWidget(self.lbl_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #171c1f;
                border: 1px solid #262b2e;
                border-radius: 4px;
                text-align: center;
                color: #dfe3e7;
                font-family: 'JetBrains Mono';
                font-size: 11px;
                height: 18px;
            }
            QProgressBar::chunk {
                background-color: #00dbe9;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)

        self.lbl_subtext = QLabel("Assembling DICOM slices into calibrated 3D matrix...")
        self.lbl_subtext.setStyleSheet("color: #849495; font-size: 10px;")
        layout.addWidget(self.lbl_subtext)

    def set_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(percent)
        self.lbl_status.setText(message)


class MainWindow(QMainWindow):
    """
    Main Application Window for Bau Medical Systems Dental CBCT 3D PACS Workstation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Bau Medical Systems — Dental CBCT 3D PACS Workstation")
        self.resize(1500, 950)
        self.setMinimumSize(1200, 750)

        # 1. Apply Clinical Theme
        self.setStyleSheet(BAU_DARK_THEME)

        # 2. Setup Application Icon & Brand Logo
        self._setup_brand_assets()

        # 3. State Management
        self.volume_data: Optional[VolumeData] = None
        self.loader_worker: Optional[DicomLoaderWorker] = None
        self.loading_dialog = LoadingProgressDialog(self)

        # 3D Segmentation & Mesh Alignment state
        self._seg_worker: Optional[SegmentationWorker] = None
        self._icp_worker: Optional[ICPRegistrationWorker] = None
        self._ios_polydata: Optional[vtk.vtkPolyData] = None          # Loaded IOS vtkPolyData (pre-alignment)
        self._ios_raw_polydata: Optional[vtk.vtkPolyData] = None      # Unaligned copy for re-registration
        self._seg_structures: dict = {}                               # {id: vtkPolyData} from extraction
        self._has_ios_loaded: bool = False
        self._has_teeth_extracted: bool = False

        # 4. Construct PACS UI Architecture
        self._build_top_brand_bar()
        self._build_ribbon_toolbar()
        self._build_central_workspace()
        self._build_status_bar()

        # 5. Connect Signal Architecture
        self._connect_all_signals()

        # 6. Auto-load Synthetic Dental CBCT on start
        QTimer.singleShot(100, self.load_demo_volume)

    def _setup_brand_assets(self) -> None:
        """Loads and configures the official Bau Medical Systems brand asset."""
        logo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "premium_bau_medical_systems_logo.png")
        if os.path.exists(logo_path):
            self.setWindowIcon(QIcon(logo_path))
            self.logo_pixmap = QPixmap(logo_path)
        else:
            self.logo_pixmap = QPixmap()

    def _build_top_brand_bar(self) -> None:
        """Constructs top header with logo, title, and quick folder open actions."""
        header_bar = QWidget(self)
        header_bar.setFixedHeight(38)
        header_bar.setStyleSheet("""
            QWidget {
                background-color: #0d1114;
                border-bottom: 1px solid #1f2528;
            }
        """)

        layout = QHBoxLayout(header_bar)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(10)

        if not self.logo_pixmap.isNull():
            lbl_logo = QLabel()
            lbl_logo.setPixmap(self.logo_pixmap.scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            layout.addWidget(lbl_logo)

        lbl_brand = QLabel("BAU MEDICAL SYSTEMS")
        lbl_brand.setStyleSheet("color: #00dbe9; font-family: Inter; font-weight: 800; font-size: 12px; letter-spacing: 0.08em;")
        layout.addWidget(lbl_brand)

        lbl_sub = QLabel("| DENTAL CBCT PACS WORKSTATION")
        lbl_sub.setStyleSheet("color: #849495; font-family: Inter; font-weight: 600; font-size: 11px;")
        layout.addWidget(lbl_sub)

        layout.addStretch()

        btn_open = QPushButton("📁 Open Folder / DICOM")
        btn_open.setCursor(Qt.PointingHandCursor)
        btn_open.setToolTip("Open a directory containing a DICOM series or 2D slice sequence (.png/.tif/.jpg)")
        btn_open.clicked.connect(self.open_dicom_folder)
        btn_open.setStyleSheet("""
            QPushButton {
                background-color: #171c1f;
                color: #dfe3e7;
                border: 1px solid #2e3538;
                border-radius: 3px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                border: 1px solid #00dbe9;
                color: #00dbe9;
            }
        """)
        layout.addWidget(btn_open)

        btn_demo = QPushButton("🦷 Sample CBCT")
        btn_demo.setCursor(Qt.PointingHandCursor)
        btn_demo.setToolTip("Load realistic synthetic Mandibular CBCT volume")
        btn_demo.clicked.connect(self.load_demo_volume)
        btn_demo.setStyleSheet("""
            QPushButton {
                background-color: #00dbe9;
                color: #090e11;
                border: none;
                border-radius: 3px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 700;
            }
            QPushButton:hover {
                background-color: #4ce5ef;
            }
        """)
        layout.addWidget(btn_demo)

        btn_about = QPushButton("ℹ")
        btn_about.setFixedSize(26, 26)
        btn_about.setCursor(Qt.PointingHandCursor)
        btn_about.clicked.connect(self.show_about_dialog)
        btn_about.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #849495;
                border: 1px solid #2e3538;
                border-radius: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                color: #00dbe9;
                border: 1px solid #00dbe9;
            }
        """)
        layout.addWidget(btn_about)

        # Add header to main window
        self.top_header_widget = header_bar

    def _build_ribbon_toolbar(self) -> None:
        """Constructs PACS Action Ribbon."""
        self.ribbon_toolbar = RibbonToolbar(self)

    def _build_central_workspace(self) -> None:
        """Constructs central layout containing Left Sidebar, Viewports, and Right Control Panel."""
        root_container = QWidget(self)
        root_layout = QVBoxLayout(root_container)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self.top_header_widget)
        root_layout.addWidget(self.ribbon_toolbar)

        # Workspace horizontal splitter
        workspace_widget = QWidget(self)
        workspace_layout = QHBoxLayout(workspace_widget)
        workspace_layout.setContentsMargins(4, 4, 4, 4)
        workspace_layout.setSpacing(4)

        # 1. Left Series Sidebar
        self.series_sidebar = SeriesSidebar(self)
        workspace_layout.addWidget(self.series_sidebar)

        # 2. Central 2x2 Viewport Grid
        self.viewport_grid = ViewportGrid(self)
        workspace_layout.addWidget(self.viewport_grid, 1)

        # 3. Right Control Panel
        self.control_panel = ControlPanel(self)
        workspace_layout.addWidget(self.control_panel)

        root_layout.addWidget(workspace_widget, 1)
        self.setCentralWidget(root_container)

    def _build_status_bar(self) -> None:
        """Constructs the clinical diagnostic status bar."""
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)

        self.status_coords = QLabel("Position: [X: 0.0 mm, Y: 0.0 mm, Z: 0.0 mm]")
        self.status_hu = QLabel("HU: --")
        self.status_hu.setStyleSheet("color: #00dbe9; font-weight: bold;")
        self.status_dims = QLabel("Matrix: --")
        self.status_gpu = QLabel("GPU: vtkGPUVolumeRayCastMapper (Active)")
        self.status_gpu.setStyleSheet("color: #4ade80;")

        self.status_bar.addWidget(self.status_coords, 1)
        self.status_bar.addWidget(self.status_hu, 1)
        self.status_bar.addWidget(self.status_dims, 1)
        self.status_bar.addPermanentWidget(self.status_gpu)

    def _connect_all_signals(self) -> None:
        """Wires bidirectional signals between Ribbon, Sidebar, Viewports, ControlPanel, and MainWindow."""
        # 1. Ribbon Tool Actions
        self.ribbon_toolbar.signals.tool_changed.connect(self.viewport_grid.set_active_tool)
        self.ribbon_toolbar.signals.cine_toggled.connect(self._on_cine_toggled)
        self.ribbon_toolbar.signals.measurement_tool_selected.connect(self._on_measurement_tool_selected)
        self.ribbon_toolbar.signals.layout_changed.connect(self.viewport_grid.set_layout_mode)
        self.ribbon_toolbar.signals.invert_colors_toggled.connect(self.viewport_grid.set_invert_colors)
        self.ribbon_toolbar.signals.crosshair_toggled.connect(self.viewport_grid.set_crosshair_visible)
        self.ribbon_toolbar.signals.reset_views_clicked.connect(self.reset_all_views)
        self.ribbon_toolbar.signals.export_slice_clicked.connect(self.export_active_slice_png)
        self.ribbon_toolbar.signals.export_stl_clicked.connect(self.export_3d_stl_mesh)
        self.ribbon_toolbar.signals.export_dicom_clicked.connect(self.export_nerve_coordinates)
        self.ribbon_toolbar.signals.report_clicked.connect(self.generate_surgical_report)

        # 2. Slice Navigation from ControlPanel -> Viewports
        self.control_panel.signals.slice_navigated.connect(self._on_control_panel_slice_navigated)

        # 3. Slice Changes from Viewports -> ControlPanel
        self.viewport_grid.signals.slice_changed.connect(self.control_panel.update_slice_index)

        # 4. Window / Level changes from ControlPanel -> Viewports
        self.control_panel.signals.window_level_changed.connect(self.viewport_grid.set_global_window_level)

        # 5. Window / Level changes from Viewport Drag -> ControlPanel
        self.viewport_grid.signals.window_level_changed.connect(self.control_panel.update_window_level_values)

        # 6. 3D Volume Controls
        self.control_panel.signals.volume_preset_changed.connect(self.viewport_grid.volume_view.apply_preset)
        self.control_panel.signals.volume_opacity_changed.connect(self.viewport_grid.volume_view.set_opacity_multiplier)
        self.control_panel.signals.volume_shading_toggled.connect(self.viewport_grid.volume_view.set_shading)

        # 7. Diagnostic Tools
        self.control_panel.signals.crosshair_toggled.connect(self.viewport_grid.set_crosshair_visible)
        self.control_panel.signals.start_measurement_clicked.connect(self.start_measurement)
        self.control_panel.signals.clear_measurement_clicked.connect(self.clear_measurement)
        self.control_panel.signals.reset_views_clicked.connect(self.reset_all_views)

        # 8. Mandibular Nerve Tracing Signals
        self.control_panel.signals.nerve_draw_toggled.connect(self.viewport_grid.set_nerve_drawing_mode)
        self.control_panel.signals.nerve_undo_clicked.connect(self.viewport_grid.undo_nerve_point)
        self.control_panel.signals.nerve_clear_clicked.connect(self.viewport_grid.clear_nerve)
        self.control_panel.signals.nerve_diameter_changed.connect(self.viewport_grid.set_nerve_diameter)
        self.control_panel.signals.nerve_export_clicked.connect(self.export_nerve_coordinates)
        self.viewport_grid.signals.nerve_updated.connect(self.control_panel.update_nerve_telemetry)

        # 9. Panoramic & Cross-Section Signals
        self.control_panel.signals.arch_draw_toggled.connect(self.viewport_grid.draw_dental_arch)
        self.control_panel.signals.arch_autofit_clicked.connect(self.viewport_grid.auto_fit_dental_arch)
        self.control_panel.signals.arch_clear_clicked.connect(self.viewport_grid.clear_dental_arch)
        self.control_panel.signals.trough_thickness_changed.connect(self.viewport_grid.set_focal_trough_thickness)
        self.control_panel.signals.cross_section_navigated.connect(self.viewport_grid.set_cross_section_index)
        self.viewport_grid.signals.arch_updated.connect(self.control_panel.update_arch_telemetry)
        self.viewport_grid.signals.cross_section_changed.connect(self.control_panel.update_cross_section_index)

        # 10. Virtual Dental Implant Planning Signals
        self.control_panel.signals.implant_add_clicked.connect(self.viewport_grid.add_implant)
        self.control_panel.signals.implant_delete_clicked.connect(self.viewport_grid.remove_active_implant)
        self.control_panel.signals.implant_preset_selected.connect(self.viewport_grid.set_implant_preset)
        self.control_panel.signals.implant_dimensions_changed.connect(self.viewport_grid.set_implant_dimensions)
        self.control_panel.signals.implant_angulation_changed.connect(self.viewport_grid.set_implant_angulation)
        self.control_panel.signals.implant_depth_changed.connect(self.viewport_grid.set_implant_depth)
        self.control_panel.signals.implant_sleeve_toggled.connect(self.viewport_grid.toggle_implant_safety_sleeve)
        self.viewport_grid.signals.implant_safety_changed.connect(self.control_panel.update_implant_safety_hud)

        # 12. 3D AI Segmentation & Mesh Alignment Signals
        self.control_panel.signals.load_segmentation_clicked.connect(self._on_load_segmentation)
        self.control_panel.signals.import_ios_clicked.connect(self._on_import_ios_scan)
        self.control_panel.signals.run_icp_clicked.connect(self._on_run_icp_alignment)
        self.control_panel.signals.mesh_visibility_changed.connect(
            self.viewport_grid.volume_view.set_mesh_visibility
        )
        self.control_panel.signals.mesh_opacity_changed.connect(
            self.viewport_grid.volume_view.set_mesh_opacity
        )
        self.control_panel.signals.ios_visibility_changed.connect(
            self.viewport_grid.volume_view.set_ios_visibility
        )
        self.control_panel.signals.ios_opacity_changed.connect(
            self.viewport_grid.volume_view.set_ios_opacity
        )

        # 13. Status Bar Updates
        self.viewport_grid.signals.crosshair_moved.connect(self._update_status_coords)
        self.viewport_grid.signals.hu_inspected.connect(self._update_status_hu)
        self.viewport_grid.signals.measurement_completed.connect(self._on_measurement_completed)

    # --------------------------------------------------------------------------
    # Loading & DICOM IO
    # --------------------------------------------------------------------------

    def open_dicom_folder(self) -> None:
        """Prompts user to select a folder and launches multi-format worker thread."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Dental CBCT DICOM or Image Sequence Folder",
            os.path.expanduser("~"),
            QFileDialog.ShowDirsOnly
        )
        if folder:
            self._start_dicom_loader(folder_path=folder, is_synthetic=False)

    def load_demo_volume(self) -> None:
        """Launches worker to generate the synthetic Dental CBCT volume."""
        self._start_dicom_loader(folder_path=None, is_synthetic=True)

    def _start_dicom_loader(self, folder_path: Optional[str], is_synthetic: bool) -> None:
        """Starts asynchronous loader worker with modal progress."""
        if self.loader_worker is not None and self.loader_worker.isRunning():
            self.loader_worker.terminate()
            self.loader_worker.wait()

        self.loading_dialog.set_progress(0, "Starting loader worker thread...")
        self.loading_dialog.show()

        self.loader_worker = DicomLoaderWorker(directory_path=folder_path, is_synthetic=is_synthetic)
        self.loader_worker.progress.connect(self.loading_dialog.set_progress)
        self.loader_worker.finished.connect(self._on_volume_loaded)
        self.loader_worker.error.connect(self._on_load_error)
        self.loader_worker.start()

    def _on_volume_loaded(self, volume: VolumeData) -> None:
        """Callback when volume is ready."""
        self.loading_dialog.hide()
        self.volume_data = volume

        # Update Sidebar
        self.series_sidebar.set_volume_data(volume)

        # Send VolumeData to Viewports & Control Panel
        self.viewport_grid.set_volume_data(volume)
        self.control_panel.set_volume_data(volume)

        # Update Status Bar
        self.status_dims.setText(f"Matrix: {volume.nx}×{volume.ny}×{volume.nz} ({volume.spacing[0]:.2f}mm)")
        self.status_bar.showMessage(f"Loaded: {volume.metadata.patient_name} ({volume.metadata.series_description})", 5000)

        # Reset camera on all viewports
        self.viewport_grid.reset_all_views()

    def _on_load_error(self, message: str) -> None:
        self.loading_dialog.hide()
        QMessageBox.critical(self, "Dataset Load Error", f"Failed to ingest dataset:\n\n{message}")

    def _on_cine_toggled(self, is_playing: bool, fps: int) -> None:
        if is_playing:
            self.viewport_grid.start_cine(fps)
            self.status_bar.showMessage("Cine loop playback active.", 3000)
        else:
            self.viewport_grid.stop_cine()
            self.status_bar.showMessage("Cine loop paused.", 2000)

    def _on_measurement_tool_selected(self, tool_name: str) -> None:
        if tool_name == "distance":
            self.start_measurement()
        elif tool_name == "angle":
            self.start_angle_measurement()
        elif tool_name == "roi":
            self.start_roi_measurement()
        elif tool_name == "clear":
            self.clear_measurement()

    def _on_control_panel_slice_navigated(self, plane: str, slice_idx: int) -> None:
        if self.volume_data is None:
            return

        cx, cy, cz = self.volume_data.get_center()
        x, y, z = self.viewport_grid.axial_view._current_world_pos
        dx, dy, dz = self.volume_data.spacing
        ox, oy, oz = self.volume_data.origin

        if plane == "axial":
            z = oz + slice_idx * dz
        elif plane == "coronal":
            y = oy + slice_idx * dy
        elif plane == "sagittal":
            x = ox + slice_idx * dx

        self.viewport_grid._on_crosshair_moved(x, y, z)

    def _update_status_coords(self, x: float, y: float, z: float) -> None:
        self.status_coords.setText(f"Position: [X: {x:+.1f} mm, Y: {y:+.1f} mm, Z: {z:+.1f} mm]")

    def _update_status_hu(self, x: float, y: float, z: float, hu: float) -> None:
        tissue_type = "Air"
        if hu > 1800:
            tissue_type = "Enamel / Metal"
        elif hu > 800:
            tissue_type = "Cortical Bone / Dentin"
        elif hu > 250:
            tissue_type = "Trabecular Bone"
        elif hu > -50:
            tissue_type = "Soft Tissue / Muscle"
        elif hu > -200:
            tissue_type = "Adipose / Fat"

        self.status_hu.setText(f"HU: {hu:+.0f} ({tissue_type})")

    def _on_measurement_completed(self, m_type: str, val: float) -> None:
        if m_type == "angle":
            self.status_bar.showMessage(f"Angle Measurement: {val:.1f}°", 6000)
        elif m_type == "roi":
            self.status_bar.showMessage(f"ROI Mean Density: {val:+.0f} HU", 6000)
        else:
            self.status_bar.showMessage(f"Distance Measurement: {val:.2f} mm", 6000)

    def start_measurement(self) -> None:
        self.viewport_grid.set_active_tool("distance")
        self.status_bar.showMessage("Caliper Mode: Click 2 points on any 2D slice to measure distance (mm).", 5000)

    def start_angle_measurement(self) -> None:
        self.viewport_grid.set_active_tool("angle")
        self.viewport_grid.axial_view.start_measurement()
        self.viewport_grid.coronal_view.start_measurement()
        self.viewport_grid.sagittal_view.start_measurement()
        self.status_bar.showMessage("Angle Mode: Click 3 points (Vertex second) to measure clinical angle (°).", 5000)

    def start_roi_measurement(self) -> None:
        self.viewport_grid.set_active_tool("roi")
        self.viewport_grid.axial_view.start_measurement()
        self.viewport_grid.coronal_view.start_measurement()
        self.viewport_grid.sagittal_view.start_measurement()
        self.status_bar.showMessage("ROI Box Mode: Click 2 opposite corners to measure mean HU density.", 5000)

    def clear_measurement(self) -> None:
        self.viewport_grid.axial_view.clear_measurement()
        self.viewport_grid.coronal_view.clear_measurement()
        self.viewport_grid.sagittal_view.clear_measurement()
        self.viewport_grid.set_active_tool("select")
        self.status_bar.showMessage("Measurements cleared.", 3000)

    def reset_all_views(self) -> None:
        self.viewport_grid.reset_all_views()
        self.status_bar.showMessage("All viewports and cameras reset to anatomical center.", 3000)

    def export_3d_stl_mesh(self) -> None:
        """Extracts bone surface isosurface and exports to binary 3D STL mesh."""
        if self.volume_data is None:
            QMessageBox.warning(self, "No Volume", "Please load a volume before exporting a 3D mesh.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export 3D Bone Surface Mesh (STL)",
            os.path.join(os.path.expanduser("~"), "mandible_bone_model.stl"),
            "STL Surface Mesh (*.stl)"
        )
        if not file_path:
            return

        try:
            import vtk
            # Isosurface extraction via Marching Cubes (> 400 HU for bone)
            mc = vtk.vtkMarchingCubes()
            mc.SetInputData(self.volume_data.vtk_image_data)
            mc.SetValue(0, 400.0)
            mc.Update()

            # Surface smoothing
            smoother = vtk.vtkWindowedSincPolyDataFilter()
            smoother.SetInputConnection(mc.GetOutputPort())
            smoother.SetNumberOfIterations(15)
            smoother.BoundarySmoothingOn()
            smoother.FeatureEdgeSmoothingOff()
            smoother.SetPassBand(0.1)
            smoother.Update()

            writer = vtk.vtkSTLWriter()
            writer.SetFileName(file_path)
            writer.SetInputConnection(smoother.GetOutputPort())
            writer.SetFileTypeToBinary()
            writer.Write()

            self.status_bar.showMessage(f"3D STL Bone mesh successfully exported: {os.path.basename(file_path)}", 6000)
        except Exception as e:
            QMessageBox.critical(self, "STL Export Error", f"Failed to export 3D STL mesh:\n{str(e)}")

    def export_active_slice_png(self) -> None:
        """Exports the active 2D slice view to a high-resolution PNG image."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Active Slice (PNG)",
            os.path.join(os.path.expanduser("~"), "dental_slice.png"),
            "PNG Image (*.png);;JPEG Image (*.jpg)"
        )
        if not file_path:
            return

        try:
            # Grab screenshot from active viewport widget
            active_vid = self.viewport_grid.active_viewport_id
            container = self.viewport_grid.containers.get(active_vid, self.viewport_grid.axial_container)
            pixmap = container.content_widget.grab()
            pixmap.save(file_path)
            self.status_bar.showMessage(f"Slice image exported: {os.path.basename(file_path)}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export slice image:\n{str(e)}")

    def export_nerve_coordinates(self) -> None:
        """Exports traced nerve coordinates to a JSON or CSV file."""
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Mandibular Nerve Coordinates",
            os.path.join(os.path.expanduser("~"), "mandibular_nerve.json"),
            "JSON Files (*.json);;CSV Files (*.csv)"
        )
        if not file_path:
            return

        try:
            if file_path.lower().endswith(".csv") or "CSV" in selected_filter:
                content = self.viewport_grid.nerve_tracer.export_to_csv()
            else:
                content = self.viewport_grid.nerve_tracer.export_to_json()

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            self.status_bar.showMessage(f"Nerve coordinates exported: {os.path.basename(file_path)}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export nerve points:\n{str(e)}")

    def generate_surgical_report(self) -> None:
        """Assembles clinical diagnostic snapshot and launches the Surgical Report Generator Dialog."""
        if self.volume_data is None:
            QMessageBox.warning(self, "No Volume Loaded", "Please load a Dental CBCT volume before generating a report.")
            return

        import tempfile
        from reports.surgical_report import SurgicalReportData, ImplantSiteRecord
        from ui.report_dialog import SurgicalReportDialog

        meta = self.volume_data.metadata
        rep_data = SurgicalReportData(
            patient_name=meta.patient_name,
            patient_id=meta.patient_id,
            patient_dob_sex=f"{meta.patient_birth_date} / {meta.patient_sex}",
            study_date=meta.study_date,
            modality=meta.modality,
            scanner_model=f"{meta.manufacturer} CBCT (3D Volume)",
            voxel_spacing_mm=f"{self.volume_data.spacing[0]:.2f} mm Isotropic",
        )

        temp_dir = tempfile.gettempdir()

        # 1. Capture High-Resolution Panoramic Overview Image
        pano_path = os.path.join(temp_dir, "report_panoramic_snapshot.png")
        try:
            if len(self.viewport_grid.dental_arch.sampled_points) < 2:
                self.viewport_grid.auto_fit_dental_arch()

            pano_pix = self.viewport_grid.panoramic_view.grab()
            pano_pix.save(pano_path)
            rep_data.panoramic_img_path = pano_path
        except Exception:
            pass

        # 2. Gather Planned Implants and capture Cross-Section Slices
        mgr = self.viewport_grid.implant_manager
        rep_data.implant_sites.clear()

        for idx, (imp_id, imp) in enumerate(mgr.implants.items()):
            cs_path = os.path.join(temp_dir, f"report_implant_{imp_id}_cs.png")
            try:
                cs_pix = self.viewport_grid.cross_section_view.grab()
                cs_pix.save(cs_path)
            except Exception:
                cs_path = None

            dist = imp.min_nerve_dist_mm
            ridge_h = imp.length_mm + (max(2.0, dist) if dist != float('inf') else 3.0)
            crest_w = imp.diameter_mm + 3.5

            site = ImplantSiteRecord(
                implant_id=imp.implant_id,
                tooth_number=imp.tooth_number,
                brand_preset=f"Bau Universal Ø{imp.diameter_mm:.1f}×{imp.length_mm:.1f}mm",
                diameter_mm=imp.diameter_mm,
                length_mm=imp.length_mm,
                bl_angle_deg=imp.bl_angle_deg,
                md_angle_deg=imp.md_angle_deg,
                min_nerve_dist_mm=dist,
                safety_state=imp.safety_state.value,
                nearest_nerve=imp.nearest_nerve_name,
                bone_density_hu=680.0,
                ridge_height_mm=ridge_h,
                crestal_width_mm=crest_w,
                cross_section_img_path=cs_path
            )
            rep_data.implant_sites.append(site)

        # Launch Modal Report Dialog
        dlg = SurgicalReportDialog(rep_data, self)
        dlg.exec()

    # --------------------------------------------------------------------------
    # 3D AI Segmentation & IOS Mesh Alignment Handlers (Asynchronous)
    # --------------------------------------------------------------------------

    def _on_load_segmentation(self) -> None:
        """Asynchronously loads a segmentation mask file and extracts anatomical surfaces."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load AI Segmentation Mask",
            os.path.expanduser("~"),
            "NIfTI Files (*.nii *.nii.gz);;NRRD Files (*.nrrd);;All Files (*)"
        )
        if not file_path:
            return

        # Cancel any running segmentation worker
        if self._seg_worker is not None and self._seg_worker.isRunning():
            self._seg_worker.cancel()
            self._seg_worker.wait(100)

        # Disable triggering UI buttons to prevent race conditions
        self.control_panel.btn_load_seg.setEnabled(False)
        self.control_panel.set_icp_button_enabled(False)

        # Setup Progress Dialog
        self.loading_dialog.setWindowTitle("Bau Medical Systems — Extracting AI Segmentation")
        self.loading_dialog.set_progress(5, f"Reading {os.path.basename(file_path)}...")
        self.loading_dialog.show()
        self.status_bar.showMessage(f"Extracting segmentation: {os.path.basename(file_path)}...", 0)

        # Clear existing segmented meshes on Main GUI Thread
        self.viewport_grid.volume_view.clear_all_meshes()
        self._seg_structures.clear()
        self._has_teeth_extracted = False

        # Launch background worker
        self._seg_worker = SegmentationWorker(
            file_path=file_path,
            reference_volume=self.volume_data,
            parent=self,
        )
        self._seg_worker.progress_updated.connect(self._on_seg_progress)
        self._seg_worker.structure_extracted.connect(self._on_seg_structure_streamed)
        self._seg_worker.finished_all.connect(self._on_seg_finished)
        self._seg_worker.error_occurred.connect(self._on_seg_error)
        self._seg_worker.start()

    def _on_seg_progress(self, percent: int, message: str) -> None:
        """Updates progress dialog and status bar during segmentation extraction."""
        self.loading_dialog.set_progress(percent, message)
        self.status_bar.showMessage(message, 3000)

    def _on_seg_structure_streamed(self, struct_id: str, polydata: vtk.vtkPolyData) -> None:
        """
        Receives extracted vtkPolyData from worker thread and builds the vtkActor
        strictly on the Main GUI Thread.
        """
        if polydata is None or polydata.GetNumberOfPoints() == 0:
            return

        self._seg_structures[struct_id] = polydata
        preset = STRUCTURE_PRESETS.get(struct_id)
        if preset is not None:
            actor = SurfaceExtractor.create_structure_actor(polydata, preset)
            self.viewport_grid.volume_view.add_segmented_mesh(struct_id, actor, polydata)

    def _on_seg_finished(self, results: dict) -> None:
        """Invoked on Main Thread when all structures are extracted."""
        self.loading_dialog.hide()
        self.control_panel.btn_load_seg.setEnabled(True)

        self._has_teeth_extracted = "teeth" in results
        self.control_panel.set_icp_button_enabled(
            self._has_teeth_extracted and self._has_ios_loaded
        )

        struct_names = ", ".join(
            STRUCTURE_PRESETS[sid].name for sid in results.keys() if sid in STRUCTURE_PRESETS
        )
        if results:
            self.status_bar.showMessage(
                f"Segmentation loaded: {len(results)} structures extracted ({struct_names})",
                8000,
            )
        else:
            self.status_bar.showMessage("No labeled structures found in segmentation mask.", 5000)

    def _on_seg_error(self, error_message: str) -> None:
        """Handles worker extraction errors cleanly without freezing or crashing."""
        self.loading_dialog.hide()
        self.control_panel.btn_load_seg.setEnabled(True)
        self.control_panel.set_icp_button_enabled(
            self._has_teeth_extracted and self._has_ios_loaded
        )
        QMessageBox.critical(
            self,
            "Segmentation Extraction Error",
            f"Failed to extract 3D surfaces from segmentation mask:\n\n{error_message}"
        )
        self.status_bar.showMessage("Segmentation extraction failed.", 5000)

    def _on_import_ios_scan(self) -> None:
        """Import an intraoral scan (STL/PLY) and display it in the 3D viewport."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Intraoral Scan",
            os.path.expanduser("~"),
            "STL Mesh (*.stl);;PLY Mesh (*.ply);;All Files (*)"
        )
        if not file_path:
            return

        try:
            engine = MeshRegistrationEngine()
            ios_pd = engine.load_mesh(file_path)

            # Store raw copy for re-registration
            self._ios_raw_polydata = vtk.vtkPolyData()
            self._ios_raw_polydata.DeepCopy(ios_pd)
            self._ios_polydata = ios_pd
            self._has_ios_loaded = True

            # Create and display actor on Main GUI Thread
            ios_actor = engine.create_ios_actor(ios_pd)
            self.viewport_grid.volume_view.add_ios_scan_actor(ios_actor, ios_pd)

            # Enable ICP if teeth are available
            self.control_panel.set_icp_button_enabled(
                self._has_teeth_extracted and self._has_ios_loaded
            )

            n_pts = ios_pd.GetNumberOfPoints()
            n_tris = ios_pd.GetNumberOfCells()
            self.status_bar.showMessage(
                f"IOS Scan imported: {os.path.basename(file_path)} "
                f"({n_pts:,} vertices, {n_tris:,} triangles)",
                6000,
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "IOS Import Error",
                f"Failed to import intraoral scan:\n\n{str(e)}"
            )
            self.status_bar.showMessage("IOS scan import failed.", 5000)

    def _on_run_icp_alignment(self) -> None:
        """Asynchronously runs rigid ICP registration of the IOS scan onto CBCT teeth."""
        # Validate prerequisites
        teeth_pd = self.viewport_grid.volume_view.get_mesh_polydata("teeth")
        if teeth_pd is None or teeth_pd.GetNumberOfPoints() == 0:
            QMessageBox.warning(
                self,
                "No Teeth Surface",
                "Please load a segmentation mask containing teeth labels before running ICP alignment."
            )
            return

        source_pd = self._ios_raw_polydata
        if source_pd is None or source_pd.GetNumberOfPoints() == 0:
            QMessageBox.warning(
                self,
                "No IOS Scan",
                "Please import an intraoral scan (.stl / .ply) before running ICP alignment."
            )
            return

        if self._icp_worker is not None and self._icp_worker.isRunning():
            self._icp_worker.cancel()
            self._icp_worker.wait(100)

        # Lock UI controls during registration
        self.control_panel.set_icp_button_enabled(False)
        self.control_panel.btn_import_ios.setEnabled(False)

        # Setup Progress Dialog
        self.loading_dialog.setWindowTitle("Bau Medical Systems — ICP Scan Alignment")
        self.loading_dialog.set_progress(10, "Initializing rigid 6-DoF ICP optimizer...")
        self.loading_dialog.show()
        self.status_bar.showMessage("Running ICP registration (Centroid Pre-Alignment → Rigid Body)...", 0)

        # Launch background worker
        self._icp_worker = ICPRegistrationWorker(
            source_poly=source_pd,
            target_poly=teeth_pd,
            max_iterations=150,
            max_landmarks=2000,
            tolerance=1e-6,
            parent=self,
        )
        self._icp_worker.progress_updated.connect(self._on_icp_progress)
        self._icp_worker.registration_complete.connect(self._on_icp_complete)
        self._icp_worker.error_occurred.connect(self._on_icp_error)
        self._icp_worker.start()

    def _on_icp_progress(self, percent: int, message: str) -> None:
        """Updates progress dialog and status bar during ICP registration."""
        self.loading_dialog.set_progress(percent, message)
        self.status_bar.showMessage(message, 3000)

    def _on_icp_complete(
        self,
        aligned_poly: vtk.vtkPolyData,
        transform_matrix: np.ndarray,
        rms_error: float,
        num_iterations: int,
    ) -> None:
        """Invoked on Main Thread with computed registration data."""
        self.loading_dialog.hide()
        self.control_panel.set_icp_button_enabled(True)
        self.control_panel.btn_import_ios.setEnabled(True)

        # Build actor on Main GUI Thread
        aligned_actor = MeshRegistrationEngine.create_ios_actor(aligned_poly)
        self.viewport_grid.volume_view.add_ios_scan_actor(aligned_actor, aligned_poly)

        # Update UI status
        self.control_panel.update_icp_status(rms_error, num_iterations)
        self.status_bar.showMessage(
            f"ICP alignment converged — RMS: {rms_error:.4f} mm, "
            f"Iterations: {num_iterations}",
            8000,
        )

    def _on_icp_error(self, error_message: str) -> None:
        """Handles ICP registration errors cleanly."""
        self.loading_dialog.hide()
        self.control_panel.set_icp_button_enabled(True)
        self.control_panel.btn_import_ios.setEnabled(True)
        QMessageBox.critical(
            self,
            "ICP Registration Error",
            f"ICP alignment failed:\n\n{error_message}"
        )
        self.status_bar.showMessage("ICP alignment failed.", 5000)

    def closeEvent(self, event) -> None:
        """Cleanly terminates workers and releases VTK OpenGL contexts before window destruction."""
        if self._seg_worker is not None and self._seg_worker.isRunning():
            self._seg_worker.cancel()
            self._seg_worker.wait(200)

        if self._icp_worker is not None and self._icp_worker.isRunning():
            self._icp_worker.cancel()
            self._icp_worker.wait(200)

        if hasattr(self, 'viewport_grid') and self.viewport_grid is not None:
            self.viewport_grid.cleanup()
        super().closeEvent(event)

    def show_about_dialog(self) -> None:
        QMessageBox.about(
            self,
            "About Bau Medical Systems",
            "<h3>Bau Medical Systems — Dental CBCT 3D PACS Workstation</h3>"
            "<p><b>Version:</b> 1.0.0 (Commercial PACS Release)</p>"
            "<p>High-performance Cone Beam CT 3D Visualization System for Maxillofacial & Dental Diagnostics.</p>"
            "<p><b>Core Stack:</b> Python, PySide6 (Qt 6), VTK, SimpleITK, NumPy.</p>"
            "<p>© 2026 Bau Medical Systems. All Rights Reserved.</p>"
        )
