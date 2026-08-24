"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Module: ui/report_dialog.py

Interactive PySide6 Report Preview & Export Dialog.
Features:
- Form controls for customizing clinic name, operating surgeon, and clinical observations.
- Live, high-resolution vector PDF page rendering via QPdfDocument / QPdfView.
- Zoom in / Zoom out / Fit to Page interactive navigation.
- Export to permanent PDF file and 1-click system viewer launch.
"""

from __future__ import annotations
import os
import tempfile
import subprocess
import sys
from typing import Optional

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QTextEdit, QPushButton, QFileDialog, QFrame,
    QSplitter, QMessageBox, QComboBox
)
from PySide6.QtPdf import QPdfDocument
from PySide6.QtPdfWidgets import QPdfView

from reports.surgical_report import SurgicalReportData, SurgicalReportGenerator


class SurgicalReportDialog(QDialog):
    """
    Modal dialog allowing the clinician to preview, edit clinical parameters, and export the PDF report.
    """

    def __init__(self, report_data: SurgicalReportData, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.report_data = report_data
        self.temp_pdf_path = os.path.join(tempfile.gettempdir(), "bau_surgical_plan_preview.pdf")

        self.setWindowTitle("Bau Medical Systems — Dental Surgical PDF Report")
        self.resize(1100, 780)
        self.setMinimumSize(900, 600)
        self.setStyleSheet("""
            QDialog {
                background-color: #0f1417;
                color: #dfe3e7;
            }
            QLabel {
                color: #dfe3e7;
                font-family: Inter;
            }
            QLineEdit, QTextEdit {
                background-color: #171c1f;
                color: #ffffff;
                border: 1px solid #262b2e;
                border-radius: 4px;
                padding: 6px;
                font-size: 11px;
            }
            QLineEdit:focus, QTextEdit:focus {
                border: 1px solid #00dbe9;
            }
        """)

        self._build_ui()
        self._generate_and_render_preview()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header Title Bar
        header = QHBoxLayout()
        title_lbl = QLabel("📑 DENTAL SURGICAL PLANNING REPORT GENERATOR")
        title_lbl.setStyleSheet("font-size: 14px; font-weight: bold; color: #00dbe9; letter-spacing: 0.05em;")
        header.addWidget(title_lbl)
        header.addStretch()

        subtitle_lbl = QLabel(f"Patient: {self.report_data.patient_name} ({self.report_data.patient_id})")
        subtitle_lbl.setStyleSheet("color: #849495; font-size: 12px; font-family: 'JetBrains Mono';")
        header.addWidget(subtitle_lbl)
        layout.addLayout(header)

        # Splitter: Left Config Form / Right PDF View
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #262b2e;
                width: 2px;
            }
        """)

        # -------------------------------------------------------------
        # Left Configuration Pane
        # -------------------------------------------------------------
        left_widget = QWidget(self)
        left_widget.setFixedWidth(320)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 10, 0)
        left_layout.setSpacing(8)

        # Clinic Information
        lbl_c_sec = QLabel("CLINIC & SURGEON DEMOGRAPHICS")
        lbl_c_sec.setStyleSheet("font-weight: bold; font-size: 10px; color: #00dbe9;")
        left_layout.addWidget(lbl_c_sec)

        left_layout.addWidget(QLabel("Clinic Name:"))
        self.txt_clinic_name = QLineEdit(self.report_data.clinic_name)
        left_layout.addWidget(self.txt_clinic_name)

        left_layout.addWidget(QLabel("Clinic Address & Phone:"))
        self.txt_clinic_addr = QLineEdit(f"{self.report_data.clinic_address} | {self.report_data.clinic_phone}")
        left_layout.addWidget(self.txt_clinic_addr)

        left_layout.addWidget(QLabel("Operating Surgeon:"))
        self.txt_surgeon = QLineEdit(self.report_data.surgeon_name)
        left_layout.addWidget(self.txt_surgeon)

        left_layout.addWidget(QLabel("Surgeon Title / Specialization:"))
        self.txt_surgeon_title = QLineEdit(self.report_data.surgeon_title)
        left_layout.addWidget(self.txt_surgeon_title)

        # Clinical Notes
        lbl_n_sec = QLabel("CLINICAL ASSESSMENT & NOTES")
        lbl_n_sec.setStyleSheet("font-weight: bold; font-size: 10px; color: #00dbe9; margin-top: 6px;")
        left_layout.addWidget(lbl_n_sec)

        self.txt_notes = QTextEdit()
        self.txt_notes.setPlainText(self.report_data.clinical_notes)
        self.txt_notes.setFixedHeight(120)
        left_layout.addWidget(self.txt_notes)

        # Refresh Preview Button
        self.btn_refresh = QPushButton("🔄 Refresh Preview")
        self.btn_refresh.setCursor(Qt.PointingHandCursor)
        self.btn_refresh.setStyleSheet("""
            QPushButton {
                background-color: #212629;
                color: #00dbe9;
                border: 1px solid #3b494b;
                border-radius: 4px;
                padding: 8px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #00dbe9;
                color: #090e11;
                border: 1px solid #00dbe9;
            }
        """)
        self.btn_refresh.clicked.connect(self._on_refresh_clicked)
        left_layout.addWidget(self.btn_refresh)

        left_layout.addStretch()
        splitter.addWidget(left_widget)

        # -------------------------------------------------------------
        # Right PDF Live Preview Pane
        # -------------------------------------------------------------
        right_widget = QWidget(self)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 0, 0, 0)
        right_layout.setSpacing(6)

        # PDF Toolbar (Zoom controls & Page navigation)
        pdf_toolbar = QHBoxLayout()
        pdf_toolbar.setSpacing(6)

        self.btn_zoom_out = QPushButton("🔍 -")
        self.btn_zoom_out.setFixedSize(36, 26)
        self.btn_zoom_out.setStyleSheet("background-color: #171c1f; border: 1px solid #262b2e; border-radius: 3px;")
        self.btn_zoom_out.clicked.connect(self._zoom_out)
        pdf_toolbar.addWidget(self.btn_zoom_out)

        self.btn_zoom_in = QPushButton("🔍 +")
        self.btn_zoom_in.setFixedSize(36, 26)
        self.btn_zoom_in.setStyleSheet("background-color: #171c1f; border: 1px solid #262b2e; border-radius: 3px;")
        self.btn_zoom_in.clicked.connect(self._zoom_in)
        pdf_toolbar.addWidget(self.btn_zoom_in)

        self.btn_fit_width = QPushButton("Fit Width")
        self.btn_fit_width.setFixedHeight(26)
        self.btn_fit_width.setStyleSheet("background-color: #171c1f; border: 1px solid #262b2e; border-radius: 3px; padding: 0 8px;")
        self.btn_fit_width.clicked.connect(self._fit_width)
        pdf_toolbar.addWidget(self.btn_fit_width)

        pdf_toolbar.addStretch()

        self.lbl_page_status = QLabel("Page 1 of 1")
        self.lbl_page_status.setStyleSheet("color: #849495; font-size: 11px;")
        pdf_toolbar.addWidget(self.lbl_page_status)

        right_layout.addLayout(pdf_toolbar)

        # QPdfDocument & QPdfView
        self.pdf_document = QPdfDocument(self)
        self.pdf_view = QPdfView(self)
        self.pdf_view.setDocument(self.pdf_document)
        self.pdf_view.setPageMode(QPdfView.PageMode.MultiPage)
        self.pdf_view.setZoomMode(QPdfView.ZoomMode.FitToWidth)
        self.pdf_view.setStyleSheet("""
            QPdfView {
                background-color: #1b2023;
                border: 1px solid #262b2e;
                border-radius: 4px;
            }
        """)
        right_layout.addWidget(self.pdf_view, 1)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        # -------------------------------------------------------------
        # Bottom Action Buttons
        # -------------------------------------------------------------
        bottom_bar = QHBoxLayout()
        bottom_bar.setSpacing(8)

        self.btn_open_external = QPushButton("👁 Open in System Viewer")
        self.btn_open_external.setCursor(Qt.PointingHandCursor)
        self.btn_open_external.setStyleSheet("""
            QPushButton {
                background-color: #171c1f;
                color: #dfe3e7;
                border: 1px solid #3b494b;
                border-radius: 4px;
                padding: 8px 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                border: 1px solid #00dbe9;
                color: #00dbe9;
            }
        """)
        self.btn_open_external.clicked.connect(self._on_open_external)
        bottom_bar.addWidget(self.btn_open_external)

        bottom_bar.addStretch()

        self.btn_save_pdf = QPushButton("💾 Save PDF Report...")
        self.btn_save_pdf.setCursor(Qt.PointingHandCursor)
        self.btn_save_pdf.setStyleSheet("""
            QPushButton {
                background-color: #00dbe9;
                color: #090e11;
                border: 1px solid #00dbe9;
                border-radius: 4px;
                padding: 8px 18px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #33e3ed;
            }
        """)
        self.btn_save_pdf.clicked.connect(self._on_save_pdf)
        bottom_bar.addWidget(self.btn_save_pdf)

        self.btn_close = QPushButton("Close")
        self.btn_close.setCursor(Qt.PointingHandCursor)
        self.btn_close.setStyleSheet("""
            QPushButton {
                background-color: #212629;
                color: #849495;
                border: 1px solid #3b494b;
                border-radius: 4px;
                padding: 8px 14px;
            }
            QPushButton:hover {
                color: #dfe3e7;
            }
        """)
        self.btn_close.clicked.connect(self.accept)
        bottom_bar.addWidget(self.btn_close)

        layout.addLayout(bottom_bar)

    def _sync_data_from_form(self) -> None:
        """Syncs GUI form values back to SurgicalReportData."""
        self.report_data.clinic_name = self.txt_clinic_name.text().strip()
        addr_parts = self.txt_clinic_addr.text().split("|")
        self.report_data.clinic_address = addr_parts[0].strip() if addr_parts else ""
        self.report_data.clinic_phone = addr_parts[1].strip() if len(addr_parts) > 1 else ""
        self.report_data.surgeon_name = self.txt_surgeon.text().strip()
        self.report_data.surgeon_title = self.txt_surgeon_title.text().strip()
        self.report_data.clinical_notes = self.txt_notes.toPlainText().strip()

    def _generate_and_render_preview(self) -> None:
        """Compiles ReportLab PDF to temporary file and loads in QPdfView."""
        self._sync_data_from_form()
        generator = SurgicalReportGenerator(self.report_data)
        generator.generate_pdf(self.temp_pdf_path)

        # Load into QPdfDocument
        self.pdf_document.load(self.temp_pdf_path)
        pages = self.pdf_document.pageCount()
        self.lbl_page_status.setText(f"Total Pages: {pages}")
        self.pdf_view.setZoomMode(QPdfView.ZoomMode.FitToWidth)

    def _on_refresh_clicked(self) -> None:
        self._generate_and_render_preview()

    def _zoom_in(self) -> None:
        self.pdf_view.setZoomMode(QPdfView.ZoomMode.Custom)
        self.pdf_view.setZoomFactor(self.pdf_view.zoomFactor() * 1.2)

    def _zoom_out(self) -> None:
        self.pdf_view.setZoomMode(QPdfView.ZoomMode.Custom)
        self.pdf_view.setZoomFactor(self.pdf_view.zoomFactor() / 1.2)

    def _fit_width(self) -> None:
        self.pdf_view.setZoomMode(QPdfView.ZoomMode.FitToWidth)

    def _on_save_pdf(self) -> None:
        """Prompts user to save final PDF to their chosen directory."""
        self._sync_data_from_form()
        clean_patient_id = self.report_data.patient_id.replace(" ", "_")
        default_filename = f"Surgical_Plan_{clean_patient_id}.pdf"
        default_path = os.path.join(os.path.expanduser("~"), default_filename)

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Surgical Planning PDF Report",
            default_path,
            "PDF Document (*.pdf)"
        )
        if filepath:
            generator = SurgicalReportGenerator(self.report_data)
            generator.generate_pdf(filepath)
            QMessageBox.information(
                self,
                "Report Saved",
                f"Surgical Planning PDF Report successfully saved to:\n\n{filepath}"
            )

    def _on_open_external(self) -> None:
        """Opens preview PDF in user's default OS viewer."""
        if os.path.exists(self.temp_pdf_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.temp_pdf_path))
