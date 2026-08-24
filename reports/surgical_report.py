"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Module: reports/surgical_report.py

High-density, professional Vector Medical PDF Report Generator for Dental Implantology.
Features:
- ReportLab-based A4 Multi-page / Single-page surgical guide.
- Clinic Header Banner with brand insignia and demographics.
- Patient & Acquisition metadata grid.
- High-resolution unrolled panoramic arch visualization.
- Multi-site implant cross-section analysis grid with calibrated calipers and nerve clearance.
- Color-coded surgical schedule table with safety status badges (SAFE, CAUTION, BREACH).
- Clinical notes, diagnostic assessment, and doctor signature block.
- Numbered canvas with footer confidentiality notice.
"""

from __future__ import annotations
import os
import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
    HRFlowable,
)
from reportlab.pdfgen import canvas


@dataclass
class ImplantSiteRecord:
    """Diagnostic and planned dimensional parameters for an individual implant site."""
    implant_id: str
    tooth_number: int
    brand_preset: str = "Bau Dental Pro Universal"
    diameter_mm: float = 4.0
    length_mm: float = 11.5
    bl_angle_deg: float = 0.0
    md_angle_deg: float = 0.0
    min_nerve_dist_mm: float = float('inf')
    safety_state: str = "safe"              # 'safe' | 'warning' | 'breach'
    nearest_nerve: str = "Left Canal"
    bone_density_hu: float = 650.0          # Estimated mean HU at bone site
    ridge_height_mm: float = 14.2           # Caliper bone height
    crestal_width_mm: float = 7.5           # Caliper crestal width
    cross_section_img_path: Optional[str] = None


@dataclass
class SurgicalReportData:
    """Aggregated clinical case data for generating the surgical PDF report."""
    # Clinic & Surgeon
    clinic_name: str = "Bau Dental Implant & Oral Surgery Center"
    clinic_address: str = "1040 Medical Plaza, Suite 400"
    clinic_phone: str = "+1 (800) 555-BAUDENTAL"
    surgeon_name: str = "Dr. Sarah Jenkins, DDS, MS"
    surgeon_title: str = "Board Certified Oral & Maxillofacial Surgeon"
    generation_date: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

    # Patient & Scan Info
    patient_name: str = "Anonymous Patient"
    patient_id: str = "P-09412"
    patient_dob_sex: str = "1982-04-15 / Male"
    study_date: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d"))
    modality: str = "Dental CBCT"
    scanner_model: str = "Bau CBCT Pro 3D (FoV 12×10 cm)"
    voxel_spacing_mm: str = "0.25 mm Isotropic"

    # Clinical Visuals & Plans
    panoramic_img_path: Optional[str] = None
    implant_sites: List[ImplantSiteRecord] = field(default_factory=list)
    clinical_notes: str = (
        "Pre-operative 3D CBCT evaluation indicates adequate bone volume at site #19. "
        "Virtual implant placed with >2.0mm safety margin to the left inferior alveolar nerve canal. "
        "Standard surgical guide recommended for guided placement."
    )
    logo_path: Optional[str] = None


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and stamp total page count in the footer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Footer divider line
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(15 * mm, 12 * mm, 195 * mm, 12 * mm)

        # Footer text
        confidential_text = "CONFIDENTIAL MEDICAL RECORD — FOR CLINICAL USE ONLY — BAU MEDICAL SYSTEMS"
        self.drawString(15 * mm, 8 * mm, confidential_text)

        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(195 * mm, 8 * mm, page_str)
        self.restoreState()


class SurgicalReportGenerator:
    """
    Builds clean, vector-formatted clinical PDF reports using ReportLab.
    """

    def __init__(self, data: SurgicalReportData) -> None:
        self.data = data
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self) -> None:
        """Configures typography hierarchy for the medical report."""
        # Primary Brand Colors
        self.c_primary = colors.HexColor("#0F172A")       # Dark Slate
        self.c_accent = colors.HexColor("#0891B2")        # Medical Cyan/Teal
        self.c_text = colors.HexColor("#1E293B")          # Slate 800
        self.c_muted = colors.HexColor("#64748B")         # Slate 500
        self.c_border = colors.HexColor("#E2E8F0")        # Light Border
        self.c_bg_subtle = colors.HexColor("#F8FAFC")     # Soft background

        self.styles.add(ParagraphStyle(
            name="ClinicHeaderTitle",
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=self.c_primary,
        ))
        self.styles.add(ParagraphStyle(
            name="ClinicHeaderSub",
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=self.c_muted,
        ))
        self.styles.add(ParagraphStyle(
            name="ReportMainTitle",
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=self.c_accent,
            alignment=2, # Right align
        ))
        self.styles.add(ParagraphStyle(
            name="SectionHeading",
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=self.c_primary,
            spaceAfter=4,
        ))
        self.styles.add(ParagraphStyle(
            name="MetaKey",
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=self.c_muted,
        ))
        self.styles.add(ParagraphStyle(
            name="MetaVal",
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=self.c_text,
        ))
        self.styles.add(ParagraphStyle(
            name="TableHead",
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.white,
            alignment=1, # Center
        ))
        self.styles.add(ParagraphStyle(
            name="TableCell",
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=self.c_text,
            alignment=1,
        ))
        self.styles.add(ParagraphStyle(
            name="TableCellBold",
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=self.c_primary,
            alignment=1,
        ))
        self.styles.add(ParagraphStyle(
            name="NotesBody",
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=self.c_text,
        ))

    def generate_pdf(self, output_path: str) -> str:
        """Compiles the report and writes the vector PDF to output_path."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=15 * mm,
            rightMargin=15 * mm,
            topMargin=15 * mm,
            bottomMargin=18 * mm,
        )

        story = []

        # 1. Header Banner & Branding
        story.append(self._build_header_banner())
        story.append(HRFlowable(width="100%", thickness=1.5, color=self.c_accent, spaceBefore=4, spaceAfter=8))

        # 2. Patient & Acquisition Demographics Grid
        story.append(self._build_demographics_card())
        story.append(Spacer(1, 8))

        # 3. Panoramic Surgical Overview (if image available)
        if self.data.panoramic_img_path and os.path.exists(self.data.panoramic_img_path):
            story.append(self._build_panoramic_section())
            story.append(Spacer(1, 8))

        # 4. Implant Planning Schedule Table
        story.append(self._build_schedule_table())
        story.append(Spacer(1, 10))

        # 5. Multi-Site Cross-Section Analysis Grid
        if self.data.implant_sites:
            sites_with_images = [s for s in self.data.implant_sites if s.cross_section_img_path and os.path.exists(s.cross_section_img_path)]
            if sites_with_images:
                story.append(self._build_cross_section_grid(sites_with_images))
                story.append(Spacer(1, 8))

        # 6. Clinical Assessment & Doctor Signature Block
        story.append(self._build_signoff_block())

        # Build with dynamic page counting
        doc.build(story, canvasmaker=NumberedCanvas)
        return output_path

    def _build_header_banner(self) -> Table:
        """Constructs clinic logo, clinic demographics, and report title banner."""
        left_flowables = [
            Paragraph(self.data.clinic_name, self.styles["ClinicHeaderTitle"]),
            Paragraph(f"{self.data.clinic_address} &bull; {self.data.clinic_phone}", self.styles["ClinicHeaderSub"]),
            Paragraph(f"Surgeon: <b>{self.data.surgeon_name}</b> ({self.data.surgeon_title})", self.styles["ClinicHeaderSub"]),
        ]

        right_flowables = [
            Paragraph("DENTAL IMPLANT PLANNING REPORT", self.styles["ReportMainTitle"]),
            Paragraph(f"Generated: {self.data.generation_date}", self.styles["ClinicHeaderSub"]),
            Paragraph(f"System: Bau Medical Systems CBCT 3D", self.styles["ClinicHeaderSub"]),
        ]

        data = [[left_flowables, right_flowables]]
        t = Table(data, colWidths=[105 * mm, 75 * mm])
        t.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        return t

    def _build_demographics_card(self) -> Table:
        """Constructs two-column patient and acquisition parameters card."""
        p_name = Paragraph(f"<b>{self.data.patient_name}</b>", self.styles["MetaVal"])
        p_id = Paragraph(self.data.patient_id, self.styles["MetaVal"])
        p_dob = Paragraph(self.data.patient_dob_sex, self.styles["MetaVal"])

        p_mod = Paragraph(f"<b>{self.data.modality}</b>", self.styles["MetaVal"])
        p_date = Paragraph(self.data.study_date, self.styles["MetaVal"])
        p_vox = Paragraph(f"{self.data.voxel_spacing_mm} &bull; {self.data.scanner_model}", self.styles["MetaVal"])

        data = [
            [
                Paragraph("PATIENT INFORMATION", self.styles["SectionHeading"]),
                "",
                Paragraph("ACQUISITION / SCAN PARAMETERS", self.styles["SectionHeading"]),
                ""
            ],
            [
                Paragraph("Patient Name:", self.styles["MetaKey"]), p_name,
                Paragraph("Modality:", self.styles["MetaKey"]), p_mod
            ],
            [
                Paragraph("Patient ID / MRN:", self.styles["MetaKey"]), p_id,
                Paragraph("Scan Date:", self.styles["MetaKey"]), p_date
            ],
            [
                Paragraph("DOB / Gender:", self.styles["MetaKey"]), p_dob,
                Paragraph("Resolution / FoV:", self.styles["MetaKey"]), p_vox
            ],
        ]

        t = Table(data, colWidths=[28 * mm, 60 * mm, 32 * mm, 60 * mm])
        t.setStyle(TableStyle([
            ('SPAN', (0, 0), (1, 0)),
            ('SPAN', (2, 0), (3, 0)),
            ('BACKGROUND', (0, 0), (-1, -1), self.c_bg_subtle),
            ('BOX', (0, 0), (-1, -1), 0.5, self.c_border),
            ('INNERGRID', (0, 1), (-1, -1), 0.3, colors.HexColor("#EDF2F7")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ]))
        return t

    def _build_panoramic_section(self) -> KeepTogether:
        """Constructs unrolled panoramic radiograph visual card."""
        elements = [
            Paragraph("PANORAMIC SURGICAL ARCH OVERVIEW (Curved MPR)", self.styles["SectionHeading"]),
            Image(self.data.panoramic_img_path, width=180 * mm, height=52 * mm),
        ]
        return KeepTogether(elements)

    def _build_schedule_table(self) -> KeepTogether:
        """Builds formatted surgical implant schedule table with color-coded safety badges."""
        elements = [
            Paragraph("PLANNED IMPLANT SPECIFICATIONS & NERVE CLEARANCE SCHEDULE", self.styles["SectionHeading"])
        ]

        headers = [
            Paragraph("Site", self.styles["TableHead"]),
            Paragraph("Preset / Brand", self.styles["TableHead"]),
            Paragraph("Dim (Ø × L)", self.styles["TableHead"]),
            Paragraph("Angulation", self.styles["TableHead"]),
            Paragraph("Bone Density", self.styles["TableHead"]),
            Paragraph("Nerve Clearance", self.styles["TableHead"]),
            Paragraph("Safety Status", self.styles["TableHead"]),
        ]

        rows = [headers]

        if not self.data.implant_sites:
            # Placeholder row if no implants are active
            empty_row = [
                Paragraph("--", self.styles["TableCell"]),
                Paragraph("No implants planned in current case", self.styles["TableCell"]),
                Paragraph("--", self.styles["TableCell"]),
                Paragraph("--", self.styles["TableCell"]),
                Paragraph("--", self.styles["TableCell"]),
                Paragraph("--", self.styles["TableCell"]),
                Paragraph("CLEAR", self.styles["TableCellBold"]),
            ]
            rows.append(empty_row)
        else:
            for site in self.data.implant_sites:
                # Status Badge Styling
                st = site.safety_state.lower()
                if st == "safe":
                    badge = Paragraph("<font color='#00875A'><b>SAFE</b> (≥2.0mm)</font>", self.styles["TableCell"])
                elif st == "warning":
                    badge = Paragraph("<font color='#B76E00'><b>CAUTION</b> (1.5-2.0mm)</font>", self.styles["TableCell"])
                else:
                    badge = Paragraph("<font color='#DE350B'><b>CRITICAL BREACH</b></font>", self.styles["TableCell"])

                dist_str = f"{site.min_nerve_dist_mm:.2f} mm" if site.min_nerve_dist_mm != float('inf') else "N/A"

                row = [
                    Paragraph(f"<b>Tooth #{site.tooth_number}</b>", self.styles["TableCellBold"]),
                    Paragraph(site.brand_preset, self.styles["TableCell"]),
                    Paragraph(f"Ø {site.diameter_mm:.1f} × {site.length_mm:.1f} mm", self.styles["TableCellBold"]),
                    Paragraph(f"BL: {site.bl_angle_deg:+.0f}° / MD: {site.md_angle_deg:+.0f}°", self.styles["TableCell"]),
                    Paragraph(f"{site.bone_density_hu:.0f} HU (Type II)", self.styles["TableCell"]),
                    Paragraph(f"<b>{dist_str}</b>", self.styles["TableCell"]),
                    badge,
                ]
                rows.append(row)

        col_widths = [20 * mm, 42 * mm, 28 * mm, 26 * mm, 22 * mm, 22 * mm, 20 * mm]
        t = Table(rows, colWidths=col_widths)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.c_primary),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('BOX', (0, 0), (-1, -1), 0.5, self.c_border),
            ('INNERGRID', (0, 0), (-1, -1), 0.3, self.c_border),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.c_bg_subtle]),
        ]))
        elements.append(t)
        return KeepTogether(elements)

    def _build_cross_section_grid(self, sites: List[ImplantSiteRecord]) -> KeepTogether:
        """Builds multi-site cross-section diagnostic image cards."""
        elements = [
            Paragraph("IMPLANT SITE TRANSVERSE CROSS-SECTION ANALYSIS", self.styles["SectionHeading"])
        ]

        card_rows = []
        for s in sites[:4]: # Cap at 4 sites for clean layout
            img_elem = Image(s.cross_section_img_path, width=42 * mm, height=42 * mm)
            caption_text = (
                f"<b>Tooth #{s.tooth_number}</b> (Ø {s.diameter_mm:.1f}×{s.length_mm:.1f}mm)<br/>"
                f"Ridge H: {s.ridge_height_mm:.1f} mm &bull; Crest W: {s.crestal_width_mm:.1f} mm<br/>"
                f"Nerve Distance: <b>{s.min_nerve_dist_mm:.2f} mm</b>"
            )
            cap_elem = Paragraph(caption_text, self.styles["TableCell"])
            card_rows.append([img_elem, cap_elem])

        t = Table(card_rows, colWidths=[45 * mm, 135 * mm])
        t.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 0.5, self.c_border),
            ('INNERGRID', (0, 0), (-1, -1), 0.3, self.c_border),
            ('BACKGROUND', (0, 0), (-1, -1), self.c_bg_subtle),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        elements.append(t)
        return KeepTogether(elements)

    def _build_signoff_block(self) -> KeepTogether:
        """Constructs clinical observations text and doctor signature verification block."""
        notes_p = Paragraph(f"<b>Clinical Notes & Surgical Assessment:</b><br/>{self.data.clinical_notes}", self.styles["NotesBody"])

        sign_p = Paragraph(
            f"<b>Surgeon Sign-Off:</b><br/><br/>"
            f"_________________________________________<br/>"
            f"<b>{self.data.surgeon_name}</b><br/>"
            f"<font color='#64748B'>{self.data.surgeon_title}</font><br/>"
            f"Date: {self.data.generation_date.split()[0]}",
            self.styles["NotesBody"]
        )

        data = [[notes_p, sign_p]]
        t = Table(data, colWidths=[115 * mm, 65 * mm])
        t.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOX', (0, 0), (-1, -1), 0.5, self.c_border),
            ('BACKGROUND', (0, 0), (-1, -1), self.c_bg_subtle),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        return KeepTogether([t])
