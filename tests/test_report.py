"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Unit Tests: test_report.py
"""

import os
import tempfile
import unittest
from PIL import Image as PILImage
from reports.surgical_report import (
    SurgicalReportData,
    ImplantSiteRecord,
    SurgicalReportGenerator
)


class TestSurgicalReport(unittest.TestCase):
    """Tests for Vector Medical PDF Report Generation with ReportLab."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_pdf = os.path.join(self.temp_dir, "test_surgical_plan.pdf")

        # Create dummy panoramic snapshot image
        self.dummy_pano = os.path.join(self.temp_dir, "dummy_pano.png")
        img = PILImage.new("RGB", (600, 200), color=(20, 30, 40))
        img.save(self.dummy_pano)

        # Create dummy cross section snapshot image
        self.dummy_cs = os.path.join(self.temp_dir, "dummy_cs.png")
        img_cs = PILImage.new("RGB", (200, 200), color=(10, 20, 30))
        img_cs.save(self.dummy_cs)

    def test_pdf_report_generation(self):
        data = SurgicalReportData(
            patient_name="DOE^JOHN",
            patient_id="MRN-88491",
            patient_dob_sex="1978-11-20 / Male",
            study_date="2026-08-24",
            panoramic_img_path=self.dummy_pano,
            implant_sites=[
                ImplantSiteRecord(
                    implant_id="imp_1",
                    tooth_number=19,
                    brand_preset="Bau Universal Ø4.0×11.5mm",
                    diameter_mm=4.0,
                    length_mm=11.5,
                    bl_angle_deg=10.0,
                    md_angle_deg=0.0,
                    min_nerve_dist_mm=3.25,
                    safety_state="safe",
                    nearest_nerve="Left Mandibular Canal",
                    bone_density_hu=720.0,
                    cross_section_img_path=self.dummy_cs
                ),
                ImplantSiteRecord(
                    implant_id="imp_2",
                    tooth_number=30,
                    brand_preset="Bau Universal Ø4.5×10.0mm",
                    diameter_mm=4.5,
                    length_mm=10.0,
                    bl_angle_deg=-5.0,
                    md_angle_deg=2.0,
                    min_nerve_dist_mm=1.85,
                    safety_state="warning",
                    nearest_nerve="Right Mandibular Canal",
                    bone_density_hu=640.0,
                    cross_section_img_path=self.dummy_cs
                ),
            ]
        )

        gen = SurgicalReportGenerator(data)
        result_path = gen.generate_pdf(self.output_pdf)

        self.assertTrue(os.path.exists(result_path))
        file_size = os.path.getsize(result_path)
        self.assertGreater(file_size, 5000, "PDF file is too small or corrupt")

    def test_empty_implant_sites_pdf(self):
        data = SurgicalReportData(
            patient_name="EMPTY^CASE",
            patient_id="MRN-00000",
            implant_sites=[]
        )
        gen = SurgicalReportGenerator(data)
        result_path = gen.generate_pdf(self.output_pdf)
        self.assertTrue(os.path.exists(result_path))


if __name__ == "__main__":
    unittest.main()
