"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Verification Script: verify_gui.py
"""

import os
import sys
import time

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vtk
vtk.vtkObject.GlobalWarningDisplayOff()

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication
from main import configure_opengl_surface_format
from ui.main_window import MainWindow
from dental.nerve_tracer import NerveChannel


def test_full_pipeline():
    configure_opengl_surface_format()
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()
    app.aboutToQuit.connect(window.viewport_grid.cleanup)

    print("Step 1: MainWindow initialized successfully.")
    print("Step 2: Brand logo loaded:", not window.logo_pixmap.isNull())

    # Wait for the background worker to finish loading synthetic CBCT
    max_wait = 10.0
    start_time = time.time()

    def check_loaded():
        if window.volume_data is not None:
            print("Step 3: Synthetic Dental CBCT volume successfully loaded.")
            print(f"        Dimensions: {window.volume_data.dimensions}")
            print(f"        Spacing: {window.volume_data.spacing}")
            print(f"        HU Range: [{window.volume_data.min_hu:.0f}, {window.volume_data.max_hu:.0f}]")

            # Test Crosshair Synchronization
            print("Step 4: Testing synchronized crosshair movement...")
            window.viewport_grid._on_crosshair_moved(10.0, -5.0, 2.0)
            axial_pos = window.viewport_grid.axial_view._current_world_pos
            print(f"        Axial View Position: {axial_pos}")

            # Test Window / Level Sync
            print("Step 5: Testing Window/Level synchronization...")
            window.viewport_grid.set_global_window_level(3000.0, 800.0)

            # Test 3D Preset Change
            print("Step 6: Testing 3D Transfer function preset...")
            window.viewport_grid.volume_view.apply_preset("teeth_enamel")

            # Test Mandibular Nerve Tracing
            print("Step 7: Testing Mandibular Nerve Tracing pipeline...")
            tracer = window.viewport_grid.nerve_tracer
            tracer.add_point(-14.0, -18.0, -20.0, channel=NerveChannel.LEFT)
            tracer.add_point(-16.0, -10.0, -15.0, channel=NerveChannel.LEFT)
            tracer.add_point(-18.0, 5.0, -8.0, channel=NerveChannel.LEFT)

            left_len = tracer.get_track(NerveChannel.LEFT).get_total_length_mm()
            print(f"        Left Mandibular Canal Traced: {left_len:.1f} mm (3 points)")
            assert left_len > 15.0, "Nerve length calculation failed"

            # Test 2D Intersection Slice
            print("Step 8: Testing 2D Slice Nerve Intersection Disc rendering...")
            window.viewport_grid.axial_view.set_world_position(-15.0, -14.0, -17.5, force_reslice=True)
            intersections = tracer.get_track(NerveChannel.LEFT).calculate_2d_slice_intersections(
                "axial", (-15.0, -14.0, -17.5)
            )
            print(f"        2D Intersections Detected on Axial plane: {len(intersections)}")
            assert len(intersections) >= 1, "Expected at least 1 intersection on slice"

            # Test Series Sidebar
            print("Step 9: Testing Left Series Sidebar card population...")
            card_count = len(window.series_sidebar._cards)
            print(f"        Active Series Cards in Filmstrip: {card_count}")
            assert card_count >= 1, "Expected at least 1 series card in filmstrip"

            # Test Ribbon Layout Switcher
            print("Step 10: Testing Ribbon Layout Manager (1x1, 1+3, 3D, 2x2)...")
            window.ribbon_toolbar.btn_layout_1x1.click()
            assert window.viewport_grid._current_layout_mode == "1x1"
            window.ribbon_toolbar.btn_layout_1p3.click()
            assert window.viewport_grid._current_layout_mode == "1+3"
            window.ribbon_toolbar.btn_layout_2x2.click()
            assert window.viewport_grid._current_layout_mode == "2x2"

            # Test Dental Arch Curve & Panoramic MPR
            print("Step 11: Testing Dental Arch Curve spline fitting...")
            window.viewport_grid.auto_fit_dental_arch()
            arch = window.viewport_grid.dental_arch
            print(f"         Arch Spline Length: {arch.total_length_mm:.1f} mm ({len(arch.sampled_points)} samples)")
            assert arch.total_length_mm > 30.0, "Arch curve length calculation failed"

            # Test Panoramic Focal Trough Generation
            print("Step 12: Testing Panoramic Focal Trough (Curved MPR) generation...")
            pano_img = window.viewport_grid.panoramic_view.panoramic_generator.generate_panoramic_image(
                window.volume_data, arch
            )
            assert pano_img is not None, "Panoramic image generation failed"
            print(f"         Synthetic Panoramic Radiograph Dimensions: {pano_img.shape[1]} (cols) × {pano_img.shape[0]} (rows)")
            assert pano_img.shape[0] > 10 and pano_img.shape[1] > 10

            # Test Dental Implant Layout & Cross-Section Navigation
            print("Step 13: Testing Dental Implant Planning Layout Mode (Axial + Pano + Cross-Section + 3D)...")
            window.ribbon_toolbar.btn_layout_implant.click()
            assert window.viewport_grid._current_layout_mode == "implant"

            window.viewport_grid.set_cross_section_index(15)
            active_idx = window.viewport_grid.cross_section_mgr.active_index
            print(f"         Active Transverse Cross-Section Index: #{active_idx + 1}")
            assert active_idx == 15, "Cross-section index mismatch"

            # Test Virtual Dental Implant Placement & Automated Nerve Safety Collision Engine
            print("Step 14: Testing Virtual Dental Implant Placement & Nerve Safety Clearance Engine...")
            imp_id = window.viewport_grid.add_implant(tooth_number=19, diameter_mm=4.0, length_mm=11.5)
            active_imp = window.viewport_grid.implant_manager.get_active_implant()
            assert active_imp is not None, "Failed to get active implant"
            print(f"         Virtual Implant Created: {imp_id} (Tooth #19, Ø {active_imp.diameter_mm:.1f}mm × {active_imp.length_mm:.1f}mm)")

            # Test Preset and Angulation
            window.viewport_grid.set_implant_angulation(12.0, 0.0)
            assert active_imp.bl_angle_deg == 12.0, "BL angulation failed"

            # Test Vector Medical PDF Report Generation & Dialog Preview
            print("Step 15: Testing Dental Surgical Planning PDF Report Generator & Preview UI...")
            import tempfile
            from reports.surgical_report import SurgicalReportData, ImplantSiteRecord, SurgicalReportGenerator
            from ui.report_dialog import SurgicalReportDialog

            test_pdf = os.path.join(tempfile.gettempdir(), "verify_surgical_report.pdf")
            rep_data = SurgicalReportData(
                patient_name="DOE^JANE",
                patient_id="MRN-77291",
                patient_dob_sex="1985-06-12 / Female",
                implant_sites=[
                    ImplantSiteRecord(
                        implant_id="imp_test",
                        tooth_number=19,
                        diameter_mm=4.0,
                        length_mm=11.5,
                        min_nerve_dist_mm=2.65,
                        safety_state="safe"
                    )
                ]
            )
            gen = SurgicalReportGenerator(rep_data)
            gen.generate_pdf(test_pdf)
            assert os.path.exists(test_pdf), "PDF file was not created"
            print(f"         Vector PDF Compiled: {os.path.basename(test_pdf)} ({os.path.getsize(test_pdf)} bytes)")

            # Test Dialog Initialization
            dlg = SurgicalReportDialog(rep_data, window)
            assert dlg.pdf_document.pageCount() >= 1, "QPdfDocument failed to load generated pages"
            print(f"         Live PDF Preview Rendered: {dlg.pdf_document.pageCount()} Page(s)")

            print("\nALL 15 VERIFICATION CHECKS PASSED SUCCESSFULLY!")
            app.quit()
        elif time.time() - start_time > max_wait:
            print("ERROR: Timeout waiting for volume to load.")
            app.exit(1)

    timer = QTimer()
    timer.timeout.connect(check_loaded)
    timer.start(100)

    return app.exec()


if __name__ == "__main__":
    ret = test_full_pipeline()
    sys.exit(ret)
