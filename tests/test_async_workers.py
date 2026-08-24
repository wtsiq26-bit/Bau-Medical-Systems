"""
Unit tests for Asynchronous Background Worker Threads.
Module: tests/test_async_workers.py
"""

import os
import tempfile
import numpy as np
import pytest
import vtk
import SimpleITK as sitk
from PySide6.QtCore import QCoreApplication

from core.volume_data import VolumeData
from core.async_workers import SegmentationWorker, ICPRegistrationWorker, PanoramicWorker
from dental.surface_extractor import STRUCTURE_PRESETS, TEETH_ENAMEL, MANDIBLE_BONE
from dental.panoramic_mpr import DentalArchCurve


@pytest.fixture(scope="session")
def qapp():
    """Ensure a QCoreApplication exists for signal/slot handling."""
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    return app


def test_segmentation_worker_execution(qapp):
    """Tests that SegmentationWorker extracts labeled structures asynchronously."""
    # Create temporary NIfTI segmentation mask
    mask_np = np.zeros((30, 30, 30), dtype=np.int32)
    # Add mandible label (1)
    mask_np[5:25, 5:25, 5:25] = 1
    # Add teeth label (3)
    mask_np[10:20, 10:20, 10:20] = 3

    sitk_img = sitk.GetImageFromArray(mask_np)
    sitk_img.SetSpacing((1.0, 1.0, 1.0))
    sitk_img.SetOrigin((0.0, 0.0, 0.0))

    temp_nii = os.path.join(tempfile.gettempdir(), "test_seg_worker.nii.gz")
    sitk.WriteImage(sitk_img, temp_nii)

    try:
        worker = SegmentationWorker(file_path=temp_nii)

        progress_events = []
        streamed_structures = {}
        finished_results = {}
        errors = []

        worker.progress_updated.connect(lambda pct, msg: progress_events.append((pct, msg)))
        worker.structure_extracted.connect(lambda sid, pd: streamed_structures.update({sid: pd}))
        worker.finished_all.connect(lambda res: finished_results.update(res))
        worker.error_occurred.connect(lambda err: errors.append(err))

        worker.start()
        assert worker.wait(10000), "SegmentationWorker timed out"
        # Process queued cross-thread signals in Qt event loop
        QCoreApplication.processEvents()

        assert len(errors) == 0, f"Worker encountered errors: {errors}"
        assert len(progress_events) > 0
        assert "mandible" in finished_results
        assert "teeth" in finished_results
        assert finished_results["mandible"].GetNumberOfPoints() > 0
        assert finished_results["teeth"].GetNumberOfPoints() > 0
        assert "mandible" in streamed_structures
        assert "teeth" in streamed_structures
    finally:
        if os.path.exists(temp_nii):
            os.remove(temp_nii)


def test_icp_registration_worker_execution(qapp):
    """Tests that ICPRegistrationWorker performs rigid alignment asynchronously."""
    # Create target sphere mesh
    target_sphere = vtk.vtkSphereSource()
    target_sphere.SetRadius(15.0)
    target_sphere.SetCenter(0.0, 0.0, 0.0)
    target_sphere.SetPhiResolution(20)
    target_sphere.SetThetaResolution(20)
    target_sphere.Update()
    target_poly = target_sphere.GetOutput()

    # Create source sphere displaced by 5mm translation
    source_sphere = vtk.vtkSphereSource()
    source_sphere.SetRadius(15.0)
    source_sphere.SetCenter(5.0, 2.0, -3.0)
    source_sphere.SetPhiResolution(20)
    source_sphere.SetThetaResolution(20)
    source_sphere.Update()
    source_poly = source_sphere.GetOutput()

    worker = ICPRegistrationWorker(
        source_poly=source_poly,
        target_poly=target_poly,
        max_iterations=100,
    )

    progress_events = []
    reg_results = []
    errors = []

    worker.progress_updated.connect(lambda pct, msg: progress_events.append((pct, msg)))
    worker.registration_complete.connect(
        lambda poly, mat, rms, iters, max_95, status: reg_results.append((poly, mat, rms, iters, max_95, status))
    )
    worker.error_occurred.connect(lambda err: errors.append(err))

    worker.start()
    assert worker.wait(10000), "ICPRegistrationWorker timed out"
    # Process queued cross-thread signals in Qt event loop
    QCoreApplication.processEvents()

    assert len(errors) == 0, f"Worker error: {errors}"
    assert len(reg_results) == 1
    aligned_poly, transform_mat, rms, num_iters, max_95, status = reg_results[0]

    assert aligned_poly.GetNumberOfPoints() == source_poly.GetNumberOfPoints()
    assert transform_mat.shape == (4, 4)
    assert rms < 1.0  # Spheres should converge with minimal residual
    assert status in ("EXCELLENT", "ACCEPTABLE")


def test_panoramic_worker_execution(qapp):
    """Tests that PanoramicWorker generates curved MPR image asynchronously."""
    # Generate synthetic volume
    from core.dicom_loader import DicomLoaderWorker
    loader = DicomLoaderWorker(is_synthetic=True)
    volume = loader._generate_synthetic_dental_cbct()

    # Create dental arch
    arch = DentalArchCurve(step_size_mm=1.0)
    arch.auto_fit_parabola(volume)

    worker = PanoramicWorker(
        volume=volume,
        arch_curve=arch,
        focal_trough_thickness_mm=5.0,
    )

    progress_events = []
    panoramic_images = []
    errors = []

    worker.progress_updated.connect(lambda pct, msg: progress_events.append((pct, msg)))
    worker.panoramic_ready.connect(lambda img: panoramic_images.append(img))
    worker.error_occurred.connect(lambda err: errors.append(err))

    worker.start()
    assert worker.wait(15000), "PanoramicWorker timed out"
    # Process queued cross-thread signals in Qt event loop
    QCoreApplication.processEvents()

    assert len(errors) == 0, f"Worker error: {errors}"
    assert len(panoramic_images) == 1
    pano_arr = panoramic_images[0]
    assert isinstance(pano_arr, np.ndarray)
    assert pano_arr.ndim == 2
    assert pano_arr.shape[0] > 10 and pano_arr.shape[1] > 10
