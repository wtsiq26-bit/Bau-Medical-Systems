"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Module: core/dicom_loader.py

Non-blocking Multi-Format Volume Loader Worker Thread (QThread).
Supported Formats:
1. Standard DICOM series (SimpleITK GDCM reader).
2. Non-standard DICOM files (pydicom multi-slice parser).
3. Image Sequences (.png, .jpg, .jpeg, .tif, .tiff, .bmp) with automatic HU normalization.
4. Built-in Realistic Dental CBCT Synthetic Generator.
"""

from __future__ import annotations
import os
import re
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from PIL import Image
from PySide6.QtCore import QThread, Signal

from core.volume_data import VolumeData, DicomMetadata


def natural_sort_key(s: str) -> List[Any]:
    """Sort strings containing numbers in natural human order (e.g. slice_1, slice_2, ..., slice_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


class DicomLoaderWorker(QThread):
    """
    Asynchronous Multi-Format Loader Worker Thread.
    Loads DICOM directories or 2D image slice sequences without blocking the Qt event loop.
    """
    progress = Signal(int, str)       # progress percentage (0-100), status message
    finished = Signal(object)         # VolumeData instance
    error = Signal(str)               # error message

    SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

    def __init__(self, directory_path: Optional[str] = None, is_synthetic: bool = False) -> None:
        super().__init__()
        self.directory_path = directory_path
        self.is_synthetic = is_synthetic

    def run(self) -> None:
        """Thread execution entry point."""
        try:
            if self.is_synthetic or not self.directory_path:
                self.progress.emit(10, "Generating Synthetic Dental CBCT Volume...")
                volume = self._generate_synthetic_dental_cbct()
                self.progress.emit(100, "Synthetic Volume Ready.")
                self.finished.emit(volume)
                return

            self.progress.emit(10, f"Scanning directory: {os.path.basename(self.directory_path)}...")

            # Tier 1: Try SimpleITK DICOM series reader
            volume = self._load_with_simpleitk(self.directory_path)

            # Tier 2: Fallback to pydicom series parser
            if volume is None:
                self.progress.emit(30, "Scanning with pydicom series parser...")
                volume = self._load_with_pydicom(self.directory_path)

            # Tier 3: Fallback to 2D Image Sequence (.png, .jpg, .tif, .bmp)
            if volume is None:
                self.progress.emit(40, "No DICOM series found. Checking for 2D image sequence (.png/.tif/.jpg)...")
                volume = self._load_from_image_sequence(self.directory_path)

            if volume is None:
                raise ValueError(
                    "No valid DICOM series or image sequence (.png, .jpg, .tif, .bmp) found in the selected folder."
                )

            self.progress.emit(100, "Volume successfully reconstructed.")
            self.finished.emit(volume)

        except Exception as e:
            self.error.emit(f"Failed to load dataset: {str(e)}")

    def _load_with_simpleitk(self, folder_path: str) -> Optional[VolumeData]:
        """Loads a DICOM series using SimpleITK with GDCM."""
        try:
            import SimpleITK as sitk
        except ImportError:
            return None

        try:
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(folder_path)
            if not dicom_names:
                return None

            total_files = len(dicom_names)
            self.progress.emit(25, f"Found {total_files} DICOM slices. Reading series...")

            reader.SetFileNames(dicom_names)
            sitk_image = reader.Execute()

            self.progress.emit(70, "Converting image buffer to calibrated Hounsfield Units...")
            np_array = sitk.GetArrayFromImage(sitk_image)  # Shape: (Nz, Ny, Nx)
            spacing = sitk_image.GetSpacing()              # (dx, dy, dz)
            origin = sitk_image.GetOrigin()                # (ox, oy, oz)
            direction = sitk_image.GetDirection()          # 9 elements

            metadata = self._extract_metadata_from_file(dicom_names[0])
            metadata.fov_dimensions_mm = (
                np_array.shape[2] * spacing[0],
                np_array.shape[1] * spacing[1],
                np_array.shape[0] * spacing[2]
            )

            self.progress.emit(90, "Building zero-copy VTK scalar volume...")
            return VolumeData(
                array=np_array,
                spacing=spacing,
                origin=origin,
                direction=direction,
                metadata=metadata
            )
        except Exception:
            return None

    def _load_with_pydicom(self, folder_path: str) -> Optional[VolumeData]:
        """Loads a DICOM series using pydicom and sorts slices by spatial position."""
        try:
            import pydicom
        except ImportError:
            return None

        all_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
        ]

        slices = []
        for i, filepath in enumerate(all_files):
            try:
                ds = pydicom.dcmread(filepath, stop_before_pixels=False, force=False)
                if hasattr(ds, 'pixel_array') and (hasattr(ds, 'ImagePositionPatient') or hasattr(ds, 'InstanceNumber')):
                    slices.append(ds)
            except Exception:
                continue

        if not slices:
            return None

        # Sort slices by ImagePositionPatient Z (or InstanceNumber as fallback)
        try:
            slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
        except Exception:
            slices.sort(key=lambda s: int(getattr(s, 'InstanceNumber', 0)))

        self.progress.emit(65, f"Assembling 3D matrix from {len(slices)} DICOM slices...")

        first = slices[0]
        pixel_spacing = getattr(first, 'PixelSpacing', [0.4, 0.4])
        dx = float(pixel_spacing[1])
        dy = float(pixel_spacing[0])

        if len(slices) > 1:
            try:
                dz = abs(float(slices[1].ImagePositionPatient[2]) - float(slices[0].ImagePositionPatient[2]))
            except Exception:
                dz = float(getattr(first, 'SliceThickness', dx))
        else:
            dz = float(getattr(first, 'SliceThickness', dx))

        nz = len(slices)
        ny, nx = first.pixel_array.shape
        volume_array = np.zeros((nz, ny, nx), dtype=np.int16)

        for k, s in enumerate(slices):
            slope = float(getattr(s, 'RescaleSlope', 1.0))
            intercept = float(getattr(s, 'RescaleIntercept', 0.0))
            raw = s.pixel_array.astype(np.float32)
            hu_slice = raw * slope + intercept
            volume_array[k, :, :] = np.clip(hu_slice, -1024, 3071).astype(np.int16)

        origin = tuple(float(x) for x in getattr(first, 'ImagePositionPatient', [0.0, 0.0, 0.0]))
        orientation = getattr(first, 'ImageOrientationPatient', [1, 0, 0, 0, 1, 0])
        x_vec = np.array([float(orientation[0]), float(orientation[1]), float(orientation[2])])
        y_vec = np.array([float(orientation[3]), float(orientation[4]), float(orientation[5])])
        z_vec = np.cross(x_vec, y_vec)
        direction = (
            x_vec[0], x_vec[1], x_vec[2],
            y_vec[0], y_vec[1], y_vec[2],
            z_vec[0], z_vec[1], z_vec[2]
        )

        metadata = self._extract_metadata_from_pydicom(first)
        metadata.fov_dimensions_mm = (nx * dx, ny * dy, nz * dz)

        return VolumeData(
            array=volume_array,
            spacing=(dx, dy, dz),
            origin=origin,
            direction=direction,
            metadata=metadata
        )

    def _load_from_image_sequence(self, folder_path: str) -> Optional[VolumeData]:
        """
        Loads an ordered sequence of 2D image slices (.png, .jpg, .tif, .bmp).
        Normalizes pixel intensities to calibrated Dental Hounsfield Units (HU):
        - Background / Black (0) -> -1000 HU (Air)
        - Soft Tissue (~30-60) -> +40 HU
        - Bone (~100-180) -> +800 to +1500 HU
        - Enamel / White (255) -> +2500 HU
        """
        # Discover all valid image files
        valid_files = [
            f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(self.SUPPORTED_IMAGE_EXTENSIONS)
        ]

        if not valid_files:
            return None

        # Naturally sort files by filename number
        valid_files.sort(key=natural_sort_key)
        total_slices = len(valid_files)
        self.progress.emit(45, f"Found {total_slices} image slices. Reading images...")

        # Read first slice to determine dimensions and bit-depth
        first_img_path = os.path.join(folder_path, valid_files[0])
        with Image.open(first_img_path) as first_img:
            # Convert to grayscale if RGB
            if first_img.mode != 'L' and first_img.mode != 'I;16' and first_img.mode != 'I':
                first_img_gray = first_img.convert('L')
            else:
                first_img_gray = first_img

            first_arr = np.array(first_img_gray)
            ny, nx = first_arr.shape
            is_16bit = (first_arr.dtype == np.uint16 or first_arr.dtype == np.int16 or first_arr.dtype == np.int32)

        volume_array = np.zeros((total_slices, ny, nx), dtype=np.int16)

        for k, filename in enumerate(valid_files):
            img_path = os.path.join(folder_path, filename)
            with Image.open(img_path) as img:
                if img.mode != 'L' and not is_16bit:
                    img_gray = img.convert('L')
                else:
                    img_gray = img
                slice_arr = np.array(img_gray, dtype=np.float32)

                # Intensity to Hounsfield Unit (HU) Calibration
                if is_16bit:
                    # If maximum value > 4000, scale 16-bit to HU range [-1000, +3000]
                    max_val = np.max(slice_arr)
                    if max_val > 4095:
                        hu_slice = (slice_arr / 65535.0) * 4000.0 - 1000.0
                    else:
                        # Standard 12-bit CT data in 16-bit container
                        hu_slice = slice_arr - 1024.0
                else:
                    # Standard 8-bit image (0..255) -> [-1000 HU, +2500 HU]
                    # 0 -> -1000 HU (Air)
                    # 73 -> +0 HU (Water/Soft tissue)
                    # 146 -> +1000 HU (Cortical bone)
                    # 255 -> +2500 HU (Enamel)
                    hu_slice = (slice_arr / 255.0) * 3500.0 - 1000.0

                volume_array[k, :, :] = np.clip(hu_slice, -1024, 3071).astype(np.int16)

            if k % 15 == 0 or k == total_slices - 1:
                pct = int(50 + (k / total_slices) * 40)
                self.progress.emit(pct, f"Processing slice {k + 1}/{total_slices}...")

        # Standard isotropic dental CBCT voxel resolution (0.4mm)
        dx, dy, dz = 0.4, 0.4, 0.4
        folder_name = os.path.basename(os.path.abspath(folder_path))

        metadata = DicomMetadata(
            patient_name=f"DATASET_{folder_name.upper()}",
            patient_id=f"SEQ-{total_slices:03d}",
            study_description="2D Slice Sequence Ingestion",
            series_description=f"Image Series ({total_slices} slices, 0.4mm)",
            study_date="20260824",
            modality="CT",
            manufacturer="Bau Medical Systems",
            window_center=500.0,
            window_width=2500.0,
            fov_dimensions_mm=(nx * dx, ny * dy, total_slices * dz)
        )

        origin = (-(nx * dx) * 0.5, -(ny * dy) * 0.5, -(total_slices * dz) * 0.5)

        self.progress.emit(95, "Constructing memory-anchored VolumeData...")
        return VolumeData(
            array=volume_array,
            spacing=(dx, dy, dz),
            origin=origin,
            metadata=metadata
        )

    def _extract_metadata_from_file(self, filepath: str) -> DicomMetadata:
        """Helper to extract metadata from a single DICOM file."""
        try:
            import pydicom
            ds = pydicom.dcmread(filepath, stop_before_pixels=True)
            return self._extract_metadata_from_pydicom(ds)
        except Exception:
            return DicomMetadata()

    def _extract_metadata_from_pydicom(self, ds: Any) -> DicomMetadata:
        """Extracts standard DICOM clinical header parameters."""
        meta = DicomMetadata()
        meta.patient_name = str(getattr(ds, 'PatientName', 'Anonymous'))
        meta.patient_id = str(getattr(ds, 'PatientID', 'N/A'))
        meta.patient_birth_date = str(getattr(ds, 'PatientBirthDate', 'N/A'))
        meta.patient_sex = str(getattr(ds, 'PatientSex', 'O'))
        meta.study_description = str(getattr(ds, 'StudyDescription', 'Dental CBCT Exam'))
        meta.series_description = str(getattr(ds, 'SeriesDescription', 'Dental Mandible Scan'))
        meta.study_date = str(getattr(ds, 'StudyDate', 'N/A'))
        meta.modality = str(getattr(ds, 'Modality', 'CT'))
        meta.manufacturer = str(getattr(ds, 'Manufacturer', 'Bau Medical Systems'))
        meta.rescale_slope = float(getattr(ds, 'RescaleSlope', 1.0))
        meta.rescale_intercept = float(getattr(ds, 'RescaleIntercept', 0.0))

        wc = getattr(ds, 'WindowCenter', 500.0)
        ww = getattr(ds, 'WindowWidth', 2500.0)
        meta.window_center = float(wc[0] if isinstance(wc, (list, tuple)) else wc)
        meta.window_width = float(ww[0] if isinstance(ww, (list, tuple)) else ww)

        return meta

    def _generate_synthetic_dental_cbct(self) -> VolumeData:
        """
        Synthesizes a realistic 3D Dental CBCT Mandibular dataset in Hounsfield Units (HU).
        """
        self.progress.emit(20, "Generating anatomical coordinate grid...")

        nx, ny, nz = 160, 160, 140
        dx, dy, dz = 0.5, 0.5, 0.5

        x = (np.arange(nx) - nx / 2.0) * dx
        y = (np.arange(ny) - ny / 2.0) * dy
        z = (np.arange(nz) - nz / 2.0) * dz

        X, Y, Z = np.meshgrid(x, y, z, indexing='xy')
        X = np.transpose(X, (2, 1, 0))
        Y = np.transpose(Y, (2, 1, 0))
        Z = np.transpose(Z, (2, 1, 0))

        volume = np.full((nz, ny, nx), -1000.0, dtype=np.float32)

        self.progress.emit(40, "Synthesizing facial soft tissue and airway...")
        soft_tissue_mask = ((X / 38.0) ** 2 + ((Y + 5.0) / 36.0) ** 2 + (Z / 35.0) ** 2) <= 1.0
        volume[soft_tissue_mask] = np.random.normal(35.0, 15.0, size=np.sum(soft_tissue_mask))

        airway_mask = ((X / 10.0) ** 2 + ((Y - 22.0) / 8.0) ** 2) <= 1.0
        volume[airway_mask] = -1000.0

        self.progress.emit(60, "Modeling Mandibular Jaw Arch and Cortical Bone...")
        arch_y = 0.022 * (X ** 2) - 18.0
        dist_to_arch = np.sqrt((Y - arch_y) ** 2 + (np.clip(Z, -25.0, 10.0) - Z) ** 2)

        mandible_mask = (dist_to_arch <= 7.5) & (Z >= -25.0) & (Z <= 10.0) & (np.abs(X) <= 32.0)
        cortical_outer = (dist_to_arch >= 5.0) & mandible_mask
        trabecular_inner = (dist_to_arch < 5.0) & mandible_mask

        volume[trabecular_inner] = np.random.normal(550.0, 60.0, size=np.sum(trabecular_inner))
        volume[cortical_outer] = np.random.normal(1250.0, 90.0, size=np.sum(cortical_outer))

        nerve_canal_mask = (
            (((X - 14.0) ** 2 + (Y - (0.022 * (14.0 ** 2) - 18.0)) ** 2) <= 1.8) |
            (((X + 14.0) ** 2 + (Y - (0.022 * (14.0 ** 2) - 18.0)) ** 2) <= 1.8)
        ) & (Z >= -20.0) & (Z <= -5.0)
        volume[nerve_canal_mask] = -50.0

        self.progress.emit(80, "Positioning Teeth, Enamel Crowns, and Root Canals...")
        tooth_x_positions = np.linspace(-26.0, 26.0, 12)
        for tx in tooth_x_positions:
            ty = 0.022 * (tx ** 2) - 18.0
            tz_crown = 14.0
            tz_root = 4.0

            root_mask = (
                ((X - tx) ** 2 + (Y - ty) ** 2 <= 2.2 ** 2) &
                (Z >= tz_root - 8.0) & (Z <= tz_crown)
            )
            volume[root_mask] = np.random.normal(1350.0, 50.0, size=np.sum(root_mask))

            pulp_mask = (
                ((X - tx) ** 2 + (Y - ty) ** 2 <= 0.6 ** 2) &
                (Z >= tz_root - 7.0) & (Z <= tz_crown - 2.0)
            )
            volume[pulp_mask] = 40.0

            crown_mask = (
                ((X - tx) ** 2 + (Y - ty) ** 2 <= 3.2 ** 2) &
                (Z >= tz_crown - 4.0) & (Z <= tz_crown + 4.0)
            )
            volume[crown_mask] = np.random.normal(2600.0, 100.0, size=np.sum(crown_mask))

        volume_int16 = np.clip(volume, -1024, 3071).astype(np.int16)

        metadata = DicomMetadata(
            patient_name="DEMO^DENTAL_CBCT",
            patient_id="BAU-CBCT-001",
            study_description="Mandibular Arch Cone Beam CT",
            series_description="Synthetic High-Res CBCT (0.5mm)",
            study_date="20260824",
            modality="CT",
            manufacturer="Bau Medical Systems",
            window_center=500.0,
            window_width=2500.0,
            fov_dimensions_mm=(nx * dx, ny * dy, nz * dz)
        )

        return VolumeData(
            array=volume_int16,
            spacing=(dx, dy, dz),
            origin=(-(nx * dx) / 2.0, -(ny * dy) / 2.0, -(nz * dz) / 2.0),
            metadata=metadata
        )
