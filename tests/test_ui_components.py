"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Unit Tests: tests/test_ui_components.py

Validates the Commercial PACS UI Workstation components:
- SeriesSidebar (Series card creation, selection, collapse/expand).
- RibbonToolbar (Signal emission for tools, layouts, cine, measurements).
- ViewportGrid (Active viewport highlighting, 2x2, 1x1, 1+3, 3D layout switching).
"""

import sys
import unittest
import numpy as np
from PySide6.QtWidgets import QApplication

from main import configure_opengl_surface_format
from core.volume_data import VolumeData, DicomMetadata
from ui.series_sidebar import SeriesSidebar, SeriesItemCard
from ui.ribbon_toolbar import RibbonToolbar
from ui.viewport_grid import ViewportGrid

# Ensure single global QApplication for testing
app = QApplication.instance()
if app is None:
    configure_opengl_surface_format()
    app = QApplication(sys.argv)


class TestPACSUIComponents(unittest.TestCase):
    """Test suite for commercial PACS UI components."""

    def setUp(self):
        # Create a mock volume data
        nz, ny, nx = 10, 20, 20
        arr = np.zeros((nz, ny, nx), dtype=np.int16)
        self.mock_volume = VolumeData(
            array=arr,
            spacing=(0.5, 0.5, 0.5),
            origin=(0.0, 0.0, 0.0),
            metadata=DicomMetadata(
                patient_name="DOE^JOHN",
                patient_id="PACS-001",
                series_description="Mandibular Scan"
            )
        )

    def test_series_sidebar(self):
        sidebar = SeriesSidebar()
        self.assertEqual(len(sidebar._cards), 0)

        sidebar.set_volume_data(self.mock_volume, series_id="series_1")
        self.assertEqual(len(sidebar._cards), 1)
        self.assertIn("series_1", sidebar._cards)
        self.assertEqual(sidebar._active_series_id, "series_1")

        # Test collapse toggle
        self.assertFalse(sidebar._is_collapsed)
        sidebar.toggle_collapse()
        self.assertTrue(sidebar._is_collapsed)
        sidebar.toggle_collapse()
        self.assertFalse(sidebar._is_collapsed)

    def test_ribbon_toolbar_signals(self):
        ribbon = RibbonToolbar()

        received_tools = []
        ribbon.signals.tool_changed.connect(lambda t: received_tools.append(t))

        ribbon.btn_pan.click()
        self.assertIn("pan", received_tools)

        ribbon.btn_zoom.click()
        self.assertIn("zoom", received_tools)

        received_layouts = []
        ribbon.signals.layout_changed.connect(lambda l: received_layouts.append(l))

        ribbon.btn_layout_1x1.click()
        self.assertIn("1x1", received_layouts)

        ribbon.btn_layout_1p3.click()
        self.assertIn("1+3", received_layouts)

    def test_viewport_grid_layouts_and_focus(self):
        grid = ViewportGrid()
        grid.set_volume_data(self.mock_volume)

        # Test Active Viewport Focus
        self.assertEqual(grid.active_viewport_id, "axial")
        grid.set_active_viewport("coronal")
        self.assertEqual(grid.active_viewport_id, "coronal")
        self.assertTrue(grid.coronal_container._is_active)
        self.assertFalse(grid.axial_container._is_active)

        # Test Layout Switching
        grid.set_layout_mode("1x1")
        self.assertEqual(grid._current_layout_mode, "1x1")

        grid.set_layout_mode("1+3")
        self.assertEqual(grid._current_layout_mode, "1+3")

        grid.set_layout_mode("3d")
        self.assertEqual(grid._current_layout_mode, "3d")

        grid.set_layout_mode("2x2")
        self.assertEqual(grid._current_layout_mode, "2x2")


if __name__ == "__main__":
    unittest.main()
