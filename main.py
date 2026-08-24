"""
Bau Medical Systems — Dental CBCT 3D DICOM Desktop Viewer
Entry Point: main.py

Key Integration Features:
- Native QSurfaceFormat configuration BEFORE QApplication creation (eliminating OpenGL/VTK flickering).
- High DPI pixmap support and desktop scaling.
- Clean application startup and graceful shutdown.
"""

import os
import sys

# Ensure local modules are resolvable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtGui import QSurfaceFormat, QFont
from PySide6.QtWidgets import QApplication

# --------------------------------------------------------------------------
# Technical Trap 1: QSurfaceFormat Initialization for VTK/Qt6 OpenGL
# Must be configured and registered globally BEFORE QApplication is created.
# --------------------------------------------------------------------------
def configure_opengl_surface_format() -> None:
    """Configures global OpenGL surface format for robust VTK embedding."""
    surface_format = QSurfaceFormat()
    surface_format.setDepthBufferSize(24)
    surface_format.setStencilBufferSize(8)
    surface_format.setVersion(3, 2)
    surface_format.setProfile(QSurfaceFormat.CoreProfile)
    surface_format.setSwapBehavior(QSurfaceFormat.DoubleBuffer)
    surface_format.setSamples(4)  # 4x Multisample anti-aliasing (MSAA)
    QSurfaceFormat.setDefaultFormat(surface_format)


def main() -> None:
    """Application main entry point."""
    # 1. Configure global OpenGL format
    configure_opengl_surface_format()

    # 2. Configure Qt Core Attributes
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)

    # 3. Create Qt Application
    app = QApplication(sys.argv)
    app.setApplicationName("Bau Medical Systems Dental CBCT")
    app.setOrganizationName("Bau Medical Systems")
    app.setApplicationVersion("1.0.0")

    # Set Default Medical Typography (Inter fallback to Segoe UI / sans-serif)
    app_font = QFont("Inter", 10)
    app_font.setStyleStrategy(QFont.PreferAntialias)
    app.setFont(app_font)

    # Suppress benign Win32 OpenGL context teardown warnings
    import vtk
    vtk.vtkObject.GlobalWarningDisplayOff()

    # 4. Import & Launch Main Window
    from ui.main_window import MainWindow
    window = MainWindow()
    window.show()

    # Ensure clean resource release on application quit
    app.aboutToQuit.connect(window.viewport_grid.cleanup)

    # 5. Execute Event Loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
