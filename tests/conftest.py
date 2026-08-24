"""
Pytest configuration and shared fixtures for Bau Medical Systems test suite.
"""

import pytest
from PySide6.QtCore import QCoreApplication


@pytest.fixture(scope="session")
def qapp():
    """Ensure a QCoreApplication exists for Qt signal/slot handling across test suites."""
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    return app
