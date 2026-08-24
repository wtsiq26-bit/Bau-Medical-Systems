"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Module: ui/styles.py

Qt Style Sheet (QSS) adhering strictly to the Bau Medical Systems Design System (DESIGN (1).md).
Palette:
- Base: #0f1417 (Clinical Dark Room)
- Surface: #171c1f / #1b2023
- Surface Highlight: #262b2e / #313539
- Primary Accent: #00dbe9 / #00f0ff (Electric Cyan)
- Secondary Accent: #b9c7e4 / #3c4962
- Text: #dfe3e7 (Sterile White), #b9cacb (Variant Gray)
- Outline: #849495, #3b494b
"""

BAU_DARK_THEME = """
/* Global Window Styling */
QMainWindow, QDialog, QWidget#root_container {
    background-color: #0f1417;
    color: #dfe3e7;
    font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
    font-size: 13px;
}

/* ToolBar & Header */
QToolBar {
    background-color: #171c1f;
    border-bottom: 1px solid #262b2e;
    padding: 4px 8px;
    spacing: 6px;
}

QToolButton {
    background-color: transparent;
    color: #dfe3e7;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 5px 10px;
    font-weight: 500;
    font-size: 12px;
}

QToolButton:hover {
    background-color: #262b2e;
    border: 1px solid #3b494b;
    color: #00dbe9;
}

QToolButton:pressed, QToolButton:checked {
    background-color: #1b2023;
    border: 1px solid #00dbe9;
    color: #00f0ff;
}

/* Push Buttons */
QPushButton {
    background-color: #1b2023;
    color: #dfe3e7;
    border: 1px solid #3b494b;
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: 500;
    font-size: 12px;
}

QPushButton:hover {
    background-color: #262b2e;
    border: 1px solid #00dbe9;
    color: #00dbe9;
}

QPushButton:pressed {
    background-color: #0a0f12;
    border: 1px solid #00f0ff;
    color: #ffffff;
}

/* Primary Action Button (Electric Cyan) */
QPushButton#btn_primary {
    background-color: #00dbe9;
    color: #002022;
    border: 1px solid #00f0ff;
    font-weight: 600;
}

QPushButton#btn_primary:hover {
    background-color: #00f0ff;
    color: #000000;
}

QPushButton#btn_primary:pressed {
    background-color: #00b4c0;
    color: #002022;
}

/* Preset Buttons */
QPushButton.preset-btn {
    background-color: #171c1f;
    color: #b9cacb;
    border: 1px solid #262b2e;
    border-radius: 3px;
    padding: 5px 8px;
    font-size: 11px;
}

QPushButton.preset-btn:hover {
    background-color: #262b2e;
    border-color: #00dbe9;
    color: #00dbe9;
}

QPushButton.preset-btn:checked {
    background-color: #262b2e;
    border-color: #00dbe9;
    color: #00dbe9;
    font-weight: bold;
}

/* Sliders */
QSlider::groove:horizontal {
    height: 4px;
    background: #262b2e;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: #00dbe9;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #dfe3e7;
    border: 2px solid #00dbe9;
    width: 14px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background: #00f0ff;
    border-color: #ffffff;
}

/* SpinBoxes & LineEdits */
QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #0a0f12;
    color: #00dbe9;
    border: 1px solid #3b494b;
    border-radius: 3px;
    padding: 4px 6px;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 12px;
}

QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
    border: 1px solid #00dbe9;
    background-color: #0f1417;
}

/* ComboBoxes */
QComboBox {
    background-color: #171c1f;
    color: #dfe3e7;
    border: 1px solid #3b494b;
    border-radius: 4px;
    padding: 5px 10px;
    font-size: 12px;
}

QComboBox:hover {
    border: 1px solid #00dbe9;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid #262b2e;
}

QComboBox QAbstractItemView {
    background-color: #171c1f;
    color: #dfe3e7;
    selection-background-color: #262b2e;
    selection-color: #00dbe9;
    border: 1px solid #3b494b;
}

/* TabWidget */
QTabWidget::pane {
    border: 1px solid #262b2e;
    background-color: #171c1f;
    border-radius: 4px;
}

QTabBar::tab {
    background-color: #0f1417;
    color: #b9cacb;
    border: 1px solid #262b2e;
    padding: 6px 14px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    font-size: 11px;
    font-weight: 500;
}

QTabBar::tab:selected {
    background-color: #171c1f;
    color: #00dbe9;
    border-bottom: 2px solid #00dbe9;
}

QTabBar::tab:hover:!selected {
    background-color: #1b2023;
    color: #dfe3e7;
}

/* Scroll Area & Scrollbars */
QScrollArea {
    border: none;
    background-color: transparent;
}

QScrollBar:vertical {
    background-color: #0f1417;
    width: 8px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background-color: #313539;
    min-height: 20px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background-color: #00dbe9;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

/* Status Bar */
QStatusBar {
    background-color: #0a0f12;
    border-top: 1px solid #1b2023;
    color: #b9cacb;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 11px;
}

QStatusBar::item {
    border: none;
}

/* Tooltip */
QToolTip {
    background-color: #1b2023;
    color: #00dbe9;
    border: 1px solid #00dbe9;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 11px;
}

/* Progress Bar */
QProgressBar {
    background-color: #0a0f12;
    border: 1px solid #262b2e;
    border-radius: 4px;
    text-align: center;
    color: #dfe3e7;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
}

QProgressBar::chunk {
    background-color: #00dbe9;
    border-radius: 3px;
}
"""
