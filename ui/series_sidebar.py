"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Module: ui/series_sidebar.py

Left-Hand Commercial PACS Series Filmstrip Sidebar.
Features:
- Collapsible vertical dark filmstrip docking widget.
- Multi-series thumbnail preview generator from 3D volumes / DICOM sequences.
- Interactive Series Cards with high-contrast middle-slice thumbnail, modality badge,
  slice counter, spatial resolution, and active focus outline.
- Instant series switching on single/double click.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any, Callable
import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, Signal, Qt, QSize
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QSizePolicy
)
from PySide6.QtGui import QPixmap, QImage, QColor, QPainter, QBrush, QPen

from core.volume_data import VolumeData


class SeriesItemCard(QFrame):
    """
    Individual Interactive PACS Series Thumbnail Card.
    """
    clicked = Signal(str)  # series_id

    def __init__(
        self,
        series_id: str,
        series_name: str,
        modality: str,
        slice_count: int,
        spacing_text: str,
        thumbnail_array: Optional[np.ndarray] = None,
        parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.series_id = series_id
        self._is_active = False

        self.setObjectName("series_item_card")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(120)

        self._apply_style()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # 1. Thumbnail Preview Box
        self.lbl_thumbnail = QLabel()
        self.lbl_thumbnail.setFixedSize(88, 88)
        self.lbl_thumbnail.setAlignment(Qt.AlignCenter)
        self.lbl_thumbnail.setStyleSheet("""
            QLabel {
                background-color: #080c0e;
                border: 1px solid #262b2e;
                border-radius: 3px;
            }
        """)
        self._generate_thumbnail_pixmap(thumbnail_array)
        layout.addWidget(self.lbl_thumbnail)

        # 2. Metadata Information Block
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(0, 2, 0, 2)
        info_layout.setSpacing(3)

        # Modality & Slice Count Badge Row
        badge_row = QHBoxLayout()
        badge_row.setSpacing(4)

        self.lbl_modality = QLabel(modality.upper())
        self.lbl_modality.setStyleSheet("""
            QLabel {
                background-color: #00dbe9;
                color: #090e11;
                font-family: Inter;
                font-size: 9px;
                font-weight: 800;
                border-radius: 2px;
                padding: 1px 4px;
            }
        """)
        badge_row.addWidget(self.lbl_modality)

        self.lbl_slices = QLabel(f"{slice_count} Slices")
        self.lbl_slices.setStyleSheet("color: #849495; font-size: 10px; font-family: 'JetBrains Mono';")
        badge_row.addWidget(self.lbl_slices)
        badge_row.addStretch()
        info_layout.addLayout(badge_row)

        # Series Description
        self.lbl_name = QLabel(series_name)
        self.lbl_name.setWordWrap(True)
        self.lbl_name.setStyleSheet("color: #dfe3e7; font-family: Inter; font-size: 11px; font-weight: 600;")
        info_layout.addWidget(self.lbl_name)

        # Voxel Resolution
        self.lbl_res = QLabel(spacing_text)
        self.lbl_res.setStyleSheet("color: #849495; font-family: 'JetBrains Mono'; font-size: 10px;")
        info_layout.addWidget(self.lbl_res)

        info_layout.addStretch()
        layout.addLayout(info_layout, 1)

    def _apply_style(self) -> None:
        if self._is_active:
            self.setStyleSheet("""
                QFrame#series_item_card {
                    background-color: #1b2023;
                    border: 2px solid #00dbe9;
                    border-radius: 4px;
                    margin-bottom: 4px;
                }
            """)
        else:
            self.setStyleSheet("""
                QFrame#series_item_card {
                    background-color: #13171a;
                    border: 1px solid #262b2e;
                    border-radius: 4px;
                    margin-bottom: 4px;
                }
                QFrame#series_item_card:hover {
                    background-color: #171c1f;
                    border: 1px solid #3b494b;
                }
            """)

    def set_active(self, active: bool) -> None:
        self._is_active = active
        self._apply_style()

    def _generate_thumbnail_pixmap(self, arr: Optional[np.ndarray]) -> None:
        if arr is None:
            self.lbl_thumbnail.setText("NO PREV")
            self.lbl_thumbnail.setStyleSheet("color: #4b5563; font-size: 9px;")
            return

        # Normalize 2D array to 8-bit grayscale
        try:
            arr_float = arr.astype(np.float32)
            min_v, max_v = np.percentile(arr_float, 2), np.percentile(arr_float, 98)
            if max_v - min_v > 0:
                norm = np.clip((arr_float - min_v) / (max_v - min_v) * 255.0, 0, 255).astype(np.uint8)
            else:
                norm = np.zeros_like(arr_float, dtype=np.uint8)

            h, w = norm.shape
            bytes_per_line = w
            qimg = QImage(norm.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
            pix = QPixmap.fromImage(qimg).scaled(84, 84, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_thumbnail.setPixmap(pix)
        except Exception:
            self.lbl_thumbnail.setText("IMG")

    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)
        self.clicked.emit(self.series_id)


class SeriesSidebar(QWidget):
    """
    Collapsible PACS Series List Filmstrip Sidebar on Left Workspace.
    """
    series_selected = Signal(str)  # series_id

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("series_sidebar")
        self.setFixedWidth(240)
        self._is_collapsed = False
        self._cards: Dict[str, SeriesItemCard] = {}
        self._active_series_id: Optional[str] = None

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header Bar with Collapse Button
        header = QFrame(self)
        header.setFixedHeight(32)
        header.setStyleSheet("""
            QFrame {
                background-color: #171c1f;
                border-bottom: 1px solid #262b2e;
            }
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 0, 6, 0)

        lbl_title = QLabel("SERIES EXPLORER")
        lbl_title.setStyleSheet("color: #00dbe9; font-family: Inter; font-size: 11px; font-weight: 700; letter-spacing: 0.06em;")
        header_layout.addWidget(lbl_title)

        header_layout.addStretch()

        self.btn_toggle = QPushButton("◀")
        self.btn_toggle.setFixedSize(22, 22)
        self.btn_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_toggle.setToolTip("Toggle Series Sidebar (Collapse/Expand)")
        self.btn_toggle.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #849495;
                border: none;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                color: #00dbe9;
                background-color: #262b2e;
                border-radius: 2px;
            }
        """)
        self.btn_toggle.clicked.connect(self.toggle_collapse)
        header_layout.addWidget(self.btn_toggle)

        main_layout.addWidget(header)

        # Scroll Area for Series Cards
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: #0f1417;
                border: none;
            }
        """)
        main_layout.addWidget(self.scroll_area)

        self.cards_container = QWidget()
        self.scroll_area.setWidget(self.cards_container)

        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(6, 6, 6, 6)
        self.cards_layout.setSpacing(6)
        self.cards_layout.addStretch()

    def add_series_item(
        self,
        series_id: str,
        series_name: str,
        modality: str,
        slice_count: int,
        spacing_text: str,
        thumbnail_array: Optional[np.ndarray] = None,
        select: bool = False
    ) -> None:
        """Adds a series card to the filmstrip."""
        if series_id in self._cards:
            return

        card = SeriesItemCard(
            series_id=series_id,
            series_name=series_name,
            modality=modality,
            slice_count=slice_count,
            spacing_text=spacing_text,
            thumbnail_array=thumbnail_array,
            parent=self.cards_container
        )
        card.clicked.connect(self._on_card_clicked)

        # Insert before stretch
        self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)
        self._cards[series_id] = card

        if select or len(self._cards) == 1:
            self.set_active_series(series_id)

    def set_volume_data(self, volume: VolumeData, series_id: str = "primary_volume") -> None:
        """Automatically builds series thumbnail from loaded VolumeData."""
        self.clear_series()

        meta = volume.metadata
        mid_z = volume.nz // 2
        mid_slice = volume.array[mid_z, :, :]

        spacing_str = f"Voxel: {volume.spacing[0]:.2f} mm"

        self.add_series_item(
            series_id=series_id,
            series_name=meta.series_description or "Dental CBCT Volume",
            modality=meta.modality or "CT",
            slice_count=volume.nz,
            spacing_text=spacing_str,
            thumbnail_array=mid_slice,
            select=True
        )

    def set_active_series(self, series_id: str) -> None:
        """Highlights the active series card."""
        self._active_series_id = series_id
        for sid, card in self._cards.items():
            card.set_active(sid == series_id)

    def _on_card_clicked(self, series_id: str) -> None:
        self.set_active_series(series_id)
        self.series_selected.emit(series_id)

    def clear_series(self) -> None:
        """Removes all series cards."""
        for card in self._cards.values():
            card.deleteLater()
        self._cards.clear()
        self._active_series_id = None

    def toggle_collapse(self) -> None:
        """Toggles sidebar between collapsed (32px) and expanded (240px)."""
        if self._is_collapsed:
            self.setFixedWidth(240)
            self.scroll_area.show()
            self.btn_toggle.setText("◀")
            self._is_collapsed = False
        else:
            self.setFixedWidth(36)
            self.scroll_area.hide()
            self.btn_toggle.setText("▶")
            self._is_collapsed = True
