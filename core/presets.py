"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Module: core/presets.py

Clinical presets for 2D Window/Level and 3D Volume Rendering transfer functions.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class WindowLevelPreset:
    """Represents a 2D Window Width (WW) and Window Level (WL) display preset."""
    name: str
    window_width: float
    window_level: float
    description: str


@dataclass(frozen=True)
class Volume3DPreset:
    """Represents a 3D Volume Rendering Transfer Function preset."""
    id: str
    name: str
    description: str


# 2D Dental CBCT Window/Level Presets (Hounsfield Units)
WL_PRESETS: Dict[str, WindowLevelPreset] = {
    "bone": WindowLevelPreset(
        name="Dental Bone",
        window_width=2500.0,
        window_level=500.0,
        description="Standard mandibular and maxillary cortical bone contrast"
    ),
    "teeth": WindowLevelPreset(
        name="Teeth & Enamel",
        window_width=3500.0,
        window_level=1200.0,
        description="High-contrast visualization of tooth enamel, dentin, and pulp"
    ),
    "soft_tissue": WindowLevelPreset(
        name="Soft Tissue & Airway",
        window_width=400.0,
        window_level=40.0,
        description="Airway passages, tongue, and pharyngeal soft tissue structures"
    ),
    "panoramic": WindowLevelPreset(
        name="Mandible & Arch",
        window_width=2000.0,
        window_level=600.0,
        description="Panoramic-style contrast optimized for dental arch tracing"
    ),
    "implant": WindowLevelPreset(
        name="Implant / Metal Artifact",
        window_width=4500.0,
        window_level=1500.0,
        description="Suppresses metal streak artifacts for titanium implant assessment"
    ),
}

# 3D Dental CBCT Transfer Function Presets
VOLUME_3D_PRESETS: List[Volume3DPreset] = [
    Volume3DPreset(
        id="bone_hard_tissue",
        name="Hard Tissue 3D (Bone & Teeth)",
        description="Ivory bone with high-density enamel highlights"
    ),
    Volume3DPreset(
        id="teeth_enamel",
        name="Enamel & Dentin Detail 3D",
        description="Isolates dental crowns and root morphology"
    ),
    Volume3DPreset(
        id="composite_soft_bone",
        name="Composite 3D (Bone + Soft Tissue)",
        description="Semi-transparent facial profile over hard tissue architecture"
    ),
]
