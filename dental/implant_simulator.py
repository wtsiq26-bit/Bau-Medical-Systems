"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Module: dental/implant_simulator.py

Interactive 3D/2D Virtual Dental Implant Placement & Automated Nerve Safety Collision Engine.
Features:
- Parametric Root-Form Conical & Cylindrical Implant Body and Platform Geometry.
- Concentric 2.0mm Virtual Safety Sleeve (vtkPolyData wireframe/translucent envelope).
- Full 6-DOF Kinematic Transformation Matrix $T_{implant} \\in \\mathrm{SE}(3)$ with Bucco-Lingual (BL) and Mesio-Distal (MD) angulations.
- Real-time Euclidean Distance & Collision Query Engine against Mandibular Nerve Spline.
- Dynamic Clinical Safety States:
  * Safe (>= 2.0 mm): Emerald Green (#00FF7F) / Cyan (#00dbe9)
  * Warning (1.5 mm - 2.0 mm): Vibrant Amber (#FFD700)
  * Critical Breach (< 1.5 mm): Crimson Red Alert (#FF0033)
- Multi-planar 2D slice projection for Cross-Section, Panoramic, and Axial viewports.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
import math
import numpy as np
import vtk

from PySide6.QtCore import QObject, Signal
from dental.nerve_tracer import NerveTracer, NerveChannel


class ImplantSafetyState(Enum):
    """Clinical safety clearance rating relative to anatomical risk structures."""
    SAFE = "safe"           # >= 2.0 mm
    WARNING = "warning"     # 1.5 mm - 2.0 mm
    BREACH = "breach"       # < 1.5 mm (Collision risk)


@dataclass
class ImplantPreset:
    """Standard commercial clinical dental implant dimensions."""
    name: str
    diameter_mm: float
    length_mm: float
    brand: str = "Bau Dental Pro"


STANDARD_IMPLANT_PRESETS: List[ImplantPreset] = [
    ImplantPreset("Ø 3.0 × 10.0 mm (Narrow Anterior)", 3.0, 10.0),
    ImplantPreset("Ø 3.5 × 10.0 mm (Standard Anterior)", 3.5, 10.0),
    ImplantPreset("Ø 3.5 × 11.5 mm (Standard Anterior)", 3.5, 11.5),
    ImplantPreset("Ø 4.0 × 10.0 mm (Premolar Universal)", 4.0, 10.0),
    ImplantPreset("Ø 4.0 × 11.5 mm (Premolar Universal)", 4.0, 11.5),
    ImplantPreset("Ø 4.5 × 10.0 mm (Standard Molar)", 4.5, 10.0),
    ImplantPreset("Ø 4.5 × 11.5 mm (Standard Molar)", 4.5, 11.5),
    ImplantPreset("Ø 4.5 × 13.0 mm (Standard Molar)", 4.5, 13.0),
    ImplantPreset("Ø 5.0 × 10.0 mm (Wide Molar)", 5.0, 10.0),
    ImplantPreset("Ø 5.0 × 11.5 mm (Wide Molar)", 5.0, 11.5),
    ImplantPreset("Ø 6.0 × 10.0 mm (Ultra-Wide)", 6.0, 10.0),
]


class ImplantSignals(QObject):
    """Qt Signals emitted on implant transformation or clearance changes."""
    implant_added = Signal(str)                                # implant_id
    implant_removed = Signal(str)                              # implant_id
    implant_selected = Signal(str)                             # implant_id
    implant_modified = Signal(str)                             # implant_id
    safety_status_changed = Signal(str, float, str, str)       # implant_id, min_dist_mm, state_str, nearest_nerve


class DentalImplant:
    """
    Parametric 3D Dental Implant with rigid kinematics and 2.0mm concentric safety sleeve.
    """

    def __init__(
        self,
        implant_id: str,
        tooth_number: int = 19,
        diameter_mm: float = 4.0,
        length_mm: float = 11.5,
        safety_margin_mm: float = 2.0,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        bl_angle_deg: float = 0.0,
        md_angle_deg: float = 0.0,
        rotation_z_deg: float = 0.0
    ) -> None:
        self.implant_id = implant_id
        self.tooth_number = tooth_number
        self.diameter_mm = float(diameter_mm)
        self.length_mm = float(length_mm)
        self.safety_margin_mm = float(safety_margin_mm)

        # Kinematic Pose in World coordinates
        self.position = list(position)  # [x, y, z] platform center
        self.bl_angle_deg = float(bl_angle_deg)
        self.md_angle_deg = float(md_angle_deg)
        self.rotation_z_deg = float(rotation_z_deg)

        self.safety_state = ImplantSafetyState.SAFE
        self.min_nerve_dist_mm: float = float('inf')
        self.nearest_nerve_name: str = "none"
        self.show_safety_sleeve: bool = True

        # VTK 3D Visualization Pipeline
        self._build_3d_geometry()
        self.update_transform()

    def _build_3d_geometry(self) -> None:
        """Constructs parametric 3D implant polydata (collar + tapered body + apex dome)."""
        r_platform = self.diameter_mm * 0.5
        r_apex = r_platform * 0.7
        l_total = self.length_mm
        collar_h = min(1.5, l_total * 0.15)
        body_h = l_total - collar_h

        # 1. Platform Collar (Cylinder)
        collar_src = vtk.vtkCylinderSource()
        collar_src.SetRadius(r_platform)
        collar_src.SetHeight(collar_h)
        collar_src.SetResolution(32)
        collar_src.CappingOn()

        # Shift collar center to -collar_h/2 (Z down)
        tf_collar = vtk.vtkTransform()
        tf_collar.RotateX(90)
        tf_collar.Translate(0.0, -collar_h * 0.5, 0.0)

        tf_filter_collar = vtk.vtkTransformPolyDataFilter()
        tf_filter_collar.SetInputConnection(collar_src.GetOutputPort())
        tf_filter_collar.SetTransform(tf_collar)
        tf_filter_collar.Update()

        # 2. Tapered Conical Body
        cone_src = vtk.vtkConeSource()
        cone_src.SetRadius(r_platform)
        cone_src.SetHeight(body_h)
        cone_src.SetResolution(32)
        cone_src.CappingOn()

        tf_cone = vtk.vtkTransform()
        tf_cone.RotateY(90)
        tf_cone.Translate(-collar_h - body_h * 0.5, 0.0, 0.0)

        tf_filter_cone = vtk.vtkTransformPolyDataFilter()
        tf_filter_cone.SetInputConnection(cone_src.GetOutputPort())
        tf_filter_cone.SetTransform(tf_cone)
        tf_filter_cone.Update()

        # 3. Append into single composite implant mesh
        append_filter = vtk.vtkAppendPolyData()
        append_filter.AddInputData(tf_filter_collar.GetOutput())
        append_filter.AddInputData(tf_filter_cone.GetOutput())
        append_filter.Update()

        self.implant_polydata = append_filter.GetOutput()

        # Transform filter for world kinematic pose
        self.world_transform = vtk.vtkTransform()

        self.implant_tf_filter = vtk.vtkTransformPolyDataFilter()
        self.implant_tf_filter.SetInputData(self.implant_polydata)
        self.implant_tf_filter.SetTransform(self.world_transform)

        self.implant_mapper = vtk.vtkPolyDataMapper()
        self.implant_mapper.SetInputConnection(self.implant_tf_filter.GetOutputPort())

        self.implant_actor = vtk.vtkActor()
        self.implant_actor.SetMapper(self.implant_mapper)
        self.implant_actor.GetProperty().SetColor(0.0, 0.86, 0.91)  # Default Electric Cyan
        self.implant_actor.GetProperty().SetSpecular(0.8)
        self.implant_actor.GetProperty().SetSpecularPower(50.0)
        self.implant_actor.GetProperty().SetAmbient(0.3)
        self.implant_actor.GetProperty().SetDiffuse(0.7)

        # 4. Concentric 2.0mm Safety Envelope (Cylinder shell)
        r_sleeve = r_platform + self.safety_margin_mm
        l_sleeve = l_total + self.safety_margin_mm

        sleeve_src = vtk.vtkCylinderSource()
        sleeve_src.SetRadius(r_sleeve)
        sleeve_src.SetHeight(l_sleeve)
        sleeve_src.SetResolution(24)
        sleeve_src.CappingOn()

        tf_sleeve = vtk.vtkTransform()
        tf_sleeve.RotateX(90)
        tf_sleeve.Translate(0.0, -l_sleeve * 0.5, 0.0)

        tf_filter_sleeve = vtk.vtkTransformPolyDataFilter()
        tf_filter_sleeve.SetInputConnection(sleeve_src.GetOutputPort())
        tf_filter_sleeve.SetTransform(tf_sleeve)
        tf_filter_sleeve.Update()

        self.sleeve_polydata = tf_filter_sleeve.GetOutput()

        self.sleeve_tf_filter = vtk.vtkTransformPolyDataFilter()
        self.sleeve_tf_filter.SetInputData(self.sleeve_polydata)
        self.sleeve_tf_filter.SetTransform(self.world_transform)

        self.sleeve_mapper = vtk.vtkPolyDataMapper()
        self.sleeve_mapper.SetInputConnection(self.sleeve_tf_filter.GetOutputPort())

        self.sleeve_actor = vtk.vtkActor()
        self.sleeve_actor.SetMapper(self.sleeve_mapper)
        self.sleeve_actor.GetProperty().SetColor(0.0, 1.0, 0.5)
        self.sleeve_actor.GetProperty().SetOpacity(0.25)
        self.sleeve_actor.GetProperty().SetRepresentationToWireframe()

    def set_dimensions(self, diameter_mm: float, length_mm: float) -> None:
        """Dynamically modifies diameter and length and rebuilds mesh."""
        self.diameter_mm = max(2.5, float(diameter_mm))
        self.length_mm = max(6.0, float(length_mm))
        self._build_3d_geometry()
        self.update_transform()

    def update_transform(self) -> None:
        """Recomputes the 4x4 rigid transformation matrix from pose angles and translation."""
        self.world_transform.Identity()
        self.world_transform.Translate(self.position[0], self.position[1], self.position[2])
        self.world_transform.RotateZ(self.rotation_z_deg)
        self.world_transform.RotateX(self.bl_angle_deg)
        self.world_transform.RotateY(self.md_angle_deg)
        self.implant_tf_filter.Modified()
        self.sleeve_tf_filter.Modified()

    def get_apical_tip_world(self) -> Tuple[float, float, float]:
        """Calculates 3D physical world coordinate of the implant apical tip."""
        # In local coordinates, apex is at (0, 0, -length_mm)
        local_apex = [0.0, 0.0, -self.length_mm, 1.0]
        matrix = self.world_transform.GetMatrix()
        world_apex = [0.0, 0.0, 0.0, 0.0]
        matrix.MultiplyPoint(local_apex, world_apex)
        return (world_apex[0], world_apex[1], world_apex[2])

    def get_platform_center_world(self) -> Tuple[float, float, float]:
        """Returns 3D physical world coordinate of platform center."""
        return tuple(self.position)

    def get_surface_sample_points(self, num_rings: int = 10, pts_per_ring: int = 12) -> np.ndarray:
        """
        Generates discrete 3D world coordinates across the implant root surface for distance queries.
        """
        matrix = self.world_transform.GetMatrix()
        r_platform = self.diameter_mm * 0.5
        r_apex = r_platform * 0.7
        l_total = self.length_mm

        samples = []
        # Sample apical tip point
        samples.append(self.get_apical_tip_world())

        # Sample concentric rings from apex to platform
        z_fractions = np.linspace(0.0, 1.0, num_rings)
        for zf in z_fractions:
            lz = -l_total * (1.0 - zf)
            lr = r_apex + (r_platform - r_apex) * zf
            for theta in np.linspace(0, 2 * math.pi, pts_per_ring, endpoint=False):
                lx = lr * math.cos(theta)
                ly = lr * math.sin(theta)
                p_in = [lx, ly, lz, 1.0]
                p_out = [0.0, 0.0, 0.0, 0.0]
                matrix.MultiplyPoint(p_in, p_out)
                samples.append((p_out[0], p_out[1], p_out[2]))

        return np.array(samples, dtype=np.float32)

    def set_safety_state(self, min_dist_mm: float, nearest_name: str = "none") -> None:
        """Updates clearance distance and color codes actors."""
        self.min_nerve_dist_mm = float(min_dist_mm)
        self.nearest_nerve_name = nearest_name

        if min_dist_mm >= 2.0:
            self.safety_state = ImplantSafetyState.SAFE
            # Emerald Green / Cyan
            color = (0.0, 0.95, 0.5)
            sleeve_color = (0.0, 1.0, 0.5)
        elif min_dist_mm >= 1.5:
            self.safety_state = ImplantSafetyState.WARNING
            # Vibrant Amber / Yellow
            color = (1.0, 0.84, 0.0)
            sleeve_color = (1.0, 0.84, 0.0)
        else:
            self.safety_state = ImplantSafetyState.BREACH
            # Crimson Red Critical Collision Alert
            color = (1.0, 0.05, 0.15)
            sleeve_color = (1.0, 0.0, 0.0)

        self.implant_actor.GetProperty().SetColor(*color)
        self.sleeve_actor.GetProperty().SetColor(*sleeve_color)


class ImplantManager:
    """
    Master Registry & Collision Engine for Virtual Dental Implant Placements.
    """

    def __init__(self) -> None:
        self.signals = ImplantSignals()
        self.implants: Dict[str, DentalImplant] = {}
        self.active_implant_id: Optional[str] = None
        self._next_id_counter: int = 1

    def add_implant(
        self,
        tooth_number: int = 19,
        diameter_mm: float = 4.0,
        length_mm: float = 11.5,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> DentalImplant:
        """Creates and registers a new dental implant in the case."""
        implant_id = f"implant_{self._next_id_counter}"
        self._next_id_counter += 1

        implant = DentalImplant(
            implant_id=implant_id,
            tooth_number=tooth_number,
            diameter_mm=diameter_mm,
            length_mm=length_mm,
            position=position
        )
        self.implants[implant_id] = implant
        self.active_implant_id = implant_id

        self.signals.implant_added.emit(implant_id)
        self.signals.implant_selected.emit(implant_id)
        return implant

    def remove_implant(self, implant_id: str) -> None:
        """Deletes an implant from the case."""
        if implant_id in self.implants:
            del self.implants[implant_id]
            self.signals.implant_removed.emit(implant_id)
            if self.active_implant_id == implant_id:
                self.active_implant_id = next(iter(self.implants)) if self.implants else None
                if self.active_implant_id:
                    self.signals.implant_selected.emit(self.active_implant_id)

    def get_active_implant(self) -> Optional[DentalImplant]:
        """Returns currently selected active implant."""
        if self.active_implant_id and self.active_implant_id in self.implants:
            return self.implants[self.active_implant_id]
        return None

    def set_active_implant(self, implant_id: str) -> None:
        """Selects the active implant."""
        if implant_id in self.implants:
            self.active_implant_id = implant_id
            self.signals.implant_selected.emit(implant_id)

    def evaluate_nerve_clearance(self, nerve_tracer: NerveTracer) -> None:
        """
        Evaluates real-time Euclidean distance from all implants to Mandibular Nerve canals.
        Updates safety state and emits safety_status_changed signal.
        """
        for implant_id, implant in self.implants.items():
            pts = implant.get_surface_sample_points(num_rings=8, pts_per_ring=8)
            min_dist = float('inf')
            nearest_name = "none"

            for p in pts:
                d, name = nerve_tracer.get_distance_to_nearest_nerve((float(p[0]), float(p[1]), float(p[2])))
                if d < min_dist:
                    min_dist = d
                    nearest_name = name or "none"

            # Deduct nerve radius (~1.0mm) for true surface-to-surface clearance
            canal_clearance = max(0.0, min_dist - 1.0) if min_dist != float('inf') else float('inf')
            implant.set_safety_state(canal_clearance, nearest_name)

            self.signals.safety_status_changed.emit(
                implant_id,
                canal_clearance,
                implant.safety_state.value,
                nearest_name
            )
