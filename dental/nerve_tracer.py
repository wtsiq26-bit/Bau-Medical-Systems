"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Module: dental/nerve_tracer.py

Interactive 3D and 2D Mandibular Inferior Alveolar Nerve Tracing System.
Features:
- Parametric 3D Cardinal Spline interpolation with hardware-accelerated tube meshing (vtkTubeFilter).
- Dynamic 2D slice intersection calculation for Axial, Coronal, and Sagittal orthogonal planes.
- Real-time nerve canal diameter adjustment (0.8mm to 4.0mm).
- Distance query engine for future implant collision detection and safety margins.
- Dual-channel support (Left and Right Mandibular Canals).
- Export of anatomical coordinates to JSON and CSV formats.
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import vtk
from PySide6.QtCore import QObject, Signal


class NerveChannel(Enum):
    """Identifies the anatomical mandibular nerve channel."""
    LEFT = "left"
    RIGHT = "right"


@dataclass
class NerveColorConfig:
    """Surgical display colors for Left and Right mandibular nerve tracks."""
    left_color: Tuple[float, float, float] = (1.0, 0.176, 0.333)   # Surgical Neon Pink (#ff2d55)
    right_color: Tuple[float, float, float] = (1.0, 0.584, 0.0)    # Neon Orange (#ff9500)
    sphere_color: Tuple[float, float, float] = (1.0, 1.0, 0.2)    # Brilliant Yellow highlight


class NerveTracerSignals(QObject):
    """Signals emitted when nerve geometry or points are modified."""
    nerve_updated = Signal(str, int, float)  # channel_name, num_points, total_length_mm
    point_added = Signal(str, float, float, float) # channel_name, x, y, z
    nerve_cleared = Signal(str)              # channel_name


class NerveTrack:
    """Manages geometry, spline generation, and 3D actors for a single nerve channel."""

    def __init__(self, channel: NerveChannel, color: Tuple[float, float, float], radius_mm: float = 1.0) -> None:
        self.channel = channel
        self.color = color
        self.radius_mm = radius_mm
        self.points: List[Tuple[float, float, float]] = []

        # 1. 3D Spline Pipeline
        self.vtk_points = vtk.vtkPoints()
        # Initialize with 2 dummy points so pipeline is valid on creation
        self.vtk_points.InsertNextPoint(0.0, 0.0, 0.0)
        self.vtk_points.InsertNextPoint(0.0, 0.0, 0.1)

        self.spline = vtk.vtkParametricSpline()
        self.spline.SetPoints(self.vtk_points)
        self.spline.ParameterizeByLengthOn()

        self.function_source = vtk.vtkParametricFunctionSource()
        self.function_source.SetParametricFunction(self.spline)
        self.function_source.SetUResolution(120)

        # 2. 3D Tube Filter
        self.tube_filter = vtk.vtkTubeFilter()
        self.tube_filter.SetInputConnection(self.function_source.GetOutputPort())
        self.tube_filter.SetRadius(self.radius_mm)
        self.tube_filter.SetNumberOfSides(24)
        self.tube_filter.CappingOn()

        self.tube_mapper = vtk.vtkPolyDataMapper()
        self.tube_mapper.SetInputConnection(self.tube_filter.GetOutputPort())

        self.tube_actor = vtk.vtkActor()
        self.tube_actor.SetMapper(self.tube_mapper)
        self.tube_actor.GetProperty().SetColor(*self.color)
        self.tube_actor.GetProperty().SetOpacity(0.95)
        self.tube_actor.GetProperty().SetAmbient(0.3)
        self.tube_actor.GetProperty().SetDiffuse(0.7)
        self.tube_actor.GetProperty().SetSpecular(0.8)
        self.tube_actor.GetProperty().SetSpecularPower(40.0)
        self.tube_actor.VisibilityOff()

        # 3. 3D Seed Point Sphere Glyphs
        self.sphere_source = vtk.vtkSphereSource()
        self.sphere_source.SetRadius(self.radius_mm * 0.75)
        self.sphere_source.SetThetaResolution(16)
        self.sphere_source.SetPhiResolution(16)

        self.glyph_points = vtk.vtkPolyData()
        self.glyph_points.SetPoints(self.vtk_points)

        self.glyph3d = vtk.vtkGlyph3D()
        self.glyph3d.SetSourceConnection(self.sphere_source.GetOutputPort())
        self.glyph3d.SetInputData(self.glyph_points)
        self.glyph3d.ScalingOff()

        self.glyph_mapper = vtk.vtkPolyDataMapper()
        self.glyph_mapper.SetInputConnection(self.glyph3d.GetOutputPort())

        self.glyph_actor = vtk.vtkActor()
        self.glyph_actor.SetMapper(self.glyph_mapper)
        self.glyph_actor.GetProperty().SetColor(1.0, 0.9, 0.2) # High-visibility yellow seed spheres
        self.glyph_actor.GetProperty().SetSpecular(0.5)
        self.glyph_actor.VisibilityOff()

    def add_point(self, x: float, y: float, z: float) -> None:
        """Appends a 3D coordinate point to the nerve track."""
        self.points.append((float(x), float(y), float(z)))
        self._rebuild_pipeline()

    def undo_last_point(self) -> Optional[Tuple[float, float, float]]:
        """Removes and returns the last added coordinate point."""
        if not self.points:
            return None
        removed = self.points.pop()
        self._rebuild_pipeline()
        return removed

    def clear(self) -> None:
        """Clears all points and hides the actors."""
        self.points.clear()
        self.vtk_points.Reset()
        self.tube_actor.VisibilityOff()
        self.glyph_actor.VisibilityOff()

    def set_radius(self, radius_mm: float) -> None:
        """Updates the nerve canal tube radius in millimeters."""
        self.radius_mm = max(0.4, float(radius_mm))
        self.tube_filter.SetRadius(self.radius_mm)
        self.sphere_source.SetRadius(self.radius_mm * 0.75)
        self._rebuild_pipeline()

    def _rebuild_pipeline(self) -> None:
        """Reconstructs the parametric spline and tube mesh."""
        self.vtk_points.Reset()
        for pt in self.points:
            self.vtk_points.InsertNextPoint(pt[0], pt[1], pt[2])
        self.vtk_points.Modified()

        n = len(self.points)
        if n >= 2:
            self.spline.SetPoints(self.vtk_points)
            self.spline.Modified()
            self.function_source.Modified()
            self.function_source.Update()
            self.tube_filter.Modified()
            self.tube_filter.Update()
            self.tube_actor.VisibilityOn()
            self.glyph_points.Modified()
            self.glyph3d.Modified()
            self.glyph3d.Update()
            self.glyph_actor.VisibilityOn()
        elif n == 1:
            self.tube_actor.VisibilityOff()
            self.glyph_points.Modified()
            self.glyph3d.Modified()
            self.glyph3d.Update()
            self.glyph_actor.VisibilityOn()
        else:
            self.tube_actor.VisibilityOff()
            self.glyph_actor.VisibilityOff()

    def get_total_length_mm(self) -> float:
        """Calculates total cumulative arc length along the nerve centerline in millimeters."""
        if len(self.points) < 2:
            return 0.0

        # Sample spline at 100 points
        samples = 100
        t_vals = np.linspace(0.0, 1.0, samples)
        length = 0.0
        prev_pt = None

        pt = [0.0, 0.0, 0.0]
        for t in t_vals:
            self.spline.Evaluate([t, 0.0, 0.0], pt, [0.0] * 9)
            if prev_pt is not None:
                length += math.dist(prev_pt, pt)
            prev_pt = list(pt)

        return length

    def get_centerline_samples(self, count: int = 150) -> List[Tuple[float, float, float]]:
        """Returns evenly spaced 3D sample points along the nerve centerline."""
        if len(self.points) == 0:
            return []
        if len(self.points) == 1:
            return [self.points[0]]

        samples = []
        pt = [0.0, 0.0, 0.0]
        for t in np.linspace(0.0, 1.0, count):
            self.spline.Evaluate([t, 0.0, 0.0], pt, [0.0] * 9)
            samples.append((pt[0], pt[1], pt[2]))
        return samples

    def get_distance_to_point(self, query_pt: Tuple[float, float, float]) -> float:
        """Calculates minimum Euclidean distance in mm from query_pt to the nerve centerline."""
        samples = self.get_centerline_samples(100)
        if not samples:
            return float('inf')
        return min(math.dist(query_pt, s) for s in samples)

    def calculate_2d_slice_intersections(
        self,
        plane_type: str,
        slice_world_pos: Tuple[float, float, float]
    ) -> List[Tuple[float, float, float, float]]:
        """
        Computes 2D intersections of the nerve track with the active orthogonal slice plane.
        Returns a list of (world_x, world_y, world_z, local_radius_mm).
        """
        if len(self.points) < 2:
            return []

        intersections = []
        samples = self.get_centerline_samples(200)
        plane = plane_type.lower()

        # Coordinate axis index to test for plane crossing:
        # Axial -> Z (index 2)
        # Coronal -> Y (index 1)
        # Sagittal -> X (index 0)
        axis_idx = 2 if plane == "axial" else (1 if plane == "coronal" else 0)
        plane_val = slice_world_pos[axis_idx]

        for i in range(len(samples) - 1):
            p1 = samples[i]
            p2 = samples[i + 1]

            v1 = p1[axis_idx]
            v2 = p2[axis_idx]

            # Check if slice plane lies between p1 and p2
            if (v1 <= plane_val <= v2) or (v2 <= plane_val <= v1):
                if abs(v2 - v1) < 1e-6:
                    alpha = 0.5
                else:
                    alpha = (plane_val - v1) / (v2 - v1)

                ix = p1[0] + alpha * (p2[0] - p1[0])
                iy = p1[1] + alpha * (p2[1] - p1[1])
                iz = p1[2] + alpha * (p2[2] - p1[2])
                intersections.append((ix, iy, iz, self.radius_mm))

        return intersections


class NerveTracer:
    """
    Master controller for Dual-Channel Mandibular Inferior Alveolar Nerve Tracing.
    """

    def __init__(self) -> None:
        self.signals = NerveTracerSignals()
        self.colors = NerveColorConfig()

        # Left and Right Nerve Tracks
        self.tracks: Dict[NerveChannel, NerveTrack] = {
            NerveChannel.LEFT: NerveTrack(NerveChannel.LEFT, self.colors.left_color, radius_mm=1.0),
            NerveChannel.RIGHT: NerveTrack(NerveChannel.RIGHT, self.colors.right_color, radius_mm=1.0),
        }

        # Active Channel for drawing
        self.active_channel: NerveChannel = NerveChannel.LEFT
        self.is_drawing_active: bool = False

    def set_active_channel(self, channel: NerveChannel) -> None:
        """Sets which nerve track (Left or Right) receives incoming clicks."""
        self.active_channel = channel

    def set_drawing_mode(self, enabled: bool) -> None:
        """Enables or disables interactive seed point drawing."""
        self.is_drawing_active = enabled

    def add_point(self, x: float, y: float, z: float, channel: Optional[NerveChannel] = None) -> None:
        """Adds a 3D coordinate point to the specified or active nerve channel."""
        target_channel = channel if channel is not None else self.active_channel
        track = self.tracks[target_channel]
        track.add_point(x, y, z)

        length_mm = track.get_total_length_mm()
        self.signals.point_added.emit(target_channel.value, x, y, z)
        self.signals.nerve_updated.emit(target_channel.value, len(track.points), length_mm)

    def undo_last_point(self, channel: Optional[NerveChannel] = None) -> Optional[Tuple[float, float, float]]:
        """Removes the last seed point from the active channel."""
        target_channel = channel if channel is not None else self.active_channel
        track = self.tracks[target_channel]
        removed = track.undo_last_point()

        length_mm = track.get_total_length_mm()
        self.signals.nerve_updated.emit(target_channel.value, len(track.points), length_mm)
        return removed

    def clear_nerve(self, channel: Optional[NerveChannel] = None) -> None:
        """Clears all points for the specified or active channel."""
        target_channel = channel if channel is not None else self.active_channel
        self.tracks[target_channel].clear()
        self.signals.nerve_cleared.emit(target_channel.value)
        self.signals.nerve_updated.emit(target_channel.value, 0, 0.0)

    def set_nerve_radius(self, radius_mm: float, channel: Optional[NerveChannel] = None) -> None:
        """Sets the nerve tube radius in millimeters."""
        if channel is not None:
            self.tracks[channel].set_radius(radius_mm)
            self.signals.nerve_updated.emit(channel.value, len(self.tracks[channel].points), self.tracks[channel].get_total_length_mm())
        else:
            for ch, track in self.tracks.items():
                track.set_radius(radius_mm)
                self.signals.nerve_updated.emit(ch.value, len(track.points), track.get_total_length_mm())

    def get_track(self, channel: NerveChannel) -> NerveTrack:
        """Returns the NerveTrack for the given channel."""
        return self.tracks[channel]

    def get_distance_to_nearest_nerve(self, world_pt: Tuple[float, float, float]) -> Tuple[float, Optional[str]]:
        """Returns (min_distance_mm, nearest_channel_name) from query world point."""
        d_left = self.tracks[NerveChannel.LEFT].get_distance_to_point(world_pt)
        d_right = self.tracks[NerveChannel.RIGHT].get_distance_to_point(world_pt)

        if d_left <= d_right:
            return (d_left, "left" if d_left != float('inf') else None)
        else:
            return (d_right, "right" if d_right != float('inf') else None)

    def export_to_dict(self) -> Dict[str, Any]:
        """Serializes all nerve tracks to a dictionary."""
        return {
            "application": "Bau Medical Systems Dental CBCT 3D Viewer",
            "version": "1.0.0",
            "units": "millimeters",
            "channels": {
                ch.value: {
                    "radius_mm": track.radius_mm,
                    "total_length_mm": round(track.get_total_length_mm(), 2),
                    "point_count": len(track.points),
                    "points": [{"x": pt[0], "y": pt[1], "z": pt[2]} for pt in track.points]
                }
                for ch, track in self.tracks.items()
            }
        }

    def export_to_json(self) -> str:
        """Returns JSON string of nerve coordinates."""
        return json.dumps(self.export_to_dict(), indent=2)

    def export_to_csv(self) -> str:
        """Returns CSV string of nerve coordinates."""
        lines = ["channel,index,x_mm,y_mm,z_mm"]
        for ch, track in self.tracks.items():
            for idx, pt in enumerate(track.points):
                lines.append(f"{ch.value},{idx},{pt[0]:.3f},{pt[1]:.3f},{pt[2]:.3f}")
        return "\n".join(lines)
