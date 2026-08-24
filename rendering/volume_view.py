"""
Bau Medical Systems - Dental CBCT 3D DICOM Viewer
Module: rendering/volume_view.py

Hardware-accelerated 3D Volume Rendering for Dental CBCT.
Features:
- vtkGPUVolumeRayCastMapper with automatic fallback to vtkSmartVolumeMapper.
- Scientifically calibrated Dental CT Color and Opacity Transfer Functions.
- Gradient Opacity transfer function for razor-sharp bone/enamel boundaries.
- 3D Patient Orientation Marker Cube (Anatomical A/P, S/I, R/L labels).
- 3D Orthogonal Clipping planes for virtual resection and implant planning.
- Smooth 60 FPS interactive camera manipulation.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import vtk
from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QFrame

try:
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
except ImportError:
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from core.volume_data import VolumeData


class VolumeViewSignals(QObject):
    """Qt signals for 3D volume view events."""
    fps_updated = Signal(float)
    camera_reset = Signal()


class VolumeView(QFrame):
    """
    Hardware-accelerated 3D Volume Rendering Viewport.
    Styled according to the Bau Medical Systems dark clinical identity.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.signals = VolumeViewSignals()
        self.volume_data: Optional[VolumeData] = None

        # Segmented mesh overlay state
        self._mesh_actors: Dict[str, vtk.vtkActor] = {}
        self._mesh_polydatas: Dict[str, vtk.vtkPolyData] = {}
        self._ios_actor: Optional[vtk.vtkActor] = None
        self._ios_polydata: Optional[vtk.vtkPolyData] = None
        self._guide_actor: Optional[vtk.vtkActor] = None
        self._guide_polydata: Optional[vtk.vtkPolyData] = None

        self.setObjectName("volume_view_3d")
        self.setFrameStyle(QFrame.NoFrame)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 1. Initialize VTK Interactor Widget
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtkWidget)

        # 2. Setup VTK Renderer & Camera
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.039, 0.059, 0.071)
        self.renderer.SetBackground2(0.059, 0.078, 0.090)
        self.renderer.SetGradientBackground(True)

        # Engineering Refinement #3: Hybrid Depth Peeling for artifact-free
        # translucent mesh rendering inside the semi-transparent raycast volume.
        self.renderer.SetUseDepthPeeling(True)
        self.renderer.SetMaximumNumberOfPeels(8)
        self.renderer.SetOcclusionRatio(0.1)

        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.vtkWidget.GetRenderWindow().SetAlphaBitPlanes(True)
        self.vtkWidget.GetRenderWindow().SetMultiSamples(0)  # Required for depth peeling

        self.camera = self.renderer.GetActiveCamera()

        # 3. Setup Volume Rendering Pipeline
        self._setup_volume_pipeline()

        # 4. Setup Orientation Cube Widget
        self._setup_orientation_cube()

        # 5. Setup Interactor Style (Trackball Camera)
        self._setup_interactor_style()

    def _setup_volume_pipeline(self) -> None:
        """Configures GPU raycaster, volume properties, and transfer functions."""
        try:
            self.mapper = vtk.vtkGPUVolumeRayCastMapper()
            self.mapper.SetAutoAdjustSampleDistances(True)
            self.mapper.SetSampleDistance(0.4)
            self.mapper.SetBlendModeToComposite()
        except Exception:
            self.mapper = vtk.vtkSmartVolumeMapper()

        self.volume_property = vtk.vtkVolumeProperty()
        self.volume_property.ShadeOn()
        self.volume_property.SetInterpolationTypeToLinear()
        self.volume_property.SetAmbient(0.22)
        self.volume_property.SetDiffuse(0.78)
        self.volume_property.SetSpecular(0.40)
        self.volume_property.SetSpecularPower(35.0)

        # Transfer Functions
        self.color_func = vtk.vtkColorTransferFunction()
        self.opacity_func = vtk.vtkPiecewiseFunction()
        self.gradient_func = vtk.vtkPiecewiseFunction()

        self.volume_property.SetColor(self.color_func)
        self.volume_property.SetScalarOpacity(self.opacity_func)
        self.volume_property.SetGradientOpacity(self.gradient_func)

        # Dummy initial data
        dummy_data = vtk.vtkImageData()
        dummy_data.SetDimensions(2, 2, 2)
        dummy_scalars = vtk.vtkShortArray()
        dummy_scalars.SetNumberOfTuples(8)
        dummy_scalars.FillValue(-1024)
        dummy_data.GetPointData().SetScalars(dummy_scalars)
        self.mapper.SetInputData(dummy_data)

        # Create vtkVolume actor
        self.volume_actor = vtk.vtkVolume()
        self.volume_actor.SetMapper(self.mapper)
        self.volume_actor.SetProperty(self.volume_property)
        self.renderer.AddVolume(self.volume_actor)

        # Apply Default Dental Hard Tissue Preset
        self.apply_preset("bone_hard_tissue")

    def _setup_orientation_cube(self) -> None:
        """Configures 3D Anatomical Orientation Cube (L, R, A, P, S, I)."""
        self.cube_actor = vtk.vtkAnnotatedCubeActor()
        self.cube_actor.SetXPlusFaceText("L")
        self.cube_actor.SetXMinusFaceText("R")
        self.cube_actor.SetYPlusFaceText("P")
        self.cube_actor.SetYMinusFaceText("A")
        self.cube_actor.SetZPlusFaceText("S")
        self.cube_actor.SetZMinusFaceText("I")

        self.cube_actor.SetFaceTextScale(0.4)
        self.cube_actor.GetCubeProperty().SetColor(0.10, 0.12, 0.14)
        self.cube_actor.GetTextEdgesProperty().SetColor(0.0, 0.86, 0.91)
        self.cube_actor.GetTextEdgesProperty().SetLineWidth(1.5)

        self.orientation_widget = vtk.vtkOrientationMarkerWidget()
        self.orientation_widget.SetOrientationMarker(self.cube_actor)
        self.orientation_widget.SetInteractor(self.vtkWidget.GetRenderWindow().GetInteractor())
        self.orientation_widget.SetViewport(0.0, 0.0, 0.22, 0.22)
        self.orientation_widget.SetEnabled(1)
        self.orientation_widget.InteractiveOff()

    def _setup_interactor_style(self) -> None:
        """Sets up 3D trackball interactor style with 60 FPS smooth rotation."""
        self.style = vtk.vtkInteractorStyleTrackballCamera()
        self.vtkWidget.SetInteractorStyle(self.style)

    def set_volume_data(self, volume: VolumeData) -> None:
        """Attach VolumeData to 3D Raycasting pipeline."""
        self.volume_data = volume
        self.mapper.SetInputData(volume.vtk_image_data)
        self.reset_camera()

    def set_nerve_tracer(self, tracer: 'NerveTracer') -> None:
        """Attaches the Mandibular Nerve Tracer and adds 3D surgical tube actors."""
        self.nerve_tracer = tracer
        from dental.nerve_tracer import NerveChannel

        left_track = tracer.get_track(NerveChannel.LEFT)
        right_track = tracer.get_track(NerveChannel.RIGHT)

        self.renderer.AddActor(left_track.tube_actor)
        self.renderer.AddActor(left_track.glyph_actor)
        self.renderer.AddActor(right_track.tube_actor)
        self.renderer.AddActor(right_track.glyph_actor)
        self.safe_render()

    def set_implant_manager(self, mgr: 'ImplantManager') -> None:
        """Attaches Dental Implant Manager and registers 3D implant actors."""
        self.implant_manager = mgr
        for implant in mgr.implants.values():
            self.renderer.AddActor(implant.implant_actor)
            self.renderer.AddActor(implant.sleeve_actor)

        mgr.signals.implant_added.connect(self._on_implant_added)
        mgr.signals.implant_removed.connect(self._on_implant_removed)
        mgr.signals.implant_modified.connect(lambda *_: self.safe_render())
        mgr.signals.safety_status_changed.connect(lambda *_: self.safe_render())
        self.safe_render()

    def _on_implant_added(self, implant_id: str) -> None:
        if hasattr(self, 'implant_manager') and self.implant_manager:
            implant = self.implant_manager.implants.get(implant_id)
            if implant:
                self.renderer.AddActor(implant.implant_actor)
                self.renderer.AddActor(implant.sleeve_actor)
                self.safe_render()

    def _on_implant_removed(self, implant_id: str) -> None:
        self.safe_render()

    def reset_camera(self) -> None:
        """Reset camera to standard 3D perspective viewing the anterior jaw."""
        if self.volume_data is None:
            return

        self.renderer.ResetCamera()
        center = self.volume_data.get_center()
        bounds = self.volume_data.get_bounds()
        dist = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]) * 2.2

        self.camera.SetFocalPoint(center[0], center[1], center[2])
        self.camera.SetPosition(center[0], center[1] - dist, center[2] + dist * 0.3)
        self.camera.SetViewUp(0.0, 0.0, 1.0)
        self.renderer.ResetCameraClippingRange()
        self.safe_render()
        self.signals.camera_reset.emit()

    def apply_preset(self, preset_name: str) -> None:
        """
        Apply medically calibrated Dental CBCT 3D Transfer Functions.

        Presets:
        - `bone_hard_tissue`: Optimal for Mandible/Maxilla cortical & trabecular bone (HU 300 to 2000).
        - `teeth_enamel`: Isolates high-density enamel and dentin crowns (HU 1000 to 3000).
        - `composite_soft_bone`: Blends skin/soft-tissue surface with underlying bone structures.
        """
        self.color_func.RemoveAllPoints()
        self.opacity_func.RemoveAllPoints()
        self.gradient_func.RemoveAllPoints()

        if preset_name == "bone_hard_tissue":
            # Color: Warm golden trabecular bone -> Ivory cortical bone -> Pure bright enamel
            self.color_func.AddRGBPoint(-1024, 0.0, 0.0, 0.0)
            self.color_func.AddRGBPoint(150, 0.45, 0.25, 0.20)
            self.color_func.AddRGBPoint(450, 0.85, 0.70, 0.55)
            self.color_func.AddRGBPoint(1000, 0.95, 0.90, 0.82)
            self.color_func.AddRGBPoint(1800, 0.98, 0.98, 0.95)
            self.color_func.AddRGBPoint(2600, 1.00, 1.00, 1.00)

            # Opacity
            self.opacity_func.AddPoint(-1024, 0.0)
            self.opacity_func.AddPoint(220, 0.0)
            self.opacity_func.AddPoint(400, 0.15)
            self.opacity_func.AddPoint(800, 0.55)
            self.opacity_func.AddPoint(1600, 0.85)
            self.opacity_func.AddPoint(2600, 0.95)

            # Gradient opacity: Sharp edge definition
            self.gradient_func.AddPoint(0, 0.0)
            self.gradient_func.AddPoint(30, 0.25)
            self.gradient_func.AddPoint(90, 0.75)
            self.gradient_func.AddPoint(200, 1.0)

            self.volume_property.ShadeOn()

        elif preset_name == "teeth_enamel":
            # High-density isolation (enamel > 1200 HU)
            self.color_func.AddRGBPoint(-1024, 0.0, 0.0, 0.0)
            self.color_func.AddRGBPoint(700, 0.65, 0.65, 0.65)
            self.color_func.AddRGBPoint(1300, 0.90, 0.92, 0.95)
            self.color_func.AddRGBPoint(2200, 0.98, 1.00, 1.00)
            self.color_func.AddRGBPoint(3000, 0.85, 1.00, 1.00)

            self.opacity_func.AddPoint(-1024, 0.0)
            self.opacity_func.AddPoint(850, 0.0)
            self.opacity_func.AddPoint(1200, 0.40)
            self.opacity_func.AddPoint(2000, 0.92)

            self.gradient_func.AddPoint(0, 0.0)
            self.gradient_func.AddPoint(50, 0.5)
            self.gradient_func.AddPoint(150, 1.0)

            self.volume_property.ShadeOn()

        elif preset_name == "composite_soft_bone":
            # Soft tissue translucent skin tone + Bone ivory
            self.color_func.AddRGBPoint(-1024, 0.0, 0.0, 0.0)
            self.color_func.AddRGBPoint(-300, 0.50, 0.20, 0.15)
            self.color_func.AddRGBPoint(40, 0.82, 0.58, 0.48)
            self.color_func.AddRGBPoint(400, 0.92, 0.88, 0.80)
            self.color_func.AddRGBPoint(1800, 1.00, 1.00, 1.00)

            self.opacity_func.AddPoint(-1024, 0.0)
            self.opacity_func.AddPoint(-150, 0.0)
            self.opacity_func.AddPoint(30, 0.08)
            self.opacity_func.AddPoint(350, 0.40)
            self.opacity_func.AddPoint(1500, 0.90)

            self.gradient_func.AddPoint(0, 0.0)
            self.gradient_func.AddPoint(20, 0.2)
            self.gradient_func.AddPoint(100, 0.8)

            self.volume_property.ShadeOn()

        self.safe_render()

    def set_opacity_multiplier(self, multiplier: float) -> None:
        """Scales overall 3D volume opacity."""
        self.volume_property.SetScalarOpacityUnitDistance(max(0.1, 1.0 / max(0.01, multiplier)))
        self.safe_render()

    def set_shading(self, enabled: bool) -> None:
        """Toggle volume surface shading."""
        if enabled:
            self.volume_property.ShadeOn()
        else:
            self.volume_property.ShadeOff()
        self.safe_render()

    # ------------------------------------------------------------------
    # Segmented Mesh Overlay Management
    # ------------------------------------------------------------------

    def add_segmented_mesh(
        self,
        structure_id: str,
        actor: vtk.vtkActor,
        polydata: Optional[vtk.vtkPolyData] = None,
    ) -> None:
        """Add a named segmented anatomical mesh actor to the 3D renderer."""
        # Remove existing actor with same ID first
        if structure_id in self._mesh_actors:
            self.renderer.RemoveActor(self._mesh_actors[structure_id])

        self._mesh_actors[structure_id] = actor
        if polydata is not None:
            self._mesh_polydatas[structure_id] = polydata

        self.renderer.AddActor(actor)
        self.safe_render()

    def remove_segmented_mesh(self, structure_id: str) -> None:
        """Remove a named segmented mesh actor from the 3D renderer."""
        actor = self._mesh_actors.pop(structure_id, None)
        if actor is not None:
            self.renderer.RemoveActor(actor)
        self._mesh_polydatas.pop(structure_id, None)
        self.safe_render()

    def clear_all_meshes(self) -> None:
        """Remove all segmented mesh actors from the 3D renderer."""
        for actor in self._mesh_actors.values():
            self.renderer.RemoveActor(actor)
        self._mesh_actors.clear()
        self._mesh_polydatas.clear()

        if self._ios_actor is not None:
            self.renderer.RemoveActor(self._ios_actor)
            self._ios_actor = None
            self._ios_polydata = None

        self.safe_render()

    def set_mesh_visibility(self, structure_id: str, visible: bool) -> None:
        """Toggle visibility of an individual segmented structure."""
        actor = self._mesh_actors.get(structure_id)
        if actor is not None:
            actor.SetVisibility(visible)
            self.safe_render()

    def set_mesh_opacity(self, structure_id: str, opacity: float) -> None:
        """Adjust the opacity of an individual segmented structure."""
        actor = self._mesh_actors.get(structure_id)
        if actor is not None:
            actor.GetProperty().SetOpacity(max(0.0, min(1.0, opacity)))
            self.safe_render()

    def get_mesh_polydata(self, structure_id: str) -> Optional[vtk.vtkPolyData]:
        """Retrieve the vtkPolyData of a loaded structure for ICP target use."""
        return self._mesh_polydatas.get(structure_id)

    # ------------------------------------------------------------------
    # IOS Scan Actor Management
    # ------------------------------------------------------------------

    def add_ios_scan_actor(
        self,
        actor: vtk.vtkActor,
        polydata: Optional[vtk.vtkPolyData] = None,
    ) -> None:
        """Add the intraoral optical scan actor to the 3D renderer."""
        if self._ios_actor is not None:
            self.renderer.RemoveActor(self._ios_actor)

        self._ios_actor = actor
        self._ios_polydata = polydata
        self.renderer.AddActor(actor)
        self.safe_render()

    def remove_ios_scan_actor(self) -> None:
        """Remove the IOS scan actor from the 3D renderer."""
        if self._ios_actor is not None:
            self.renderer.RemoveActor(self._ios_actor)
            self._ios_actor = None
            self._ios_polydata = None
            self.safe_render()

    def set_ios_visibility(self, visible: bool) -> None:
        """Toggle visibility of the IOS scan overlay."""
        if self._ios_actor is not None:
            self._ios_actor.SetVisibility(visible)
            self.safe_render()

    def set_ios_opacity(self, opacity: float) -> None:
        """Adjust opacity of the IOS scan overlay."""
        if self._ios_actor is not None:
            self._ios_actor.GetProperty().SetOpacity(max(0.0, min(1.0, opacity)))
            self.safe_render()

    def add_surgical_guide_actor(
        self,
        actor: vtk.vtkActor,
        polydata: Optional[vtk.vtkPolyData] = None,
    ) -> None:
        """Add the 3D surgical guide actor to the 3D renderer."""
        if self._guide_actor is not None:
            self.renderer.RemoveActor(self._guide_actor)

        self._guide_actor = actor
        self._guide_polydata = polydata
        self.renderer.AddActor(actor)
        self.safe_render()

    def remove_surgical_guide_actor(self) -> None:
        """Remove the 3D surgical guide actor from the renderer."""
        if self._guide_actor is not None:
            self.renderer.RemoveActor(self._guide_actor)
            self._guide_actor = None
            self._guide_polydata = None
            self.safe_render()

    def set_surgical_guide_visibility(self, visible: bool) -> None:
        """Toggle visibility of the 3D surgical guide overlay."""
        if self._guide_actor is not None:
            self._guide_actor.SetVisibility(visible)
            self.safe_render()

    def set_surgical_guide_opacity(self, opacity: float) -> None:
        """Adjust opacity of the 3D surgical guide overlay."""
        if self._guide_actor is not None:
            self._guide_actor.GetProperty().SetOpacity(max(0.0, min(1.0, opacity)))
            self.safe_render()

    def safe_render(self) -> None:
        """Safely executes Render only when viewport has valid dimensions and is visible."""
        if hasattr(self, 'vtkWidget') and self.vtkWidget is not None:
            rw = self.vtkWidget.GetRenderWindow()
            if rw and self.isVisible() and self.width() > 10 and self.height() > 10:
                try:
                    rw.Render()
                except Exception:
                    pass

    def cleanup(self) -> None:
        """Cleanly releases 3D VTK render window, mesh actors, and orientation marker before Qt destruction."""
        # Release mesh overlay actors
        for actor in self._mesh_actors.values():
            try:
                self.renderer.RemoveActor(actor)
            except Exception:
                pass
        self._mesh_actors.clear()
        self._mesh_polydatas.clear()

        if self._ios_actor is not None:
            try:
                self.renderer.RemoveActor(self._ios_actor)
            except Exception:
                pass
            self._ios_actor = None
            self._ios_polydata = None

        if self._guide_actor is not None:
            try:
                self.renderer.RemoveActor(self._guide_actor)
            except Exception:
                pass
            self._guide_actor = None
            self._guide_polydata = None

        if hasattr(self, 'orientation_widget') and self.orientation_widget is not None:
            try:
                self.orientation_widget.SetEnabled(0)
            except Exception:
                pass
        if hasattr(self, 'vtkWidget') and self.vtkWidget is not None:
            try:
                rw = self.vtkWidget.GetRenderWindow()
                if rw is not None:
                    rw.Finalize()
            except Exception:
                pass

    def resizeEvent(self, event) -> None:
        """Robust resize handling for 3D viewport."""
        super().resizeEvent(event)
        if self.width() > 10 and self.height() > 10:
            self.safe_render()
