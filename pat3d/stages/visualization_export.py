from __future__ import annotations

from dataclasses import dataclass

from pat3d.contracts import SceneRenderer
from pat3d.models import PhysicsOptimizationResult, PhysicsReadyScene, RenderResult, SceneLayout


@dataclass(slots=True)
class VisualizationExportOutputs:
    scene_state: SceneLayout | PhysicsReadyScene | PhysicsOptimizationResult
    render_result: RenderResult


class VisualizationExportStage:
    stage_name = "visualization_export"

    def __init__(self, scene_renderer: SceneRenderer) -> None:
        self._scene_renderer = scene_renderer

    def run(
        self,
        scene_state: SceneLayout | PhysicsReadyScene | PhysicsOptimizationResult,
    ) -> VisualizationExportOutputs:
        render_result = self._scene_renderer.render(scene_state)
        return VisualizationExportOutputs(
            scene_state=scene_state,
            render_result=render_result,
        )

