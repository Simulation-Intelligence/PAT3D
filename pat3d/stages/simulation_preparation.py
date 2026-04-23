from __future__ import annotations

from dataclasses import dataclass

from pat3d.contracts import MeshSimplifier
from pat3d.models import ObjectAssetCatalog, PhysicsReadyScene, SceneLayout


@dataclass(slots=True)
class SimulationPreparationOutputs:
    scene_layout: SceneLayout
    object_assets: ObjectAssetCatalog
    physics_ready_scene: PhysicsReadyScene


class SimulationPreparationStage:
    stage_name = "simulation_preparation"

    def __init__(self, mesh_simplifier: MeshSimplifier) -> None:
        self._mesh_simplifier = mesh_simplifier

    def run(
        self,
        scene_layout: SceneLayout,
        object_assets: ObjectAssetCatalog,
    ) -> SimulationPreparationOutputs:
        physics_ready_scene = self._mesh_simplifier.simplify(scene_layout, object_assets)
        return SimulationPreparationOutputs(
            scene_layout=scene_layout,
            object_assets=object_assets,
            physics_ready_scene=physics_ready_scene,
        )

