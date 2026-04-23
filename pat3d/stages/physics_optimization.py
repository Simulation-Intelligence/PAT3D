from __future__ import annotations

from dataclasses import dataclass

from pat3d.contracts import PhysicsOptimizer
from pat3d.models import PhysicsOptimizationResult, PhysicsReadyScene


@dataclass(slots=True)
class PhysicsOptimizationOutputs:
    physics_ready_scene: PhysicsReadyScene
    optimization_result: PhysicsOptimizationResult


class PhysicsOptimizationStage:
    stage_name = "physics_optimization"

    def __init__(self, physics_optimizer: PhysicsOptimizer) -> None:
        self._physics_optimizer = physics_optimizer

    def run(self, physics_ready_scene: PhysicsReadyScene) -> PhysicsOptimizationOutputs:
        optimization_result = self._physics_optimizer.optimize(physics_ready_scene)
        return PhysicsOptimizationOutputs(
            physics_ready_scene=physics_ready_scene,
            optimization_result=optimization_result,
        )

