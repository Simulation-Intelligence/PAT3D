from __future__ import annotations

from pat3d.contracts import PhysicsOptimizer
from pat3d.models import PhysicsOptimizationResult, PhysicsReadyScene
from pat3d.storage import make_stage_metadata


class PassthroughPhysicsOptimizer(PhysicsOptimizer):
    """Keep simulation-prep poses unchanged when private physics is unavailable."""

    def __init__(self, *, reason: str = "physics_disabled", requested_kind: str | None = None) -> None:
        self._reason = reason
        self._requested_kind = requested_kind

    def optimize(self, physics_ready_scene: PhysicsReadyScene) -> PhysicsOptimizationResult:
        notes = ("identity_passthrough", self._reason)
        if self._requested_kind:
            notes = notes + (f"requested_kind={self._requested_kind}",)
        return PhysicsOptimizationResult(
            scene_id=physics_ready_scene.scene_id,
            initial_scene=physics_ready_scene,
            optimized_object_poses=physics_ready_scene.object_poses,
            metrics={
                "identity_passthrough": 1.0,
                "physics_passthrough": 1.0,
            },
            artifacts=(),
            metadata=make_stage_metadata(
                stage_name="physics_optimization",
                provider_name="passthrough_physics",
                notes=notes,
            ),
        )
