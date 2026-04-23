from __future__ import annotations

from pathlib import Path
import shutil
from typing import Callable

from PIL import Image, ImageDraw

from pat3d.contracts import SceneRenderer
from pat3d.models import ArtifactRef, PhysicsOptimizationResult, PhysicsReadyScene, RenderResult, SceneLayout
from pat3d.storage import make_stage_metadata


class LegacySceneRenderer(SceneRenderer):
    def __init__(
        self,
        *,
        renderer: Callable[[SceneLayout | PhysicsReadyScene | PhysicsOptimizationResult], RenderResult | dict] | None = None,
        metadata_factory: Callable[..., object] = make_stage_metadata,
        output_root: str = "results/rendered_images",
        reference_image_root: str = "data/ref_img",
        legacy_root: str = "pat3d/legacy_vendor/mask_loss",
    ) -> None:
        self._renderer = renderer
        self._metadata_factory = metadata_factory
        self._output_root = output_root
        self._reference_image_root = reference_image_root
        self._legacy_root = legacy_root

    def render(
        self,
        scene_state: SceneLayout | PhysicsReadyScene | PhysicsOptimizationResult,
    ) -> RenderResult:
        if self._renderer is not None:
            result = self._renderer(scene_state)
            if isinstance(result, RenderResult):
                return result
            return RenderResult(
                scene_id=scene_state.scene_id,
                render_images=tuple(result["render_images"]),
                camera_metadata=result.get("camera_metadata"),
                render_config=result.get("render_config"),
                metadata=self._metadata_factory(
                    stage_name="visualization",
                    provider_name="legacy_renderer",
                ),
            )

        output_path = Path(self._output_root) / scene_state.scene_id / "preview.png"
        self._materialize_fallback_preview(scene_state.scene_id, output_path)
        reference_artifact = ArtifactRef(
            artifact_type="render_image",
            path=str(output_path),
            format="png",
            role="preview",
            metadata_path=None,
        )
        return RenderResult(
            scene_id=scene_state.scene_id,
            render_images=(reference_artifact,),
            camera_metadata=ArtifactRef(
                artifact_type="legacy_renderer_reference",
                path=str(Path(self._legacy_root) / "render" / "render.py"),
                format="py",
                role="legacy_source",
                metadata_path=None,
            ),
            render_config={"mode": "fallback"},
            metadata=self._metadata_factory(
                stage_name="visualization",
                provider_name="legacy_renderer_fallback",
            ),
        )

    def _materialize_fallback_preview(self, scene_id: str, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        source = Path(self._reference_image_root) / f"{scene_id}.png"
        if source.exists():
            shutil.copyfile(source, output_path)
            return
        image = Image.new("RGB", (512, 512), color=(240, 236, 228))
        draw = ImageDraw.Draw(image)
        draw.text((24, 24), f"PAT3D preview\n{scene_id}", fill=(32, 32, 32))
        image.save(output_path)
