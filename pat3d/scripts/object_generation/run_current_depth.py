from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PAT3D current depth estimation in a subprocess.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    try:
        from pat3d.preprocessing.depth import get_depth

        payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
        scene_name = str(payload["scene_name"])
        image_folder = str(payload["image_folder"])
        output_dir = str(payload["output_dir"])
        provider_name = str(payload.get("provider_name") or "current_depth")

        get_depth(scene_name, image_folder, output_dir)

        scene_dir = Path(output_dir) / scene_name
        output_payload = {
            "provider_name": provider_name,
            "depth_array_path": str(scene_dir / f"{scene_name}_depth.npy"),
            "depth_visualization_path": str(scene_dir / f"{scene_name}_depth.png"),
            "point_cloud_path": str(scene_dir / f"{scene_name}_point_cloud.npz"),
        }
        Path(args.output_json).write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
