from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pat3d.runtime.physics_metrics import load_legacy_diff_sim_metrics, load_runtime_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract legacy Diff GIPC optimization loss history.")
    parser.add_argument("--runtime")
    parser.add_argument("--scene-id")
    parser.add_argument("--phys-root", default="_phys_result")
    args = parser.parse_args()

    if not args.runtime and not args.scene_id:
        parser.error("either --runtime or --scene-id is required")

    runtime_payload = load_runtime_payload(args.runtime) if args.runtime else None
    metrics = load_legacy_diff_sim_metrics(
        runtime_payload=runtime_payload,
        runtime_path=args.runtime,
        scene_id=args.scene_id,
        phys_result_root=args.phys_root,
    )
    sys.stdout.write(json.dumps(metrics, indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
