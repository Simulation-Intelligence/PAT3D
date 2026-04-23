from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pat3d.runtime.trajectory import load_legacy_diff_sim_trajectory, load_runtime_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract legacy Diff GIPC optimization trajectory snapshots.")
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--phys-root", default="_phys_result")
    args = parser.parse_args()

    runtime_payload = load_runtime_payload(args.runtime)
    trajectory = load_legacy_diff_sim_trajectory(
        runtime_payload,
        phys_result_root=args.phys_root,
    )
    sys.stdout.write(json.dumps(trajectory, indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
