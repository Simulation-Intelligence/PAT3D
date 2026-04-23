from __future__ import annotations

import json
import sys

from pat3d.providers.current_text_to_3d import _load_generator_from_import_path


def main() -> int:
    raw_payload = sys.stdin.read()
    if not raw_payload.strip():
        print("current_text_to_3d worker expected a JSON payload on stdin.", file=sys.stderr)
        return 2
    payload = json.loads(raw_payload)
    generator = _load_generator_from_import_path(str(payload["generator_import_path"]))
    generator(
        str(payload["prompt"]),
        str(payload["reference_image_root"]),
        str(payload["output_root"]),
        str(payload["scene_id"]),
        str(payload["object_name"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
