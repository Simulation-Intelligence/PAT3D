from __future__ import annotations

import argparse
import json
from typing import Sequence


def preview_legacy_preprocess(*args, **kwargs):
    from pat3d.runtime.legacy_preprocess import preview_legacy_preprocess as _impl

    return _impl(*args, **kwargs)


def run_legacy_preprocess(*args, **kwargs):
    from pat3d.runtime.legacy_preprocess import run_legacy_preprocess as _impl

    return _impl(*args, **kwargs)


def _legacy_preprocess_failure_result(error: Exception) -> dict[str, object]:
    from pat3d.runtime.errors import coerce_runtime_failure

    return coerce_runtime_failure(
        error,
        phase="legacy_preprocess",
        code="legacy_preprocess_failed",
        user_message="The legacy preprocess request failed.",
        technical_message=str(error),
        retryable=False,
    ).to_result()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or validate the legacy preprocess path from a canonical SceneRequest."
    )
    parser.add_argument("--request", required=True, help="Path to canonical SceneRequest JSON.")
    parser.add_argument(
        "--output",
        help="Optional path to write preprocess result JSON. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate canonical preprocess mapping without executing the legacy preprocessor.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        from pat3d.runtime.execution import (
            load_json_file,
            scene_request_from_dict,
            to_jsonable,
            write_execution_result,
        )

        scene_request = scene_request_from_dict(load_json_file(args.request))

        if args.validate_only:
            result = preview_legacy_preprocess(scene_request)
        else:
            result = run_legacy_preprocess(scene_request)

        if args.output:
            write_execution_result(result, args.output)
        else:
            print(json.dumps(to_jsonable(result), indent=2, sort_keys=True))
        return 0
    except Exception as error:
        payload = _legacy_preprocess_failure_result(error)
        if args.output:
            try:
                from pat3d.runtime.execution import write_execution_result

                write_execution_result(payload, args.output)
            except Exception:
                print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
