from __future__ import annotations

import argparse
import json
from typing import Sequence


def build_and_run_pipeline(*args, **kwargs):
    from pat3d.runtime.execution import build_and_run_pipeline as _impl

    return _impl(*args, **kwargs)


def _runtime_failure_result(error: Exception) -> dict[str, object]:
    from pat3d.runtime.errors import coerce_runtime_failure

    return coerce_runtime_failure(
        error,
        phase="input_loading",
        code="runtime_cli_failed",
        user_message="The runtime request could not be processed.",
        technical_message=str(error),
        retryable=False,
    ).to_result()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and run a canonical PAT3D pipeline from runtime config."
    )
    parser.add_argument("--runtime-config", required=True, help="Path to runtime config JSON.")
    parser.add_argument("--request", required=True, help="Path to scene request JSON.")
    parser.add_argument(
        "--object-catalog",
        help="Optional path to object catalog JSON matching the canonical schema.",
    )
    parser.add_argument(
        "--object-hints",
        help="Optional path to JSON string list used as object hints.",
    )
    parser.add_argument(
        "--object-reference-images",
        help="Optional path to JSON object mapping object ids to ArtifactRef payloads.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write JSON output. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate runtime config and canonical inputs without executing the pipeline.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        from pat3d.runtime.config import RuntimeConfig
        from pat3d.runtime.execution import (
            artifact_ref_map_from_dict,
            load_json_file,
            object_catalog_from_dict,
            scene_request_from_dict,
            string_list_from_json_file,
            to_jsonable,
            write_execution_result,
        )

        runtime_config = RuntimeConfig.from_json_file(args.runtime_config)
        request = scene_request_from_dict(load_json_file(args.request))
        object_catalog = (
            object_catalog_from_dict(load_json_file(args.object_catalog))
            if args.object_catalog
            else None
        )
        object_hints = string_list_from_json_file(args.object_hints) if args.object_hints else None
        object_reference_images = (
            artifact_ref_map_from_dict(load_json_file(args.object_reference_images))
            if args.object_reference_images
            else None
        )

        runtime_config.validate()
        if args.validate_only:
            result = {
                "status": "validated",
                "pipeline": runtime_config.pipeline,
                "scene_id": request.scene_id,
                "provider_roles": sorted(runtime_config.providers),
                "has_object_catalog": object_catalog is not None,
                "has_object_hints": object_hints is not None,
                "has_object_reference_images": object_reference_images is not None,
            }
        else:
            result = build_and_run_pipeline(
                runtime_config,
                request,
                object_hints=object_hints,
                object_catalog=object_catalog,
                object_reference_images=object_reference_images,
            )

        if args.output:
            write_execution_result(result, args.output)
        else:
            print(json.dumps(to_jsonable(result), indent=2, sort_keys=True))
        return 0
    except Exception as error:
        payload = _runtime_failure_result(error)
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
