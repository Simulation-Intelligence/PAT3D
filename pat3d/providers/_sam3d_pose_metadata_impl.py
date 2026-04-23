from __future__ import annotations

import json
from typing import Any, Iterable

POSE_NOTE_PREFIX = "sam3d_pose="
SCENE_MANIFEST_NOTE_PREFIX = "sam3d_scene_manifest="
SCENE_GAUSSIAN_NOTE_PREFIX = "sam3d_scene_gaussian="


def _json_note(prefix: str, payload: dict[str, Any]) -> str:
    return f"{prefix}{json.dumps(payload, sort_keys=True, separators=(',', ':'))}"


def build_pose_note(
    *,
    rotation_wxyz: Iterable[float] | None,
    translation_xyz: Iterable[float] | None,
    scale_xyz: Iterable[float] | None,
) -> str:
    payload = {
        "rotation_wxyz": [float(value) for value in (rotation_wxyz or ())],
        "translation_xyz": [float(value) for value in (translation_xyz or ())],
        "scale_xyz": [float(value) for value in (scale_xyz or ())],
    }
    return _json_note(POSE_NOTE_PREFIX, payload)


def build_scene_manifest_note(path: str) -> str:
    return _json_note(SCENE_MANIFEST_NOTE_PREFIX, {"path": str(path)})


def build_scene_gaussian_note(path: str) -> str:
    return _json_note(SCENE_GAUSSIAN_NOTE_PREFIX, {"path": str(path)})


def parse_json_note(notes: Iterable[str], prefix: str) -> dict[str, Any] | None:
    for note in notes:
        if not note.startswith(prefix):
            continue
        payload = note[len(prefix) :]
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
        return None
    return None
