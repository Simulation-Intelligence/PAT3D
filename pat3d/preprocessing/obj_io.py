from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def rewrite_obj_vertex_lines(
    obj_lines: Sequence[str],
    vertices_xyz: Iterable[Sequence[float]] | np.ndarray,
) -> list[str]:
    rewritten = list(obj_lines)
    vertex_lines = [index for index, line in enumerate(rewritten) if line.startswith("v ")]
    vertices = np.asarray(list(vertices_xyz), dtype=float)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices_xyz must be an iterable of 3D coordinates.")
    if len(vertex_lines) != len(vertices):
        raise ValueError(
            "OBJ vertex rewrite expected the same number of vertex lines and coordinates "
            f"({len(vertex_lines)} != {len(vertices)})."
        )

    for line_index, (x_value, y_value, z_value) in zip(vertex_lines, vertices):
        rewritten[line_index] = f"v {float(x_value):.8f} {float(y_value):.8f} {float(z_value):.8f}\n"
    return rewritten
