from __future__ import annotations

from itertools import product

import numpy as np


def clamp_circle_center_to_bounds(
    center_xz: np.ndarray,
    *,
    radius: float,
    bounds_xz: np.ndarray,
) -> np.ndarray:
    center = np.asarray(center_xz, dtype=float).reshape(2)
    bounds = np.asarray(bounds_xz, dtype=float).reshape(2, 2)
    min_corner = bounds[0]
    max_corner = bounds[1]
    if np.any(max_corner <= min_corner):
        return center.copy()

    min_allowed = min_corner + radius
    max_allowed = max_corner - radius
    if np.any(max_allowed < min_allowed):
        return ((min_corner + max_corner) * 0.5).astype(float)
    return np.clip(center, min_allowed, max_allowed).astype(float)


def pack_circle_centers_in_bounds(
    preferred_centers_xz: np.ndarray,
    radii: np.ndarray,
    *,
    bounds_xz: np.ndarray,
    gap: float = 0.0,
    grid_resolution: int = 17,
) -> np.ndarray:
    centers = np.asarray(preferred_centers_xz, dtype=float)
    normalized_radii = np.asarray(radii, dtype=float).reshape(-1)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("preferred_centers_xz must be shape (N, 2)")
    if centers.shape[0] != normalized_radii.shape[0]:
        raise ValueError("preferred_centers_xz and radii must have the same length")
    if centers.shape[0] == 0:
        return np.empty((0, 2), dtype=float)

    safe_gap = max(float(gap), 0.0)
    bounds = np.asarray(bounds_xz, dtype=float).reshape(2, 2)
    order = np.argsort(-normalized_radii, kind="stable")
    packed = np.zeros_like(centers, dtype=float)
    placed_indices: list[int] = []

    for current_index in order:
        radius = max(float(normalized_radii[current_index]), 0.0)
        preferred = clamp_circle_center_to_bounds(
            centers[current_index],
            radius=radius,
            bounds_xz=bounds,
        )
        candidates = _candidate_centers(
            preferred,
            radius=radius,
            bounds_xz=bounds,
            grid_resolution=grid_resolution,
        )
        best_center = preferred
        best_clearance = -np.inf
        chosen = False
        for candidate in candidates:
            min_clearance = np.inf
            for placed_index in placed_indices:
                required_distance = radius + float(normalized_radii[placed_index]) + safe_gap
                actual_distance = float(np.linalg.norm(candidate - packed[placed_index]))
                min_clearance = min(min_clearance, actual_distance - required_distance)
            if not placed_indices:
                min_clearance = np.inf
            if min_clearance > best_clearance:
                best_clearance = min_clearance
                best_center = candidate
            if min_clearance >= -1e-8:
                best_center = candidate
                chosen = True
                break
        packed[current_index] = best_center
        placed_indices.append(current_index)
        if not chosen:
            packed[current_index] = _relax_circle_against_placed(
                packed[current_index],
                radius=radius,
                packed=packed,
                placed_indices=placed_indices,
                radii=normalized_radii,
                bounds_xz=bounds,
                gap=safe_gap,
            )

    return packed


def _candidate_centers(
    preferred_center: np.ndarray,
    *,
    radius: float,
    bounds_xz: np.ndarray,
    grid_resolution: int,
) -> list[np.ndarray]:
    bounds = np.asarray(bounds_xz, dtype=float).reshape(2, 2)
    min_corner = bounds[0]
    max_corner = bounds[1]
    center = ((min_corner + max_corner) * 0.5).astype(float)
    min_allowed = min_corner + radius
    max_allowed = max_corner - radius
    if np.any(max_allowed < min_allowed):
        return [center]

    resolution = max(3, min(int(grid_resolution), 41))
    xs = np.linspace(min_allowed[0], max_allowed[0], resolution)
    zs = np.linspace(min_allowed[1], max_allowed[1], resolution)
    candidates = [preferred_center, center]
    for x_value, z_value in product(xs, zs):
        candidates.append(np.array([float(x_value), float(z_value)], dtype=float))

    unique_candidates: list[np.ndarray] = []
    seen_keys: set[tuple[float, float]] = set()
    for candidate in candidates:
        clamped = clamp_circle_center_to_bounds(candidate, radius=radius, bounds_xz=bounds)
        key = (round(float(clamped[0]), 6), round(float(clamped[1]), 6))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_candidates.append(clamped)
    unique_candidates.sort(key=lambda value: float(np.linalg.norm(value - preferred_center)))
    return unique_candidates


def _relax_circle_against_placed(
    current_center: np.ndarray,
    *,
    radius: float,
    packed: np.ndarray,
    placed_indices: list[int],
    radii: np.ndarray,
    bounds_xz: np.ndarray,
    gap: float,
) -> np.ndarray:
    adjusted = np.asarray(current_center, dtype=float).copy()
    bounds = np.asarray(bounds_xz, dtype=float).reshape(2, 2)
    for _ in range(32):
        changed = False
        for placed_index in placed_indices[:-1]:
            other_center = packed[placed_index]
            required_distance = radius + float(radii[placed_index]) + gap
            delta = adjusted - other_center
            distance = float(np.linalg.norm(delta))
            if distance >= required_distance - 1e-8:
                continue
            if distance <= 1e-8:
                delta = np.array([1.0, 0.0], dtype=float)
                distance = 1.0
            direction = delta / distance
            adjusted = adjusted + direction * (required_distance - distance)
            adjusted = clamp_circle_center_to_bounds(adjusted, radius=radius, bounds_xz=bounds)
            changed = True
        if not changed:
            break
    return adjusted
