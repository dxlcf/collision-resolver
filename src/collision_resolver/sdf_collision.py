from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import open3d as o3d
from loguru import logger

from collision_resolver.mesh_repair import ensure_watertight_mesh

SDFQueryFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class DirectionCollisionResult:
    label: str
    collision: bool
    penetrating_count: int
    sample_count: int
    max_penetration_depth: float
    bbox_min: np.ndarray | None
    bbox_max: np.ndarray | None
    penetration_points: np.ndarray


@dataclass
class CollisionReport:
    result_b_in_a: DirectionCollisionResult
    result_a_in_b: DirectionCollisionResult
    overall_collision: bool
    overall_depth: float
    overall_bbox_min: np.ndarray | None
    overall_bbox_max: np.ndarray | None


@dataclass
class RuntimeParameters:
    scene_scale: float
    target_spacing: float
    sample_count_a: int
    sample_count_b: int
    gradient_eps: float
    safety_margin: float


@dataclass
class ResolveReport:
    resolved: bool
    iterations: int
    total_translation: np.ndarray
    first_direction: np.ndarray | None
    first_step: np.ndarray | None


@dataclass
class PushContribution:
    label: str
    collision: bool
    max_depth: float
    push_vector: np.ndarray
    fallback_direction: np.ndarray | None


def load_mesh(mesh_path: str | Path) -> o3d.geometry.TriangleMesh:
    path = Path(mesh_path)
    if not path.is_file():
        raise FileNotFoundError(f"Mesh file does not exist: {path}")

    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise ValueError(f"Failed to load mesh or mesh is empty: {path}")
    if len(mesh.triangles) == 0:
        raise ValueError(f"Mesh has no triangles: {path}")
    logger.info(
        "Loaded mesh {}: {} vertices, {} triangles",
        path,
        len(mesh.vertices),
        len(mesh.triangles),
    )

    mesh = ensure_watertight_mesh(mesh, mesh_label=path)
    mesh.compute_vertex_normals()
    return mesh


def compute_scene_scale(
    mesh_a: o3d.geometry.TriangleMesh,
    mesh_b: o3d.geometry.TriangleMesh,
) -> float:
    bbox_a = mesh_a.get_axis_aligned_bounding_box()
    bbox_b = mesh_b.get_axis_aligned_bounding_box()
    min_bound = np.minimum(bbox_a.min_bound, bbox_b.min_bound)
    max_bound = np.maximum(bbox_a.max_bound, bbox_b.max_bound)
    diagonal = float(np.linalg.norm(max_bound - min_bound))
    if diagonal <= 0.0:
        raise ValueError("Combined mesh bbox diagonal must be positive.")
    return diagonal


def compute_surface_sample_count(
    mesh: o3d.geometry.TriangleMesh,
    target_spacing: float,
    min_samples: int,
    max_samples: int,
) -> int:
    if target_spacing <= 0.0:
        raise ValueError("Auto sample spacing must be positive.")
    if min_samples <= 0:
        raise ValueError("min_samples must be a positive integer.")
    if max_samples < min_samples:
        raise ValueError("max_samples must be greater than or equal to min_samples.")

    surface_area = float(mesh.get_surface_area())
    if surface_area <= 0.0:
        return min_samples

    estimated = int(np.ceil(surface_area / (target_spacing**2)))
    return int(np.clip(estimated, min_samples, max_samples))


def resolve_runtime_parameters(
    mesh_a: o3d.geometry.TriangleMesh,
    mesh_b: o3d.geometry.TriangleMesh,
    *,
    samples: int | None,
    sample_spacing_ratio: float,
    min_samples: int,
    max_samples: int,
    gradient_eps: float | None,
    gradient_eps_ratio: float,
    safety_margin: float | None,
    safety_margin_ratio: float,
) -> RuntimeParameters:
    scene_scale = compute_scene_scale(mesh_a, mesh_b)

    if samples is not None and samples <= 0:
        raise ValueError("samples must be a positive integer.")
    if sample_spacing_ratio <= 0.0:
        raise ValueError("sample_spacing_ratio must be positive.")
    if gradient_eps is not None and gradient_eps <= 0.0:
        raise ValueError("gradient_eps must be positive.")
    if gradient_eps_ratio <= 0.0:
        raise ValueError("gradient_eps_ratio must be positive.")
    if safety_margin is not None and safety_margin < 0.0:
        raise ValueError("safety_margin must be non-negative.")
    if safety_margin_ratio < 0.0:
        raise ValueError("safety_margin_ratio must be non-negative.")

    target_spacing = scene_scale * sample_spacing_ratio
    if samples is None:
        sample_count_a = compute_surface_sample_count(
            mesh=mesh_a,
            target_spacing=target_spacing,
            min_samples=min_samples,
            max_samples=max_samples,
        )
        sample_count_b = compute_surface_sample_count(
            mesh=mesh_b,
            target_spacing=target_spacing,
            min_samples=min_samples,
            max_samples=max_samples,
        )
    else:
        sample_count_a = int(samples)
        sample_count_b = int(samples)

    resolved_gradient_eps = (
        float(gradient_eps) if gradient_eps is not None else scene_scale * gradient_eps_ratio
    )
    resolved_safety_margin = (
        float(safety_margin) if safety_margin is not None else scene_scale * safety_margin_ratio
    )

    return RuntimeParameters(
        scene_scale=scene_scale,
        target_spacing=target_spacing,
        sample_count_a=sample_count_a,
        sample_count_b=sample_count_b,
        gradient_eps=resolved_gradient_eps,
        safety_margin=resolved_safety_margin,
    )


def build_raycast_scene(mesh: o3d.geometry.TriangleMesh) -> o3d.t.geometry.RaycastingScene:
    mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_tensor)
    return scene


def query_sdf(scene: o3d.t.geometry.RaycastingScene, points: np.ndarray) -> np.ndarray:
    points_tensor = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
    return scene.compute_signed_distance(points_tensor).numpy()


def query_sdf_gradient(
    scene: o3d.t.geometry.RaycastingScene,
    points: np.ndarray,
    eps: float,
) -> np.ndarray:
    gradients = np.zeros_like(points, dtype=np.float64)
    for axis in range(3):
        delta = np.zeros(3, dtype=np.float64)
        delta[axis] = eps
        sdf_plus = query_sdf(scene, points + delta)
        sdf_minus = query_sdf(scene, points - delta)
        gradients[:, axis] = (sdf_plus - sdf_minus) / (2.0 * eps)
    return gradients


def analyze_direction(
    sdf_query: SDFQueryFn,
    sampled_mesh: o3d.geometry.TriangleMesh,
    sample_count: int,
    label: str,
    penetration_tolerance: float = 0.0,
) -> DirectionCollisionResult:
    if penetration_tolerance < 0.0:
        raise ValueError("penetration_tolerance must be non-negative.")

    sampled_pcd = sampled_mesh.sample_points_uniformly(number_of_points=sample_count)
    sampled_points = np.asarray(sampled_pcd.points)

    sdf_values = sdf_query(sampled_points)
    penetration_mask = sdf_values < -penetration_tolerance
    penetration_points = sampled_points[penetration_mask]

    if not np.any(penetration_mask):
        return DirectionCollisionResult(
            label=label,
            collision=False,
            penetrating_count=0,
            sample_count=sample_count,
            max_penetration_depth=0.0,
            bbox_min=None,
            bbox_max=None,
            penetration_points=np.empty((0, 3), dtype=float),
        )

    return DirectionCollisionResult(
        label=label,
        collision=True,
        penetrating_count=int(np.count_nonzero(penetration_mask)),
        sample_count=sample_count,
        max_penetration_depth=float(
            np.max(-sdf_values[penetration_mask] - penetration_tolerance),
        ),
        bbox_min=penetration_points.min(axis=0),
        bbox_max=penetration_points.max(axis=0),
        penetration_points=penetration_points,
    )


def merge_bboxes(
    results: list[DirectionCollisionResult],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    mins = [r.bbox_min for r in results if r.bbox_min is not None]
    maxs = [r.bbox_max for r in results if r.bbox_max is not None]
    if not mins:
        return None, None
    return np.min(np.vstack(mins), axis=0), np.max(np.vstack(maxs), axis=0)


def detect_collision(
    mesh_a: o3d.geometry.TriangleMesh,
    mesh_b: o3d.geometry.TriangleMesh,
    sample_count_a: int,
    sample_count_b: int,
    *,
    sdf_query_a: SDFQueryFn | None = None,
    sdf_query_b: SDFQueryFn | None = None,
    penetration_tolerance: float = 0.0,
) -> CollisionReport:
    if penetration_tolerance < 0.0:
        raise ValueError("penetration_tolerance must be non-negative.")

    if sdf_query_a is None:
        scene_a = build_raycast_scene(mesh_a)

        def query_a(points: np.ndarray) -> np.ndarray:
            return query_sdf(scene_a, points)

    else:
        query_a = sdf_query_a

    if sdf_query_b is None:
        scene_b = build_raycast_scene(mesh_b)

        def query_b(points: np.ndarray) -> np.ndarray:
            return query_sdf(scene_b, points)

    else:
        query_b = sdf_query_b

    result_b_in_a = analyze_direction(
        sdf_query=query_a,
        sampled_mesh=mesh_b,
        sample_count=sample_count_b,
        label="Mesh B surface inside Mesh A SDF",
        penetration_tolerance=penetration_tolerance,
    )
    result_a_in_b = analyze_direction(
        sdf_query=query_b,
        sampled_mesh=mesh_a,
        sample_count=sample_count_a,
        label="Mesh A surface inside Mesh B SDF",
        penetration_tolerance=penetration_tolerance,
    )

    overall_bbox_min, overall_bbox_max = merge_bboxes([result_b_in_a, result_a_in_b])
    return CollisionReport(
        result_b_in_a=result_b_in_a,
        result_a_in_b=result_a_in_b,
        overall_collision=result_b_in_a.collision or result_a_in_b.collision,
        overall_depth=max(result_b_in_a.max_penetration_depth, result_a_in_b.max_penetration_depth),
        overall_bbox_min=overall_bbox_min,
        overall_bbox_max=overall_bbox_max,
    )


def compute_push_contribution(
    scene: o3d.t.geometry.RaycastingScene,
    sampled_mesh: o3d.geometry.TriangleMesh,
    sample_count: int,
    eps: float,
    direction_sign: float,
    label: str,
    penetration_tolerance: float = 0.0,
) -> PushContribution:
    if penetration_tolerance < 0.0:
        raise ValueError("penetration_tolerance must be non-negative.")

    sampled_pcd = sampled_mesh.sample_points_uniformly(number_of_points=sample_count)
    sampled_points = np.asarray(sampled_pcd.points)

    sdf_values = query_sdf(scene, sampled_points)
    penetration_mask = sdf_values < -penetration_tolerance
    if not np.any(penetration_mask):
        return PushContribution(
            label=label,
            collision=False,
            max_depth=0.0,
            push_vector=np.zeros(3, dtype=np.float64),
            fallback_direction=None,
        )

    penetration_points = sampled_points[penetration_mask]
    penetration_depths = -sdf_values[penetration_mask] - penetration_tolerance
    gradients = query_sdf_gradient(scene, penetration_points, eps)
    grad_norms = np.linalg.norm(gradients, axis=1, keepdims=True)
    gradients_unit = gradients / np.clip(grad_norms, 1e-12, None)

    signed_gradients = gradients_unit * direction_sign
    push_vector = np.sum(signed_gradients * penetration_depths[:, None], axis=0)
    deepest_idx = int(np.argmax(penetration_depths))

    return PushContribution(
        label=label,
        collision=True,
        max_depth=float(np.max(penetration_depths)),
        push_vector=push_vector,
        fallback_direction=signed_gradients[deepest_idx],
    )


def resolve_collision_by_translation(
    mesh_a: o3d.geometry.TriangleMesh,
    mesh_b: o3d.geometry.TriangleMesh,
    sample_count_a: int,
    sample_count_b: int,
    eps: float,
    max_iters: int,
    safety_margin: float,
    penetration_tolerance: float = 0.0,
) -> ResolveReport:
    if eps <= 0.0:
        raise ValueError("gradient epsilon must be positive.")
    if max_iters <= 0:
        raise ValueError("max_iters must be a positive integer.")
    if safety_margin < 0.0:
        raise ValueError("safety_margin must be non-negative.")
    if penetration_tolerance < 0.0:
        raise ValueError("penetration_tolerance must be non-negative.")

    scene_a = build_raycast_scene(mesh_a)
    total_translation = np.zeros(3, dtype=np.float64)
    first_direction = None
    first_step = None

    for iteration in range(1, max_iters + 1):
        scene_b = build_raycast_scene(mesh_b)
        contribution_b_in_a = compute_push_contribution(
            scene=scene_a,
            sampled_mesh=mesh_b,
            sample_count=sample_count_b,
            eps=eps,
            direction_sign=1.0,
            label="Mesh B surface inside Mesh A SDF",
            penetration_tolerance=penetration_tolerance,
        )
        contribution_a_in_b = compute_push_contribution(
            scene=scene_b,
            sampled_mesh=mesh_a,
            sample_count=sample_count_a,
            eps=eps,
            direction_sign=-1.0,
            label="Mesh A surface inside Mesh B SDF",
            penetration_tolerance=penetration_tolerance,
        )

        if not contribution_b_in_a.collision and not contribution_a_in_b.collision:
            return ResolveReport(
                resolved=True,
                iterations=iteration - 1,
                total_translation=total_translation,
                first_direction=first_direction,
                first_step=first_step,
            )

        combined_vector = contribution_b_in_a.push_vector + contribution_a_in_b.push_vector
        vector_norm = float(np.linalg.norm(combined_vector))

        if vector_norm > 1e-12:
            direction = combined_vector / vector_norm
        else:
            preferred = contribution_b_in_a
            if contribution_a_in_b.max_depth > contribution_b_in_a.max_depth:
                preferred = contribution_a_in_b

            if preferred.fallback_direction is None:
                return ResolveReport(
                    resolved=False,
                    iterations=iteration - 1,
                    total_translation=total_translation,
                    first_direction=first_direction,
                    first_step=first_step,
                )
            direction = preferred.fallback_direction / np.clip(
                np.linalg.norm(preferred.fallback_direction),
                1e-12,
                None,
            )

        step_distance = max(contribution_b_in_a.max_depth, contribution_a_in_b.max_depth) + safety_margin
        step_vector = direction * step_distance
        mesh_b.translate(step_vector)
        total_translation += step_vector

        if first_direction is None:
            first_direction = direction.copy()
            first_step = step_vector.copy()

        logger.info(
            "Resolve iter {}: direction={}, step={}, total_translation={}",
            iteration,
            np.round(direction, 6).tolist(),
            np.round(step_vector, 6).tolist(),
            np.round(total_translation, 6).tolist(),
        )

    return ResolveReport(
        resolved=False,
        iterations=max_iters,
        total_translation=total_translation,
        first_direction=first_direction,
        first_step=first_step,
    )
