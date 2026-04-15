from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from collision_resolver.preprocess_cache import (
    PreprocessedMeshAsset,
    SDFVolume,
    build_raycast_scene,
    query_sdf_from_volume,
)


@dataclass
class DirectionLossResult:
    label: str
    sample_count: int
    penetrating_count: int
    max_penetration_depth: float
    loss: float
    bbox_min: np.ndarray | None
    bbox_max: np.ndarray | None
    penetration_points_world: np.ndarray


@dataclass
class SymmetricLossReport:
    result_b_to_a: DirectionLossResult
    result_a_to_b: DirectionLossResult
    total_loss: float
    overall_collision: bool
    overall_depth: float
    overall_bbox_min: np.ndarray | None
    overall_bbox_max: np.ndarray | None


@dataclass
class JointOptimizationReport:
    success: bool
    stop_reason: str
    iterations: int
    initial_loss: float
    final_loss: float
    transform_a_initial: np.ndarray
    transform_b_initial: np.ndarray
    transform_a_optimized: np.ndarray
    transform_b_optimized: np.ndarray
    initial_report: SymmetricLossReport
    final_report: SymmetricLossReport
    loss_history: list[float]


@dataclass
class _LossContext:
    asset_a: PreprocessedMeshAsset
    asset_b: PreprocessedMeshAsset
    scene_a: o3d.t.geometry.RaycastingScene
    scene_b: o3d.t.geometry.RaycastingScene


def identity_transform() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def validate_transform_matrix(transform: np.ndarray, label: str) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{label} must have shape (4, 4).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} contains non-finite values.")

    try:
        np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{label} must be invertible.") from exc
    return matrix


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points must be shaped as (N, 3).")

    homogeneous = np.hstack(
        [
            points.astype(np.float64),
            np.ones((len(points), 1), dtype=np.float64),
        ]
    )
    transformed = homogeneous @ transform.T
    w = transformed[:, 3:4]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("Transform produced invalid homogeneous coordinates.")
    return transformed[:, :3] / w


def _merge_bboxes(
    results: list[DirectionLossResult],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    mins = [r.bbox_min for r in results if r.bbox_min is not None]
    maxs = [r.bbox_max for r in results if r.bbox_max is not None]
    if not mins:
        return None, None
    return np.min(np.vstack(mins), axis=0), np.max(np.vstack(maxs), axis=0)


def _bbox_from_points(points: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    if len(points) == 0:
        return None, None
    return points.min(axis=0), points.max(axis=0)


def build_loss_context(
    *,
    asset_a: PreprocessedMeshAsset,
    asset_b: PreprocessedMeshAsset,
) -> _LossContext:
    return _LossContext(
        asset_a=asset_a,
        asset_b=asset_b,
        scene_a=build_raycast_scene(asset_a.mesh),
        scene_b=build_raycast_scene(asset_b.mesh),
    )


def _evaluate_symmetric_collision_loss_with_context(
    *,
    context: _LossContext,
    transform_a: np.ndarray,
    transform_b: np.ndarray,
) -> SymmetricLossReport:
    ta = validate_transform_matrix(transform_a, "transform_a")
    tb = validate_transform_matrix(transform_b, "transform_b")
    ta_inv = np.linalg.inv(ta)
    tb_inv = np.linalg.inv(tb)

    result_b_to_a = evaluate_directional_penetration_loss(
        source_surface_points_local=context.asset_b.surface_points,
        source_transform=tb,
        target_transform_inv=ta_inv,
        target_sdf_volume=context.asset_a.sdf_volume,
        target_scene=context.scene_a,
        label="B->A",
    )
    result_a_to_b = evaluate_directional_penetration_loss(
        source_surface_points_local=context.asset_a.surface_points,
        source_transform=ta,
        target_transform_inv=tb_inv,
        target_sdf_volume=context.asset_b.sdf_volume,
        target_scene=context.scene_b,
        label="A->B",
    )

    overall_bbox_min, overall_bbox_max = _merge_bboxes([result_b_to_a, result_a_to_b])
    return SymmetricLossReport(
        result_b_to_a=result_b_to_a,
        result_a_to_b=result_a_to_b,
        total_loss=float(result_b_to_a.loss + result_a_to_b.loss),
        overall_collision=(result_b_to_a.penetrating_count > 0 or result_a_to_b.penetrating_count > 0),
        overall_depth=max(result_b_to_a.max_penetration_depth, result_a_to_b.max_penetration_depth),
        overall_bbox_min=overall_bbox_min,
        overall_bbox_max=overall_bbox_max,
    )


def _skew(vector: np.ndarray) -> np.ndarray:
    vx, vy, vz = vector
    return np.array(
        [
            [0.0, -vz, vy],
            [vz, 0.0, -vx],
            [-vy, vx, 0.0],
        ],
        dtype=np.float64,
    )


def _exp_so3(omega: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(omega))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64) + _skew(omega)

    axis = omega / theta
    axis_skew = _skew(axis)
    return (
        np.eye(3, dtype=np.float64)
        + np.sin(theta) * axis_skew
        + (1.0 - np.cos(theta)) * (axis_skew @ axis_skew)
    )


def se3_delta_matrix(delta: np.ndarray) -> np.ndarray:
    delta_vec = np.asarray(delta, dtype=np.float64)
    if delta_vec.shape != (6,):
        raise ValueError("SE(3) delta must have shape (6,).")

    translation = delta_vec[:3]
    omega = delta_vec[3:]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _exp_so3(omega)
    transform[:3, 3] = translation
    return transform


def apply_joint_delta(
    *,
    transform_a: np.ndarray,
    transform_b: np.ndarray,
    joint_delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    delta_vec = np.asarray(joint_delta, dtype=np.float64)
    if delta_vec.shape != (12,):
        raise ValueError("Joint delta must have shape (12,).")

    delta_a = se3_delta_matrix(delta_vec[:6])
    delta_b = se3_delta_matrix(delta_vec[6:])
    updated_a = delta_a @ transform_a
    updated_b = delta_b @ transform_b
    return updated_a, updated_b


def _finite_difference_gradient(
    *,
    context: _LossContext,
    transform_a: np.ndarray,
    transform_b: np.ndarray,
    grad_eps: float,
) -> np.ndarray:
    gradient = np.zeros((12,), dtype=np.float64)
    for index in range(12):
        delta = np.zeros((12,), dtype=np.float64)
        delta[index] = grad_eps

        plus_a, plus_b = apply_joint_delta(
            transform_a=transform_a,
            transform_b=transform_b,
            joint_delta=delta,
        )
        minus_a, minus_b = apply_joint_delta(
            transform_a=transform_a,
            transform_b=transform_b,
            joint_delta=-delta,
        )

        loss_plus = _evaluate_symmetric_collision_loss_with_context(
            context=context,
            transform_a=plus_a,
            transform_b=plus_b,
        ).total_loss
        loss_minus = _evaluate_symmetric_collision_loss_with_context(
            context=context,
            transform_a=minus_a,
            transform_b=minus_b,
        ).total_loss
        gradient[index] = (loss_plus - loss_minus) / (2.0 * grad_eps)
    return gradient


def evaluate_directional_penetration_loss(
    *,
    source_surface_points_local: np.ndarray,
    source_transform: np.ndarray,
    target_transform_inv: np.ndarray,
    target_sdf_volume: SDFVolume,
    target_scene: o3d.t.geometry.RaycastingScene,
    label: str,
) -> DirectionLossResult:
    source_points_world = apply_transform(source_surface_points_local, source_transform)
    source_points_in_target_local = apply_transform(source_points_world, target_transform_inv)

    sdf_values = query_sdf_from_volume(
        target_sdf_volume,
        source_points_in_target_local,
        fallback_scene=target_scene,
    ).astype(np.float64)

    penetration_depths = np.maximum(0.0, -sdf_values)
    penetration_mask = penetration_depths > 0.0
    penetration_points_world = source_points_world[penetration_mask]
    bbox_min, bbox_max = _bbox_from_points(penetration_points_world)

    return DirectionLossResult(
        label=label,
        sample_count=int(len(source_surface_points_local)),
        penetrating_count=int(np.count_nonzero(penetration_mask)),
        max_penetration_depth=(
            float(np.max(penetration_depths)) if np.any(penetration_mask) else 0.0
        ),
        loss=float(np.mean(np.square(penetration_depths))),
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        penetration_points_world=penetration_points_world,
    )


def evaluate_symmetric_collision_loss(
    *,
    asset_a: PreprocessedMeshAsset,
    asset_b: PreprocessedMeshAsset,
    transform_a: np.ndarray,
    transform_b: np.ndarray,
) -> SymmetricLossReport:
    context = build_loss_context(asset_a=asset_a, asset_b=asset_b)
    return _evaluate_symmetric_collision_loss_with_context(
        context=context,
        transform_a=transform_a,
        transform_b=transform_b,
    )


def optimize_joint_transforms(
    *,
    asset_a: PreprocessedMeshAsset,
    asset_b: PreprocessedMeshAsset,
    transform_a_init: np.ndarray,
    transform_b_init: np.ndarray,
    max_iters: int = 30,
    grad_eps: float = 1e-4,
    init_step: float = 1.0,
    backtrack_factor: float = 0.5,
    armijo_c: float = 1e-4,
    grad_tol: float = 1e-6,
    loss_tol: float = 1e-10,
    min_step: float = 1e-8,
) -> JointOptimizationReport:
    if max_iters <= 0:
        raise ValueError("max_iters must be a positive integer.")
    if grad_eps <= 0.0:
        raise ValueError("grad_eps must be positive.")
    if init_step <= 0.0:
        raise ValueError("init_step must be positive.")
    if not 0.0 < backtrack_factor < 1.0:
        raise ValueError("backtrack_factor must be in (0, 1).")
    if not 0.0 < armijo_c < 1.0:
        raise ValueError("armijo_c must be in (0, 1).")
    if grad_tol < 0.0:
        raise ValueError("grad_tol must be non-negative.")
    if loss_tol < 0.0:
        raise ValueError("loss_tol must be non-negative.")
    if min_step <= 0.0:
        raise ValueError("min_step must be positive.")

    context = build_loss_context(asset_a=asset_a, asset_b=asset_b)
    transform_a = validate_transform_matrix(transform_a_init, "transform_a_init").copy()
    transform_b = validate_transform_matrix(transform_b_init, "transform_b_init").copy()
    initial_transform_a = transform_a.copy()
    initial_transform_b = transform_b.copy()

    current_report = _evaluate_symmetric_collision_loss_with_context(
        context=context,
        transform_a=transform_a,
        transform_b=transform_b,
    )
    initial_report = current_report
    current_loss = float(current_report.total_loss)
    loss_history = [current_loss]

    if current_loss <= loss_tol:
        return JointOptimizationReport(
            success=True,
            stop_reason="loss_below_tolerance",
            iterations=0,
            initial_loss=current_loss,
            final_loss=current_loss,
            transform_a_initial=initial_transform_a,
            transform_b_initial=initial_transform_b,
            transform_a_optimized=transform_a,
            transform_b_optimized=transform_b,
            initial_report=initial_report,
            final_report=current_report,
            loss_history=loss_history,
        )

    for iteration in range(1, max_iters + 1):
        gradient = _finite_difference_gradient(
            context=context,
            transform_a=transform_a,
            transform_b=transform_b,
            grad_eps=grad_eps,
        )
        grad_norm = float(np.linalg.norm(gradient))
        if grad_norm <= grad_tol:
            return JointOptimizationReport(
                success=True,
                stop_reason="gradient_below_tolerance",
                iterations=iteration - 1,
                initial_loss=loss_history[0],
                final_loss=current_loss,
                transform_a_initial=initial_transform_a,
                transform_b_initial=initial_transform_b,
                transform_a_optimized=transform_a,
                transform_b_optimized=transform_b,
                initial_report=initial_report,
                final_report=current_report,
                loss_history=loss_history,
            )

        descent_direction = -gradient
        armijo_rhs_factor = armijo_c * (grad_norm**2)
        step = init_step
        accepted = False
        accepted_report: SymmetricLossReport | None = None
        accepted_a = transform_a
        accepted_b = transform_b
        accepted_loss = current_loss

        while step >= min_step:
            candidate_a, candidate_b = apply_joint_delta(
                transform_a=transform_a,
                transform_b=transform_b,
                joint_delta=step * descent_direction,
            )
            candidate_report = _evaluate_symmetric_collision_loss_with_context(
                context=context,
                transform_a=candidate_a,
                transform_b=candidate_b,
            )
            candidate_loss = float(candidate_report.total_loss)

            if candidate_loss <= current_loss - step * armijo_rhs_factor:
                accepted = True
                accepted_a = candidate_a
                accepted_b = candidate_b
                accepted_report = candidate_report
                accepted_loss = candidate_loss
                break
            step *= backtrack_factor

        if not accepted or accepted_report is None:
            return JointOptimizationReport(
                success=False,
                stop_reason="line_search_failed",
                iterations=iteration - 1,
                initial_loss=loss_history[0],
                final_loss=current_loss,
                transform_a_initial=initial_transform_a,
                transform_b_initial=initial_transform_b,
                transform_a_optimized=transform_a,
                transform_b_optimized=transform_b,
                initial_report=initial_report,
                final_report=current_report,
                loss_history=loss_history,
            )

        transform_a = accepted_a
        transform_b = accepted_b
        current_report = accepted_report
        previous_loss = current_loss
        current_loss = accepted_loss
        loss_history.append(current_loss)

        if current_loss <= loss_tol:
            return JointOptimizationReport(
                success=True,
                stop_reason="loss_below_tolerance",
                iterations=iteration,
                initial_loss=loss_history[0],
                final_loss=current_loss,
                transform_a_initial=initial_transform_a,
                transform_b_initial=initial_transform_b,
                transform_a_optimized=transform_a,
                transform_b_optimized=transform_b,
                initial_report=initial_report,
                final_report=current_report,
                loss_history=loss_history,
            )

        if abs(previous_loss - current_loss) <= loss_tol:
            return JointOptimizationReport(
                success=True,
                stop_reason="loss_improvement_below_tolerance",
                iterations=iteration,
                initial_loss=loss_history[0],
                final_loss=current_loss,
                transform_a_initial=initial_transform_a,
                transform_b_initial=initial_transform_b,
                transform_a_optimized=transform_a,
                transform_b_optimized=transform_b,
                initial_report=initial_report,
                final_report=current_report,
                loss_history=loss_history,
            )

    return JointOptimizationReport(
        success=False,
        stop_reason="max_iters_reached",
        iterations=max_iters,
        initial_loss=loss_history[0],
        final_loss=current_loss,
        transform_a_initial=initial_transform_a,
        transform_b_initial=initial_transform_b,
        transform_a_optimized=transform_a,
        transform_b_optimized=transform_b,
        initial_report=initial_report,
        final_report=current_report,
        loss_history=loss_history,
    )
