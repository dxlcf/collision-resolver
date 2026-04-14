from __future__ import annotations

import numpy as np

from collision_resolver.config import ResolutionConfig
from collision_resolver.models import (
    AABB,
    CollisionDetectionResult,
    MeshInstance,
    OptimizationResult,
    OptimizationSummary,
    PlanarDelta,
    Pose,
    PreprocessedMesh,
    ResolutionState,
    Stage,
    StageDiagnostic,
)
from collision_resolver.planar_configuration import solve_minimum_planar_translation


class PoseResolver:
    def __init__(self, config: ResolutionConfig) -> None:
        self.config = config

    def optimize(
        self,
        movable: MeshInstance,
        movable_artifact: PreprocessedMesh,
        obstacles: tuple[MeshInstance, ...],
        obstacle_artifacts: tuple[PreprocessedMesh, ...],
        initial_detection: CollisionDetectionResult,
    ) -> OptimizationResult:
        diagnostics: list[StageDiagnostic] = []
        epsilon_xy = self._epsilon_xy(movable_artifact)

        if not initial_detection.is_colliding:
            planar_delta = PlanarDelta(dx=0.0, dy=0.0, translation_norm=0.0)
            return OptimizationResult(
                status=ResolutionState.SKIPPED,
                candidate_pose=movable.pose,
                planar_delta=planar_delta,
                summary=self._build_summary(
                    movable_artifact=movable_artifact,
                    obstacle_artifacts=obstacle_artifacts,
                    epsilon_xy=epsilon_xy,
                    iterations=0,
                    stop_reason="initial_clear",
                    forbidden_region_count=0,
                    search_radius=0.0,
                ),
                diagnostics=tuple(diagnostics),
            )

        movable_world_vertices = movable.pose.apply(movable_artifact.normalized_vertices)
        movable_world_aabb = AABB.from_points(movable_world_vertices)
        obstacle_world_aabbs = tuple(
            AABB.from_points(obstacle.pose.apply(obstacle_artifact.normalized_vertices))
            for obstacle, obstacle_artifact in zip(obstacles, obstacle_artifacts, strict=True)
        )

        convex_pair_meshes: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for obstacle, obstacle_artifact in zip(obstacles, obstacle_artifacts, strict=True):
            for movable_part in movable_artifact.convex_parts:
                movable_part_world = movable.pose.apply(movable_part.vertices)
                for obstacle_part in obstacle_artifact.convex_parts:
                    obstacle_part_world = obstacle.pose.apply(obstacle_part.vertices)
                    convex_pair_meshes.append(
                        (
                            movable_part_world,
                            movable_part.faces,
                            obstacle_part_world,
                            obstacle_part.faces,
                        )
                    )

        search_result = solve_minimum_planar_translation(
            convex_pair_meshes=tuple(convex_pair_meshes),
            movable_world_aabb=movable_world_aabb,
            obstacle_world_aabbs=obstacle_world_aabbs,
            epsilon_xy=epsilon_xy,
        )
        if search_result is None:
            diagnostics.append(
                StageDiagnostic(
                    stage=Stage.RESOLVE,
                    code="PLANAR_SEARCH_FAILED",
                    message="平面配置空间未能生成满足最优性公差要求的候选平移解。",
                )
            )
            return OptimizationResult(
                status=ResolutionState.FAILED,
                candidate_pose=None,
                planar_delta=None,
                summary=None,
                diagnostics=tuple(diagnostics),
            )

        dx = float(search_result.point[0])
        dy = float(search_result.point[1])
        translation = np.asarray(movable.pose.translation, dtype=np.float64).copy()
        translation[0] += dx
        translation[1] += dy
        candidate_pose = Pose(
            translation=translation,
            rotation=np.asarray(movable.pose.rotation, dtype=np.float64).copy(),
        )
        planar_delta = PlanarDelta(dx=dx, dy=dy, translation_norm=float(np.linalg.norm(search_result.point)))
        summary = self._build_summary(
            movable_artifact=movable_artifact,
            obstacle_artifacts=obstacle_artifacts,
            epsilon_xy=epsilon_xy,
            iterations=search_result.forbidden_region_count,
            stop_reason="global_minimum_with_tolerance",
            forbidden_region_count=search_result.forbidden_region_count,
            search_radius=search_result.search_radius,
        )
        return OptimizationResult(
            status=ResolutionState.CANDIDATE,
            candidate_pose=candidate_pose,
            planar_delta=planar_delta,
            summary=summary,
            diagnostics=tuple(diagnostics),
        )

    def _epsilon_xy(self, movable_artifact: PreprocessedMesh) -> float:
        shortest_side = float(np.min(movable_artifact.metadata.extents))
        return max(shortest_side * 0.05, 1e-9)

    def _build_summary(
        self,
        *,
        movable_artifact: PreprocessedMesh,
        obstacle_artifacts: tuple[PreprocessedMesh, ...],
        epsilon_xy: float,
        iterations: int,
        stop_reason: str,
        forbidden_region_count: int,
        search_radius: float,
    ) -> OptimizationSummary:
        movable_obb_side_lengths = np.asarray(movable_artifact.metadata.extents, dtype=np.float64)
        return OptimizationSummary(
            iterations=iterations,
            stop_reason=stop_reason,
            optimality_tolerance=epsilon_xy,
            motion_constraints={
                "rotation_locked": True,
                "z_locked": True,
                "xy_only": True,
                "objective": "min_planar_translation_l2",
            },
            search_metadata={
                "movable_obb_side_lengths": movable_obb_side_lengths.tolist(),
                "epsilon_xy": epsilon_xy,
                "forbidden_region_count": forbidden_region_count,
                "exact_verification_count": 0,
                "sdf_acceleration_used": all(artifact.sdf_field is not None for artifact in obstacle_artifacts),
                "search_radius": search_radius,
            },
            sdf_metadata=self._sdf_metadata(obstacle_artifacts),
        )

    def _sdf_metadata(self, obstacle_artifacts: tuple[PreprocessedMesh, ...]) -> dict[str, object]:
        return {
            artifact.resource_id: {
                "cache_key": str(artifact.cache_key),
                **artifact.sdf_field.summary(),
            }
            for artifact in obstacle_artifacts
            if artifact.sdf_field is not None
        }
