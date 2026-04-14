from __future__ import annotations

from itertools import chain

from collision_resolver.backends import FclAdapter, default_fcl_adapter
from collision_resolver.config import DetectionConfig
from collision_resolver.models import (
    AABB,
    CollisionDetectionResult,
    Contact,
    DetectionState,
    MeshInstance,
    PairCollisionResult,
    PreprocessedMesh,
    Stage,
    StageDiagnostic,
)


class CollisionDetector:
    def __init__(self, config: DetectionConfig, *, fcl_adapter: FclAdapter | None = None) -> None:
        self.config = config
        self.fcl_adapter = fcl_adapter or default_fcl_adapter()

    def detect_scene(
        self,
        movable: MeshInstance,
        movable_artifact: PreprocessedMesh,
        obstacles: tuple[MeshInstance, ...],
        obstacle_artifacts: tuple[PreprocessedMesh, ...],
    ) -> CollisionDetectionResult:
        pair_results = tuple(
            self._detect_pair(movable, movable_artifact, obstacle, artifact)
            for obstacle, artifact in zip(obstacles, obstacle_artifacts, strict=True)
        )
        diagnostics = tuple(chain.from_iterable(pair.diagnostics for pair in pair_results))
        contacts = tuple(chain.from_iterable(pair.contacts for pair in pair_results))

        if any(pair.status == DetectionState.FAILED for pair in pair_results):
            status = DetectionState.FAILED
        elif contacts:
            status = DetectionState.COLLIDING
        else:
            status = DetectionState.CLEAR

        if contacts:
            representative = max(contacts, key=lambda item: item.penetration_depth)
            max_penetration = representative.penetration_depth
            normal = representative.normal
            point = representative.point
        else:
            max_penetration = 0.0
            normal = None
            point = None

        return CollisionDetectionResult(
            status=status,
            is_colliding=bool(contacts),
            contacts=contacts,
            max_penetration_depth=max_penetration,
            representative_normal=normal,
            representative_contact_point=point,
            candidate_pair_count=sum(pair.candidate_pair_count for pair in pair_results),
            checked_pair_count=sum(pair.checked_pair_count for pair in pair_results),
            pair_results=pair_results,
            diagnostics=diagnostics,
        )

    def _detect_pair(
        self,
        movable: MeshInstance,
        movable_artifact: PreprocessedMesh,
        obstacle: MeshInstance,
        obstacle_artifact: PreprocessedMesh,
    ) -> PairCollisionResult:
        pair_id = f"{movable.resource.resource_id}->{obstacle.resource.resource_id}"
        world_aabb_movable = AABB.from_points(movable.pose.apply(movable_artifact.normalized_vertices))
        world_aabb_obstacle = AABB.from_points(obstacle.pose.apply(obstacle_artifact.normalized_vertices))

        if not world_aabb_movable.overlaps(world_aabb_obstacle, padding=self.config.broadphase_padding):
            return PairCollisionResult(
                pair_id=pair_id,
                obstacle_id=obstacle.resource.resource_id,
                status=DetectionState.CLEAR,
                is_colliding=False,
                contacts=(),
                max_penetration_depth=0.0,
                representative_normal=None,
                representative_contact_point=None,
                candidate_pair_count=0,
                checked_pair_count=0,
            )

        candidate_pairs: list[tuple[int, int]] = []
        for movable_index, movable_geometry in enumerate(movable_artifact.fcl_geometries):
            movable_part_aabb = AABB.from_points(movable.pose.apply(movable_geometry.vertices))
            for obstacle_index, obstacle_geometry in enumerate(obstacle_artifact.fcl_geometries):
                obstacle_part_aabb = AABB.from_points(obstacle.pose.apply(obstacle_geometry.vertices))
                if movable_part_aabb.overlaps(obstacle_part_aabb, padding=self.config.broadphase_padding):
                    candidate_pairs.append((movable_index, obstacle_index))

        contacts: list[Contact] = []
        diagnostics: list[StageDiagnostic] = []
        checked_pairs = 0
        try:
            for movable_index, obstacle_index in candidate_pairs:
                checked_pairs += 1
                contacts.extend(
                    self.fcl_adapter.collide(
                        pair_id,
                        obstacle.resource.resource_id,
                        movable_artifact.fcl_geometries[movable_index],
                        movable.pose,
                        obstacle_artifact.fcl_geometries[obstacle_index],
                        obstacle.pose,
                    )
                )
        except Exception as exc:
            diagnostics.append(
                StageDiagnostic(
                    stage=Stage.DETECT,
                    code="NARROWPHASE_FAILED",
                    message="精确碰撞检测阶段失败。",
                    details={"error": str(exc), "pair_id": pair_id},
                )
            )
            return PairCollisionResult(
                pair_id=pair_id,
                obstacle_id=obstacle.resource.resource_id,
                status=DetectionState.FAILED,
                is_colliding=False,
                contacts=(),
                max_penetration_depth=0.0,
                representative_normal=None,
                representative_contact_point=None,
                candidate_pair_count=len(candidate_pairs),
                checked_pair_count=checked_pairs,
                diagnostics=tuple(diagnostics),
            )

        if contacts:
            representative = max(contacts, key=lambda item: item.penetration_depth)
            max_penetration = representative.penetration_depth
            normal = representative.normal
            point = representative.point
            status = DetectionState.COLLIDING
        else:
            max_penetration = 0.0
            normal = None
            point = None
            status = DetectionState.CLEAR

        return PairCollisionResult(
            pair_id=pair_id,
            obstacle_id=obstacle.resource.resource_id,
            status=status,
            is_colliding=bool(contacts),
            contacts=tuple(contacts[: self.config.contact_limit]),
            max_penetration_depth=max_penetration,
            representative_normal=normal,
            representative_contact_point=point,
            candidate_pair_count=len(candidate_pairs),
            checked_pair_count=checked_pairs,
            diagnostics=tuple(diagnostics),
        )
