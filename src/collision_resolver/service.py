from __future__ import annotations

from dataclasses import replace

from collision_resolver.config import ServiceConfig
from collision_resolver.detection import CollisionDetector
from collision_resolver.exceptions import CollisionResolverError
from collision_resolver.models import (
    FinalState,
    MeshInstance,
    RequestMode,
    ServiceRequest,
    ServiceResult,
    Stage,
    StageDiagnostic,
)
from collision_resolver.preprocess import MeshPreprocessor
from collision_resolver.resolution import PoseResolver


class CollisionResolverService:
    def __init__(
        self,
        config: ServiceConfig | None = None,
        *,
        preprocessor: MeshPreprocessor | None = None,
        detector: CollisionDetector | None = None,
        resolver: PoseResolver | None = None,
    ) -> None:
        self.config = config or ServiceConfig()
        self.preprocessor = preprocessor or MeshPreprocessor(self.config.preprocessing)
        self.detector = detector or CollisionDetector(self.config.detection)
        self.resolver = resolver or PoseResolver(self.config.resolution)

    def process(self, request: ServiceRequest) -> ServiceResult:
        diagnostics: list[StageDiagnostic] = []
        try:
            movable_artifact, movable_cache_hit = self.preprocessor.prepare(
                request.movable.resource,
                require_sdf=False,
                require_samples=True,
            )
            diagnostics.append(self._cache_diagnostic(request.movable.resource.resource_id, movable_cache_hit))

            obstacle_artifacts = []
            for obstacle in request.obstacles:
                artifact, cache_hit = self.preprocessor.prepare(obstacle.resource, require_sdf=True, require_samples=False)
                diagnostics.append(self._cache_diagnostic(obstacle.resource.resource_id, cache_hit))
                obstacle_artifacts.append(artifact)
        except CollisionResolverError as exc:
            diagnostics.append(self._error_diagnostic(Stage.PREPROCESS, exc.code, exc.message))
            return ServiceResult(
                mode=request.mode,
                status=FinalState.FAILED,
                initial_detection=None,
                optimization=None,
                verification=None,
                diagnostics=tuple(diagnostics),
            )

        initial_detection = self.detector.detect_scene(
            request.movable,
            movable_artifact,
            request.obstacles,
            tuple(obstacle_artifacts),
        )
        diagnostics.extend(initial_detection.diagnostics)

        if request.mode == RequestMode.DETECT:
            status = {
                "clear": FinalState.CLEAR,
                "colliding": FinalState.COLLIDING,
                "failed": FinalState.FAILED,
            }[initial_detection.status.value]
            return ServiceResult(
                mode=request.mode,
                status=status,
                initial_detection=initial_detection,
                optimization=None,
                verification=None,
                diagnostics=tuple(diagnostics),
            )

        if initial_detection.status.value == "failed":
            return ServiceResult(
                mode=request.mode,
                status=FinalState.FAILED,
                initial_detection=initial_detection,
                optimization=None,
                verification=None,
                diagnostics=tuple(diagnostics),
            )

        optimization = self.resolver.optimize(
            request.movable,
            movable_artifact,
            request.obstacles,
            tuple(obstacle_artifacts),
            initial_detection,
        )
        diagnostics.extend(optimization.diagnostics)

        if optimization.status.value == "failed":
            return ServiceResult(
                mode=request.mode,
                status=FinalState.FAILED,
                initial_detection=initial_detection,
                optimization=optimization,
                verification=None,
                diagnostics=tuple(diagnostics),
            )

        candidate_pose = optimization.candidate_pose or request.movable.pose
        verification = self.detector.detect_scene(
            MeshInstance(resource=request.movable.resource, pose=candidate_pose),
            movable_artifact,
            request.obstacles,
            tuple(obstacle_artifacts),
        )
        diagnostics.extend(verification.diagnostics)
        if optimization.summary is not None:
            search_metadata = dict(optimization.summary.search_metadata)
            search_metadata["exact_verification_count"] = 1
            optimization = replace(
                optimization,
                summary=replace(optimization.summary, search_metadata=search_metadata),
            )

        if verification.status.value == "failed":
            final_status = FinalState.FAILED
        elif verification.is_colliding:
            final_status = FinalState.FAILED
        else:
            final_status = FinalState.RESOLVED

        if not initial_detection.is_colliding and optimization.status.value == "skipped":
            final_status = FinalState.RESOLVED

        return ServiceResult(
            mode=request.mode,
            status=final_status,
            initial_detection=initial_detection,
            optimization=optimization,
            verification=verification,
            diagnostics=tuple(diagnostics),
        )

    def _cache_diagnostic(self, resource_id: str, cache_hit: bool) -> StageDiagnostic:
        return StageDiagnostic(
            stage=Stage.PREPROCESS,
            code="CACHE_HIT" if cache_hit else "CACHE_MISS",
            message="命中已有预处理缓存。" if cache_hit else "执行新的预处理并写入缓存。",
            details={"resource_id": resource_id},
        )

    def _error_diagnostic(self, stage: Stage, code: str, message: str) -> StageDiagnostic:
        return StageDiagnostic(stage=stage, code=code, message=message)
