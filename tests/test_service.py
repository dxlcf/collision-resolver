from __future__ import annotations

import numpy as np
import pytest

from collision_resolver.backends import default_coacd_adapter, default_fcl_adapter
from collision_resolver.config import DetectionConfig, PreprocessingConfig, ResolutionConfig, ServiceConfig
from collision_resolver.exceptions import BackendUnavailableError
from collision_resolver.models import MeshInstance, Pose, RequestMode, ServiceRequest
from collision_resolver.preprocess import MeshPreprocessor
from collision_resolver.detection import CollisionDetector
from collision_resolver.resolution import PoseResolver
from collision_resolver.service import CollisionResolverService
from tests.helpers import FakeCoacdAdapter, FakeFclAdapter, cube_mesh, make_test_service


def test_service_initialization_fails_fast_when_coacd_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import(name: str):
        if name == "coacd":
            raise ModuleNotFoundError(name)
        return object()

    monkeypatch.setattr("collision_resolver.backends.importlib.import_module", fail_import)

    with pytest.raises(BackendUnavailableError, match="CoACD 未安装或无法导入"):
        default_coacd_adapter()


def test_service_initialization_fails_fast_when_fcl_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import(name: str):
        if name == "fcl":
            raise ModuleNotFoundError(name)
        return object()

    monkeypatch.setattr("collision_resolver.backends.importlib.import_module", fail_import)

    with pytest.raises(BackendUnavailableError, match="python-fcl 未安装或无法导入"):
        default_fcl_adapter()


def test_service_constructor_checks_runtime_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("collision_resolver.backends.importlib.import_module", fail_import)

    with pytest.raises(BackendUnavailableError):
        CollisionResolverService(ServiceConfig())


def test_detect_request_returns_clear_for_separated_meshes() -> None:
    service = make_test_service()
    request = ServiceRequest(
        mode=RequestMode.DETECT,
        movable=MeshInstance(resource=cube_mesh("movable"), pose=Pose(translation=np.array([2.5, 0.0, 0.0]), rotation=np.eye(3))),
        obstacles=(MeshInstance(resource=cube_mesh("obstacle")),),
    )

    result = service.process(request)

    assert result.status.value == "clear"
    assert result.initial_detection is not None
    assert not result.initial_detection.is_colliding


def test_resolve_request_can_find_collision_free_pose_and_records_sdf_metadata() -> None:
    service = make_test_service()
    request = ServiceRequest(
        mode=RequestMode.RESOLVE,
        movable=MeshInstance(resource=cube_mesh("movable"), pose=Pose(translation=np.array([0.6, 0.0, 0.0]), rotation=np.eye(3))),
        obstacles=(MeshInstance(resource=cube_mesh("obstacle")),),
    )

    result = service.process(request)

    assert result.status.value == "resolved"
    assert result.optimization is not None
    assert result.optimization.candidate_pose is not None
    assert result.optimization.planar_delta is not None
    assert result.optimization.summary is not None
    assert "obstacle" in result.optimization.summary.sdf_metadata
    assert "cache_key" in result.optimization.summary.sdf_metadata["obstacle"]
    assert result.optimization.planar_delta.dx > 0.39
    assert abs(result.optimization.planar_delta.dy) < 1e-3
    assert np.allclose(result.optimization.candidate_pose.rotation, np.eye(3))
    assert result.verification is not None
    assert not result.verification.is_colliding


def test_resolve_request_ignores_old_motion_budget_and_still_finds_planar_solution() -> None:
    service = make_test_service(max_translation_delta=0.05, max_iterations=40)
    request = ServiceRequest(
        mode=RequestMode.RESOLVE,
        movable=MeshInstance(resource=cube_mesh("movable"), pose=Pose(translation=np.array([0.6, 0.0, 0.0]), rotation=np.eye(3))),
        obstacles=(MeshInstance(resource=cube_mesh("obstacle")),),
    )

    result = service.process(request)

    assert result.status.value == "resolved"
    assert result.optimization is not None
    assert result.optimization.planar_delta is not None
    assert result.optimization.planar_delta.dx > 0.39
    assert result.verification is not None
    assert not result.verification.is_colliding


def test_resolve_request_uses_tie_break_for_symmetric_minimum_translation() -> None:
    config = ServiceConfig(
        preprocessing=PreprocessingConfig(surface_sample_count=128, sdf_resolution=20),
        detection=DetectionConfig(),
        resolution=ResolutionConfig(
            learning_rate=0.08,
            max_iterations=60,
            max_translation_delta=0.2,
            safety_margin=0.01,
            patience=20,
            translation_weight=0.1,
            rotation_weight=0.0,
            collision_weight=1.0,
        ),
    )
    fake_coacd = FakeCoacdAdapter()
    fake_fcl = FakeFclAdapter()
    service = CollisionResolverService(
        config,
        preprocessor=MeshPreprocessor(config.preprocessing, coacd_adapter=fake_coacd, fcl_adapter=fake_fcl),
        detector=CollisionDetector(config.detection, fcl_adapter=fake_fcl),
        resolver=PoseResolver(config.resolution),
    )
    request = ServiceRequest(
        mode=RequestMode.RESOLVE,
        movable=MeshInstance(resource=cube_mesh("movable"), pose=Pose(translation=np.array([0.0, 0.0, 0.0]), rotation=np.eye(3))),
        obstacles=(MeshInstance(resource=cube_mesh("obstacle")),),
    )

    result = service.process(request)

    assert result.status.value == "resolved"
    assert result.optimization is not None
    assert result.optimization.candidate_pose is not None
    assert result.optimization.planar_delta is not None
    assert abs(result.optimization.planar_delta.dx) < 1e-3
    assert result.optimization.planar_delta.dy < -0.99
    assert result.verification is not None
    assert not result.verification.is_colliding
