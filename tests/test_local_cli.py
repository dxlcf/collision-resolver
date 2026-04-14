from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from collision_resolver.local_cli import build_request, euler_xyz_degrees_to_matrix, main, serialize_result
from collision_resolver.models import FinalState, MeshInstance, Pose, RequestMode, ServiceRequest, ServiceResult
from tests.helpers import cube_mesh


def test_euler_xyz_degrees_to_matrix_builds_expected_z_rotation() -> None:
    rotation = euler_xyz_degrees_to_matrix((0.0, 0.0, 90.0))
    expected = np.array(
        (
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    assert np.allclose(rotation, expected)


def test_build_request_uses_second_mesh_as_movable_by_default(monkeypatch) -> None:
    monkeypatch.setattr(
        "collision_resolver.local_cli.load_mesh_resource",
        lambda path, resource_id, unit: cube_mesh(resource_id, unit=unit),
    )
    args = argparse.Namespace(
        mesh_a="fixed.ply",
        mesh_b="movable.ply",
        translation_a=(0.0, 0.0, 0.0),
        translation_b=(1.0, 2.0, 3.0),
        euler_a=(0.0, 0.0, 0.0),
        euler_b=(0.0, 0.0, 0.0),
        unit_a="m",
        unit_b="cm",
        movable="second",
        output=None,
    )

    request = build_request(args)

    assert request.mode == RequestMode.RESOLVE
    assert request.movable.resource.resource_id == "mesh_b"
    assert request.obstacles[0].resource.resource_id == "mesh_a"
    assert np.allclose(request.movable.pose.translation, np.array([1.0, 2.0, 3.0]))
    assert request.movable.resource.unit == "cm"


def test_serialize_result_keeps_initial_and_optimization_sections() -> None:
    request = build_simple_request()
    result = ServiceResult(
        mode=RequestMode.RESOLVE,
        status=FinalState.RESOLVED,
        initial_detection=None,
        optimization=None,
        verification=None,
        diagnostics=(),
    )

    payload = serialize_result(request, result)

    assert payload["final_status"] == "resolved"
    assert payload["movable_resource_id"] == "mesh_b"
    assert payload["obstacle_resource_ids"] == ["mesh_a"]
    assert payload["initial_detection"] is None
    assert payload["optimization"] is None
    assert payload["verification"] is None


def test_serialize_result_includes_planar_delta_and_motion_constraints() -> None:
    from collision_resolver.models import OptimizationResult, OptimizationSummary, PlanarDelta, ResolutionState

    request = build_simple_request()
    result = ServiceResult(
        mode=RequestMode.RESOLVE,
        status=FinalState.RESOLVED,
        initial_detection=None,
        optimization=OptimizationResult(
            status=ResolutionState.CANDIDATE,
            candidate_pose=request.movable.pose,
            planar_delta=PlanarDelta(dx=0.1, dy=-0.2, translation_norm=np.sqrt(0.05)),
            summary=OptimizationSummary(
                iterations=3,
                stop_reason="global_minimum_with_tolerance",
                optimality_tolerance=0.05,
                motion_constraints={
                    "rotation_locked": True,
                    "z_locked": True,
                    "xy_only": True,
                    "objective": "min_planar_translation_l2",
                },
                search_metadata={"epsilon_xy": 0.05},
                sdf_metadata={},
            ),
            diagnostics=(),
        ),
        verification=None,
        diagnostics=(),
    )

    payload = serialize_result(request, result)

    assert payload["optimization"]["planar_delta"]["dx"] == 0.1
    assert payload["optimization"]["planar_delta"]["dy"] == -0.2
    assert payload["optimization"]["summary"]["optimality_tolerance"] == 0.05
    assert payload["optimization"]["summary"]["motion_constraints"]["xy_only"] is True


def test_main_calls_visualization_when_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(
        "collision_resolver.local_cli.build_request",
        lambda args: build_simple_request(),
    )
    monkeypatch.setattr("collision_resolver.local_cli.build_local_service", lambda: _StubService())

    calls: list[tuple[str, str]] = []

    def record_visualization(request, result) -> None:
        calls.append((request.movable.resource.resource_id, result.status.value))

    monkeypatch.setattr("collision_resolver.local_cli.visualize_result", record_visualization)

    output_path = tmp_path / "result.json"
    main(["a.ply", "b.ply", "--visualize", "--output", str(output_path)])

    captured = capsys.readouterr()
    assert captured.out == ""
    assert calls == [("mesh_b", "resolved")]
    assert output_path.is_file()


def test_load_mesh_resource_records_source_filename_in_metadata(tmp_path) -> None:
    trimesh = pytest.importorskip("trimesh")
    mesh_path = tmp_path / "sample.ply"
    mesh = trimesh.Trimesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        process=False,
    )
    mesh.export(mesh_path)

    from collision_resolver.local_cli import load_mesh_resource

    resource = load_mesh_resource(mesh_path, resource_id="mesh_a", unit="m")

    assert resource.metadata["source_filename"] == "sample.ply"
    assert Path(resource.metadata["source_path"]) == mesh_path.resolve()


def build_simple_request():
    return ServiceRequest(
        mode=RequestMode.RESOLVE,
        movable=MeshInstance(
            resource=cube_mesh("mesh_b"),
            pose=Pose(translation=np.array([1.0, 0.0, 0.0]), rotation=np.eye(3)),
        ),
        obstacles=(
            MeshInstance(
                resource=cube_mesh("mesh_a"),
                pose=Pose(translation=np.array([0.0, 0.0, 0.0]), rotation=np.eye(3)),
            ),
        ),
    )


class _StubService:
    def process(self, request: ServiceRequest) -> ServiceResult:
        return ServiceResult(
            mode=request.mode,
            status=FinalState.RESOLVED,
            initial_detection=None,
            optimization=None,
            verification=None,
            diagnostics=(),
        )
