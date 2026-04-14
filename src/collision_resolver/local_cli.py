from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from collision_resolver.cache import FilesystemConvexDecompositionCache
from collision_resolver.config import ServiceConfig
from collision_resolver.exceptions import BackendUnavailableError
from collision_resolver.models import (
    CollisionDetectionResult,
    Contact,
    FinalState,
    MeshInstance,
    MeshResource,
    OptimizationResult,
    Pose,
    RequestMode,
    ServiceRequest,
    ServiceResult,
    StageDiagnostic,
)
from collision_resolver.preprocess import MeshPreprocessor
from collision_resolver.service import CollisionResolverService
from collision_resolver.visualization import build_visualization_inputs, visualize_detection_result

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CONVEX_CACHE_DIR = PROJECT_ROOT / "cache"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/run_local_case.py",
        description=(
            "本地调用脚本：输入两个 PLY 文件路径和对应初始姿态，默认第一个模型固定，第二个模型可动。"
            "脚本输出初始碰撞检测结果、位姿优化结果和复检结果。"
        ),
    )
    parser.add_argument("mesh_a", help="第一个 PLY 文件路径；默认作为固定障碍物")
    parser.add_argument("mesh_b", help="第二个 PLY 文件路径；默认作为可动物体")
    parser.add_argument(
        "--translation-a",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help="第一个模型的平移，单位与网格 unit 一致，默认 0 0 0",
    )
    parser.add_argument(
        "--translation-b",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help="第二个模型的平移，单位与网格 unit 一致，默认 0 0 0",
    )
    parser.add_argument(
        "--euler-a",
        nargs=3,
        type=float,
        metavar=("RX", "RY", "RZ"),
        default=(0.0, 0.0, 0.0),
        help="第一个模型的 XYZ 欧拉角，单位为度，默认 0 0 0",
    )
    parser.add_argument(
        "--euler-b",
        nargs=3,
        type=float,
        metavar=("RX", "RY", "RZ"),
        default=(0.0, 0.0, 0.0),
        help="第二个模型的 XYZ 欧拉角，单位为度，默认 0 0 0",
    )
    parser.add_argument(
        "--unit-a",
        choices=("m", "cm", "mm"),
        default="m",
        help="第一个模型的长度单位，默认 m",
    )
    parser.add_argument(
        "--unit-b",
        choices=("m", "cm", "mm"),
        default="m",
        help="第二个模型的长度单位，默认 m",
    )
    parser.add_argument(
        "--movable",
        choices=("first", "second"),
        default="second",
        help="指定哪个模型可动，默认 second",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="可选，输出 JSON 文件路径；未提供时打印到标准输出",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="可选，使用 open3d 先后可视化初始碰撞区域与优化后的结果",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    request = build_request(args)
    try:
        service = build_local_service()
    except BackendUnavailableError as exc:
        raise SystemExit(f"{exc.message} [code={exc.code}]") from exc

    result = service.process(request)
    payload = serialize_result(request, result)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)

    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)

    if args.visualize:
        visualize_result(request, result)


def build_local_service() -> CollisionResolverService:
    config = ServiceConfig()
    preprocessor = MeshPreprocessor(
        config.preprocessing,
        convex_cache=FilesystemConvexDecompositionCache(LOCAL_CONVEX_CACHE_DIR),
    )
    return CollisionResolverService(config, preprocessor=preprocessor)


def build_request(args: argparse.Namespace) -> ServiceRequest:
    instance_a = MeshInstance(
        resource=load_mesh_resource(Path(args.mesh_a), resource_id="mesh_a", unit=args.unit_a),
        pose=Pose(
            translation=np.asarray(args.translation_a, dtype=np.float64),
            rotation=euler_xyz_degrees_to_matrix(args.euler_a),
        ),
    )
    instance_b = MeshInstance(
        resource=load_mesh_resource(Path(args.mesh_b), resource_id="mesh_b", unit=args.unit_b),
        pose=Pose(
            translation=np.asarray(args.translation_b, dtype=np.float64),
            rotation=euler_xyz_degrees_to_matrix(args.euler_b),
        ),
    )

    if args.movable == "first":
        movable = instance_a
        obstacles = (instance_b,)
    else:
        movable = instance_b
        obstacles = (instance_a,)

    return ServiceRequest(mode=RequestMode.RESOLVE, movable=movable, obstacles=obstacles)


def load_mesh_resource(path: Path, *, resource_id: str, unit: str) -> MeshResource:
    if not path.is_file():
        raise SystemExit(f"PLY 文件不存在: {path}")
    if path.suffix.lower() != ".ply":
        raise SystemExit(f"当前本地脚本仅支持 .ply 文件: {path}")

    try:
        import trimesh
    except ModuleNotFoundError as exc:
        raise SystemExit("缺少 trimesh 依赖，请先执行 `uv sync --dev --extra mesh`。") from exc

    mesh = trimesh.load(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise SystemExit(f"文件未加载为单个三角网格，请检查输入是否为单个 PLY 网格: {path}")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise SystemExit(f"顶点数组必须是 (N, 3): {path}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise SystemExit(f"当前仅支持三角面网格，faces 必须是 (M, 3): {path}")

    return MeshResource(
        resource_id=resource_id,
        vertices=vertices,
        faces=faces,
        unit=unit,
        metadata={
            "source_filename": path.name,
            "source_path": str(path.resolve()),
        },
    )


def euler_xyz_degrees_to_matrix(values: list[float] | tuple[float, float, float]) -> np.ndarray:
    rx, ry, rz = np.deg2rad(np.asarray(values, dtype=np.float64))
    cx, sx = float(np.cos(rx)), float(np.sin(rx))
    cy, sy = float(np.cos(ry)), float(np.sin(ry))
    cz, sz = float(np.cos(rz)), float(np.sin(rz))

    rotation_x = np.array(((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)), dtype=np.float64)
    rotation_y = np.array(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)), dtype=np.float64)
    rotation_z = np.array(((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64)
    return rotation_z @ rotation_y @ rotation_x


def serialize_result(request: ServiceRequest, result: ServiceResult) -> dict[str, Any]:
    movable_id = request.movable.resource.resource_id
    obstacle_ids = [obstacle.resource.resource_id for obstacle in request.obstacles]
    return {
        "mode": result.mode.value,
        "final_status": result.status.value,
        "movable_resource_id": movable_id,
        "obstacle_resource_ids": obstacle_ids,
        "initial_detection": serialize_detection_result(result.initial_detection),
        "optimization": serialize_optimization_result(result.optimization),
        "verification": serialize_detection_result(result.verification),
        "diagnostics": [serialize_diagnostic(item) for item in result.diagnostics],
    }


def visualize_result(request: ServiceRequest, result: ServiceResult) -> None:
    initial_movable, initial_obstacles = build_visualization_inputs(request.movable, request.obstacles)
    visualize_detection_result(
        title="collision-resolver: 初始碰撞检测",
        movable=initial_movable,
        obstacles=initial_obstacles,
        detection=result.initial_detection,
    )

    final_pose = request.movable.pose
    if result.optimization is not None and result.optimization.candidate_pose is not None:
        final_pose = result.optimization.candidate_pose
    final_movable, final_obstacles = build_visualization_inputs(request.movable, request.obstacles, pose=final_pose)
    visualize_detection_result(
        title=f"collision-resolver: 优化后结果 ({result.status.value})",
        movable=final_movable,
        obstacles=final_obstacles,
        detection=result.verification or result.initial_detection,
    )


def serialize_detection_result(result: CollisionDetectionResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "status": result.status.value,
        "is_colliding": result.is_colliding,
        "max_penetration_depth": result.max_penetration_depth,
        "candidate_pair_count": result.candidate_pair_count,
        "checked_pair_count": result.checked_pair_count,
        "representative_normal": to_list_or_none(result.representative_normal),
        "representative_contact_point": to_list_or_none(result.representative_contact_point),
        "contacts": [serialize_contact(contact) for contact in result.contacts],
    }


def serialize_optimization_result(result: OptimizationResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "status": result.status.value,
        "candidate_pose": None if result.candidate_pose is None else serialize_pose(result.candidate_pose),
        "planar_delta": None
        if result.planar_delta is None
        else {
            "dx": result.planar_delta.dx,
            "dy": result.planar_delta.dy,
            "translation_norm": result.planar_delta.translation_norm,
        },
        "summary": None
        if result.summary is None
        else {
            "iterations": result.summary.iterations,
            "stop_reason": result.summary.stop_reason,
            "optimality_tolerance": result.summary.optimality_tolerance,
            "motion_constraints": result.summary.motion_constraints,
            "search_metadata": result.summary.search_metadata,
            "sdf_metadata": result.summary.sdf_metadata,
        },
        "diagnostics": [serialize_diagnostic(item) for item in result.diagnostics],
    }


def serialize_pose(pose: Pose) -> dict[str, Any]:
    return {
        "translation": pose.translation.tolist(),
        "rotation": pose.rotation.tolist(),
    }


def serialize_contact(contact: Contact) -> dict[str, Any]:
    return {
        "pair_id": contact.pair_id,
        "obstacle_id": contact.obstacle_id,
        "movable_part_id": contact.movable_part_id,
        "obstacle_part_id": contact.obstacle_part_id,
        "point": contact.point.tolist(),
        "normal": contact.normal.tolist(),
        "penetration_depth": contact.penetration_depth,
    }


def serialize_diagnostic(diagnostic: StageDiagnostic) -> dict[str, Any]:
    return {
        "stage": diagnostic.stage.value,
        "code": diagnostic.code,
        "message": diagnostic.message,
        "details": diagnostic.details,
    }


def to_list_or_none(values: np.ndarray | None) -> list[float] | None:
    return None if values is None else values.tolist()
