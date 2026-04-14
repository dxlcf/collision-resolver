from __future__ import annotations

from typing import Iterable

import numpy as np

from collision_resolver.models import CollisionDetectionResult, Contact, MeshInstance, Pose


def visualize_detection_result(
    *,
    title: str,
    movable: MeshInstance,
    obstacles: tuple[MeshInstance, ...],
    detection: CollisionDetectionResult | None,
) -> None:
    o3d = _import_open3d()
    geometries = []
    movable_color = (0.85, 0.35, 0.15) if detection is not None and detection.is_colliding else (0.20, 0.65, 0.35)
    geometries.append(_build_mesh_geometry(o3d, movable, movable_color))

    obstacle_palette = (
        (0.55, 0.60, 0.68),
        (0.45, 0.52, 0.60),
        (0.60, 0.55, 0.50),
    )
    for index, obstacle in enumerate(obstacles):
        geometries.append(_build_mesh_geometry(o3d, obstacle, obstacle_palette[index % len(obstacle_palette)]))

    if detection is not None and detection.contacts:
        geometries.extend(_build_contact_geometries(o3d, detection.contacts))

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=_scene_scale(movable, obstacles) * 0.25)
    geometries.append(frame)
    o3d.visualization.draw_geometries(geometries, window_name=title, mesh_show_back_face=True)


def build_visualization_inputs(
    movable: MeshInstance,
    obstacles: tuple[MeshInstance, ...],
    *,
    pose: Pose | None = None,
) -> tuple[MeshInstance, tuple[MeshInstance, ...]]:
    rendered_movable = MeshInstance(resource=movable.resource, pose=pose or movable.pose)
    rendered_obstacles = tuple(MeshInstance(resource=item.resource, pose=item.pose) for item in obstacles)
    return rendered_movable, rendered_obstacles


def _import_open3d():
    try:
        import open3d as o3d
    except ModuleNotFoundError as exc:
        raise SystemExit("缺少 open3d 依赖，请先执行 `uv sync --dev --extra mesh --extra visualization`。") from exc
    return o3d


def _build_mesh_geometry(o3d, instance: MeshInstance, color: tuple[float, float, float]):
    vertices = instance.pose.apply(instance.resource.vertices)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(instance.resource.faces)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh


def _build_contact_geometries(o3d, contacts: Iterable[Contact]) -> list:
    contact_items = list(contacts)
    if not contact_items:
        return []

    max_depth = max(contact.penetration_depth for contact in contact_items)
    marker_radius = max(max_depth * 0.18, 1e-3)
    line_points: list[np.ndarray] = []
    line_segments: list[list[int]] = []
    line_colors: list[list[float]] = []
    geometries = []

    for contact in contact_items:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color((0.95, 0.10, 0.10))
        sphere.translate(contact.point)
        geometries.append(sphere)

        start = np.asarray(contact.point, dtype=np.float64)
        end = start + np.asarray(contact.normal, dtype=np.float64) * max(contact.penetration_depth, marker_radius * 2.0)
        start_index = len(line_points)
        line_points.extend((start, end))
        line_segments.append([start_index, start_index + 1])
        line_colors.append([1.0, 0.2, 0.2])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(line_points, dtype=np.float64))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(line_segments, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.asarray(line_colors, dtype=np.float64))
    geometries.append(line_set)
    return geometries


def _scene_scale(movable: MeshInstance, obstacles: tuple[MeshInstance, ...]) -> float:
    all_vertices = [movable.pose.apply(movable.resource.vertices)]
    all_vertices.extend(item.pose.apply(item.resource.vertices) for item in obstacles)
    stacked = np.vstack(all_vertices)
    extents = np.max(stacked, axis=0) - np.min(stacked, axis=0)
    return float(max(np.max(extents), 1e-3))
