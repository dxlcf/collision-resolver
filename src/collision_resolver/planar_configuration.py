from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from shapely import BufferJoinStyle
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Point, Polygon, box
from shapely.ops import nearest_points, unary_union

from collision_resolver.models import AABB

_AXIS_TOLERANCE = 1e-8
_POLYGON_TOLERANCE = 1e-9


@dataclass(frozen=True)
class PlanarSearchResult:
    point: np.ndarray
    search_radius: float
    forbidden_region_count: int


def solve_minimum_planar_translation(
    *,
    convex_pair_meshes: tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...],
    movable_world_aabb: AABB,
    obstacle_world_aabbs: tuple[AABB, ...],
    epsilon_xy: float,
) -> PlanarSearchResult | None:
    upper_bound = _planar_clear_upper_bound(movable_world_aabb, obstacle_world_aabbs) + max(epsilon_xy, 1e-6)
    search_box = box(-upper_bound, -upper_bound, upper_bound, upper_bound)

    pair_polygons: list[Polygon] = []
    for movable_vertices, movable_faces, obstacle_vertices, obstacle_faces in convex_pair_meshes:
        polygon = _build_pair_forbidden_region(
            movable_vertices=movable_vertices,
            movable_faces=movable_faces,
            obstacle_vertices=obstacle_vertices,
            obstacle_faces=obstacle_faces,
            search_radius=upper_bound,
        )
        if polygon is not None and not polygon.is_empty and polygon.area > 0.0:
            pair_polygons.append(polygon)

    if not pair_polygons:
        return PlanarSearchResult(point=np.zeros(2, dtype=np.float64), search_radius=upper_bound, forbidden_region_count=0)

    forbidden = unary_union(pair_polygons)
    clearance = max(epsilon_xy * 0.01, _POLYGON_TOLERANCE)
    expanded_forbidden = forbidden.buffer(clearance, join_style=BufferJoinStyle.mitre)
    free_space = search_box.difference(expanded_forbidden)
    if free_space.is_empty:
        return None

    origin = Point(0.0, 0.0)
    nearest = nearest_points(origin, free_space)[1]
    point = np.array([float(nearest.x), float(nearest.y)], dtype=np.float64)
    point = _apply_tie_break(expanded_forbidden, point)
    return PlanarSearchResult(point=point, search_radius=upper_bound, forbidden_region_count=len(pair_polygons))


def _build_pair_forbidden_region(
    *,
    movable_vertices: np.ndarray,
    movable_faces: np.ndarray,
    obstacle_vertices: np.ndarray,
    obstacle_faces: np.ndarray,
    search_radius: float,
) -> Polygon | None:
    polygon = [
        np.array([-search_radius, -search_radius], dtype=np.float64),
        np.array([search_radius, -search_radius], dtype=np.float64),
        np.array([search_radius, search_radius], dtype=np.float64),
        np.array([-search_radius, search_radius], dtype=np.float64),
    ]

    for axis in _candidate_axes(movable_vertices, movable_faces, obstacle_vertices, obstacle_faces):
        movable_projection = movable_vertices @ axis
        obstacle_projection = obstacle_vertices @ axis
        lower = float(np.min(obstacle_projection) - np.max(movable_projection))
        upper = float(np.max(obstacle_projection) - np.min(movable_projection))
        if lower > upper + _AXIS_TOLERANCE:
            return None

        normal_xy = axis[:2]
        normal_xy_norm = float(np.linalg.norm(normal_xy))
        if normal_xy_norm <= _AXIS_TOLERANCE:
            if lower <= 0.0 <= upper:
                continue
            return None

        polygon = _clip_polygon_halfplane(polygon, normal_xy, upper, keep_less_equal=True)
        if len(polygon) < 3:
            return None
        polygon = _clip_polygon_halfplane(polygon, normal_xy, lower, keep_less_equal=False)
        if len(polygon) < 3:
            return None

    region = Polygon([tuple(vertex.tolist()) for vertex in polygon])
    if region.is_empty:
        return None
    if not region.is_valid:
        region = region.buffer(0)
    if region.is_empty or region.area <= 0.0:
        return None
    return region


def _clip_polygon_halfplane(
    polygon: list[np.ndarray],
    normal_xy: np.ndarray,
    bound: float,
    *,
    keep_less_equal: bool,
) -> list[np.ndarray]:
    if not polygon:
        return []

    def is_inside(point: np.ndarray) -> bool:
        value = float(np.dot(normal_xy, point))
        if keep_less_equal:
            return value <= bound + _POLYGON_TOLERANCE
        return value >= bound - _POLYGON_TOLERANCE

    clipped: list[np.ndarray] = []
    for start, end in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        start_inside = is_inside(start)
        end_inside = is_inside(end)

        if start_inside and end_inside:
            clipped.append(end.copy())
            continue
        if start_inside and not end_inside:
            intersection = _segment_halfplane_intersection(start, end, normal_xy, bound)
            if intersection is not None:
                clipped.append(intersection)
            continue
        if not start_inside and end_inside:
            intersection = _segment_halfplane_intersection(start, end, normal_xy, bound)
            if intersection is not None:
                clipped.append(intersection)
            clipped.append(end.copy())

    return _deduplicate_polygon_vertices(clipped)


def _segment_halfplane_intersection(
    start: np.ndarray,
    end: np.ndarray,
    normal_xy: np.ndarray,
    bound: float,
) -> np.ndarray | None:
    direction = end - start
    denominator = float(np.dot(normal_xy, direction))
    if abs(denominator) <= _POLYGON_TOLERANCE:
        return None
    t = (bound - float(np.dot(normal_xy, start))) / denominator
    t = min(max(t, 0.0), 1.0)
    return start + t * direction


def _deduplicate_polygon_vertices(vertices: list[np.ndarray]) -> list[np.ndarray]:
    if not vertices:
        return []

    deduplicated = [vertices[0]]
    for vertex in vertices[1:]:
        if np.linalg.norm(vertex - deduplicated[-1]) > _POLYGON_TOLERANCE:
            deduplicated.append(vertex)
    if len(deduplicated) > 1 and np.linalg.norm(deduplicated[0] - deduplicated[-1]) <= _POLYGON_TOLERANCE:
        deduplicated.pop()
    return deduplicated


def _candidate_axes(
    movable_vertices: np.ndarray,
    movable_faces: np.ndarray,
    obstacle_vertices: np.ndarray,
    obstacle_faces: np.ndarray,
) -> tuple[np.ndarray, ...]:
    movable_face_normals, movable_edges = _polyhedron_axes(movable_vertices, movable_faces)
    obstacle_face_normals, obstacle_edges = _polyhedron_axes(obstacle_vertices, obstacle_faces)
    axes = list(movable_face_normals)
    axes.extend(obstacle_face_normals)
    for movable_edge in movable_edges:
        for obstacle_edge in obstacle_edges:
            cross = np.cross(movable_edge, obstacle_edge)
            if np.linalg.norm(cross) > _AXIS_TOLERANCE:
                axes.append(cross)
    return _unique_unit_vectors(axes)


def _polyhedron_axes(vertices: np.ndarray, faces: np.ndarray) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    normals: list[np.ndarray] = []
    edges: list[np.ndarray] = []
    triangles = vertices[faces]
    for triangle in triangles:
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        if np.linalg.norm(normal) > _AXIS_TOLERANCE:
            normals.append(normal)
        edges.extend((triangle[1] - triangle[0], triangle[2] - triangle[1], triangle[0] - triangle[2]))
    return _unique_unit_vectors(normals), _unique_unit_vectors(edges)


def _unique_unit_vectors(vectors: Iterable[np.ndarray]) -> tuple[np.ndarray, ...]:
    unique: dict[tuple[float, float, float], np.ndarray] = {}
    for vector in vectors:
        norm = float(np.linalg.norm(vector))
        if norm <= _AXIS_TOLERANCE:
            continue
        unit = np.asarray(vector, dtype=np.float64) / norm
        canonical = _canonicalize_axis(unit)
        key = tuple(np.round(canonical, 8))
        unique.setdefault(key, canonical)
    return tuple(unique.values())


def _canonicalize_axis(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    for component in axis:
        if abs(component) <= _AXIS_TOLERANCE:
            continue
        if component < 0.0:
            return -axis
        return axis
    return axis


def _planar_clear_upper_bound(movable_world_aabb: AABB, obstacle_world_aabbs: tuple[AABB, ...]) -> float:
    margin = 1e-6
    positive_x = max((obstacle.maximum[0] - movable_world_aabb.minimum[0] + margin) for obstacle in obstacle_world_aabbs)
    negative_x = max((movable_world_aabb.maximum[0] - obstacle.minimum[0] + margin) for obstacle in obstacle_world_aabbs)
    positive_y = max((obstacle.maximum[1] - movable_world_aabb.minimum[1] + margin) for obstacle in obstacle_world_aabbs)
    negative_y = max((movable_world_aabb.maximum[1] - obstacle.minimum[1] + margin) for obstacle in obstacle_world_aabbs)
    return float(max(min(positive_x, negative_x, positive_y, negative_y), margin))


def _apply_tie_break(forbidden_geometry, nearest_point: np.ndarray) -> np.ndarray:
    boundary_segments = _boundary_segments(forbidden_geometry)
    if not boundary_segments:
        return nearest_point

    best_norm = float(np.linalg.norm(nearest_point))
    candidates = [nearest_point]
    for start, end in boundary_segments:
        candidates.extend(_segment_candidates(start, end, best_norm + _POLYGON_TOLERANCE))

    filtered = [
        candidate
        for candidate in candidates
        if abs(np.linalg.norm(candidate) - best_norm) <= 1e-6
    ]
    if not filtered:
        filtered = candidates

    return min(
        filtered,
        key=lambda item: (
            abs(float(item[0])),
            abs(float(item[1])),
            float(item[0]),
            float(item[1]),
        ),
    )


def _boundary_segments(geometry) -> list[tuple[np.ndarray, np.ndarray]]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        segments = _ring_segments(geometry.exterior)
        for ring in geometry.interiors:
            segments.extend(_ring_segments(ring))
        return segments
    if isinstance(geometry, MultiPolygon):
        segments: list[tuple[np.ndarray, np.ndarray]] = []
        for polygon in geometry.geoms:
            segments.extend(_boundary_segments(polygon))
        return segments
    if isinstance(geometry, GeometryCollection):
        segments: list[tuple[np.ndarray, np.ndarray]] = []
        for item in geometry.geoms:
            segments.extend(_boundary_segments(item))
        return segments
    if isinstance(geometry, MultiLineString):
        segments: list[tuple[np.ndarray, np.ndarray]] = []
        for line in geometry.geoms:
            segments.extend(_boundary_segments(line))
        return segments
    if isinstance(geometry, LineString):
        coordinates = [np.asarray(point, dtype=np.float64) for point in geometry.coords]
        return [(coordinates[index], coordinates[index + 1]) for index in range(len(coordinates) - 1)]
    return []


def _ring_segments(ring) -> list[tuple[np.ndarray, np.ndarray]]:
    coordinates = [np.asarray(point, dtype=np.float64) for point in ring.coords]
    return [(coordinates[index], coordinates[index + 1]) for index in range(len(coordinates) - 1)]


def _segment_candidates(start: np.ndarray, end: np.ndarray, radius_limit: float) -> list[np.ndarray]:
    candidates = [start, end]
    direction = end - start
    length_squared = float(np.dot(direction, direction))
    if length_squared > _POLYGON_TOLERANCE:
        projection_t = -float(np.dot(start, direction)) / length_squared
        if 0.0 <= projection_t <= 1.0:
            candidates.append(start + projection_t * direction)

        if abs(direction[0]) > _POLYGON_TOLERANCE:
            zero_x_t = -float(start[0]) / float(direction[0])
            if 0.0 <= zero_x_t <= 1.0:
                candidates.append(start + zero_x_t * direction)

        if abs(direction[1]) > _POLYGON_TOLERANCE:
            zero_y_t = -float(start[1]) / float(direction[1])
            if 0.0 <= zero_y_t <= 1.0:
                candidates.append(start + zero_y_t * direction)

    return [
        np.asarray(candidate, dtype=np.float64)
        for candidate in candidates
        if float(np.linalg.norm(candidate)) <= radius_limit + _POLYGON_TOLERANCE
    ]
