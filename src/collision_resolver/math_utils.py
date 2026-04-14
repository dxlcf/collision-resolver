from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import numpy.typing as npt
import torch

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def as_float_array(values: Iterable[float] | FloatArray, *, shape: tuple[int, ...] | None = None) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if shape is not None:
        array = np.reshape(array, shape)
    return array


def as_int_array(values: Iterable[int] | IntArray, *, shape: tuple[int, ...] | None = None) -> IntArray:
    array = np.asarray(values, dtype=np.int64)
    if shape is not None:
        array = np.reshape(array, shape)
    return array


def identity_matrix() -> FloatArray:
    return np.eye(3, dtype=np.float64)


def triangle_areas(vertices: FloatArray, faces: IntArray) -> FloatArray:
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return np.linalg.norm(cross, axis=1) * 0.5


def triangle_normals(vertices: FloatArray, faces: IntArray) -> FloatArray:
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    return cross / safe_norms


def signed_mesh_volume(vertices: FloatArray, faces: IntArray) -> float:
    triangles = vertices[faces]
    return float(np.sum(np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2]))) / 6.0)


def has_consistent_winding(faces: IntArray) -> bool:
    edge_orientations: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face in faces:
        directed_edges = ((int(face[0]), int(face[1])), (int(face[1]), int(face[2])), (int(face[2]), int(face[0])))
        for start, end in directed_edges:
            key = (start, end) if start < end else (end, start)
            edge_orientations.setdefault(key, []).append((start, end))
    for entries in edge_orientations.values():
        if len(entries) != 2:
            return False
        first, second = entries
        if first[0] != second[1] or first[1] != second[0]:
            return False
    return True


def sample_surface_points(
    vertices: FloatArray,
    faces: IntArray,
    count: int,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray]:
    areas = triangle_areas(vertices, faces)
    probabilities = areas / areas.sum()
    triangle_indices = rng.choice(len(faces), size=count, p=probabilities)
    triangles = vertices[faces[triangle_indices]]
    normals = triangle_normals(vertices, faces)[triangle_indices]

    r1 = np.sqrt(rng.random(count))
    r2 = rng.random(count)
    barycentric = np.stack((1.0 - r1, r1 * (1.0 - r2), r1 * r2), axis=1)
    points = np.sum(triangles * barycentric[:, :, None], axis=1)
    return points, normals


def mesh_signed_distance(
    points: FloatArray,
    vertices: FloatArray,
    faces: IntArray,
    *,
    sign_method: str,
    chunk_size: int = 2048,
) -> FloatArray:
    triangles = vertices[faces]
    unsigned = np.empty(len(points), dtype=np.float64)
    inside = np.empty(len(points), dtype=bool)
    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        chunk = points[start:end]
        unsigned[start:end] = _unsigned_distance_to_triangles(chunk, triangles)
        if sign_method != "ray_casting":
            raise ValueError(f"不支持的 SDF 符号判定方式: {sign_method}")
        inside[start:end] = _points_inside_mesh(chunk, triangles)
    return np.where(inside, -unsigned, unsigned)


def axis_angle_to_matrix_numpy(axis_angle: FloatArray) -> FloatArray:
    theta = float(np.linalg.norm(axis_angle))
    if theta < 1e-12:
        return identity_matrix()
    axis = axis_angle / theta
    x, y, z = axis
    skew = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=np.float64)
    return identity_matrix() + math.sin(theta) * skew + (1.0 - math.cos(theta)) * (skew @ skew)


def axis_angle_to_matrix_torch(axis_angle: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(axis_angle)
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    if torch.isclose(theta, torch.zeros_like(theta)):
        return eye
    axis = axis_angle / theta
    x, y, z = axis.unbind()
    skew = torch.stack(
        (
            torch.stack((torch.zeros_like(x), -z, y)),
            torch.stack((z, torch.zeros_like(x), -x)),
            torch.stack((-y, x, torch.zeros_like(x))),
        )
    )
    return eye + torch.sin(theta) * skew + (1.0 - torch.cos(theta)) * (skew @ skew)


def transform_points_numpy(points: FloatArray, rotation: FloatArray, translation: FloatArray) -> FloatArray:
    return points @ rotation.T + translation


def inverse_transform_points_numpy(points: FloatArray, rotation: FloatArray, translation: FloatArray) -> FloatArray:
    return (points - translation) @ rotation


def transform_points_torch(points: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    return points @ rotation.T + translation


def inverse_transform_points_torch(points: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    return (points - translation) @ rotation


def _unsigned_distance_to_triangles(points: FloatArray, triangles: FloatArray) -> FloatArray:
    p = points[:, None, :]
    a = triangles[None, :, 0, :]
    b = triangles[None, :, 1, :]
    c = triangles[None, :, 2, :]
    ba = b - a
    cb = c - b
    ac = a - c
    pa = p - a
    pb = p - b
    pc = p - c
    normal = np.cross(ba, ac)

    edge0 = np.sum(np.cross(ba, normal) * pa, axis=2)
    edge1 = np.sum(np.cross(cb, normal) * pb, axis=2)
    edge2 = np.sum(np.cross(ac, normal) * pc, axis=2)
    outside = (np.sign(edge0) + np.sign(edge1) + np.sign(edge2)) < 2.0

    seg0 = _segment_distance_squared(pa, ba)
    seg1 = _segment_distance_squared(pb, cb)
    seg2 = _segment_distance_squared(pc, ac)
    edge_distance = np.sqrt(np.minimum(np.minimum(seg0, seg1), seg2))

    normal_norm = np.linalg.norm(normal, axis=2)
    safe_normal_norm = np.where(normal_norm > 0.0, normal_norm, 1.0)
    plane_distance = np.abs(np.sum(normal * pa, axis=2)) / safe_normal_norm
    distances = np.where(outside, edge_distance, plane_distance)
    return np.min(distances, axis=1)


def _segment_distance_squared(point_vectors: FloatArray, edges: FloatArray) -> FloatArray:
    edge_dot = np.sum(edges * edges, axis=2)
    safe_edge_dot = np.where(edge_dot > 0.0, edge_dot, 1.0)
    projection = np.clip(np.sum(point_vectors * edges, axis=2) / safe_edge_dot, 0.0, 1.0)
    delta = point_vectors - edges * projection[:, :, None]
    return np.sum(delta * delta, axis=2)


def _points_inside_mesh(points: FloatArray, triangles: FloatArray) -> npt.NDArray[np.bool_]:
    direction = np.array([1.0, 0.317, 0.213], dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    epsilon = 1e-9
    v0 = triangles[:, 0, :]
    edge1 = triangles[:, 1, :] - v0
    edge2 = triangles[:, 2, :] - v0
    pvec = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    det = np.einsum("ij,ij->i", edge1, pvec)
    valid = np.abs(det) > epsilon
    inv_det = np.zeros_like(det)
    inv_det[valid] = 1.0 / det[valid]

    tvec = points[:, None, :] - v0[None, :, :]
    u = np.einsum("ptj,tj->pt", tvec, pvec) * inv_det[None, :]
    qvec = np.cross(tvec, edge1[None, :, :])
    v = np.einsum("j,ptj->pt", direction, qvec) * inv_det[None, :]
    t = np.einsum("ptj,tj->pt", qvec, edge2) * inv_det[None, :]
    hits = (
        valid[None, :]
        & (u >= -epsilon)
        & (v >= -epsilon)
        & (u + v <= 1.0 + epsilon)
        & (t > epsilon)
    )
    return (np.count_nonzero(hits, axis=1) % 2) == 1
