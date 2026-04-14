from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from loguru import logger


class WatertightRepairError(RuntimeError):
    """Raised when a mesh cannot be repaired to watertight."""


def _cleanup_mesh_topology(mesh: o3d.geometry.TriangleMesh) -> None:
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.orient_triangles()


def _compute_scene_diagonal(mesh: o3d.geometry.TriangleMesh) -> float:
    extent = mesh.get_axis_aligned_bounding_box().get_extent()
    diagonal = float(np.linalg.norm(extent))
    if diagonal <= 0.0:
        raise ValueError("Mesh bbox diagonal must be positive for volumetric repair.")
    return diagonal


def _repair_with_voxel_reconstruction(
    mesh: o3d.geometry.TriangleMesh,
    voxel_pitch_ratio: float,
) -> o3d.geometry.TriangleMesh:
    diagonal = _compute_scene_diagonal(mesh)
    voxel_pitch = diagonal * voxel_pitch_ratio

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    if vertices.size == 0 or triangles.size == 0:
        raise WatertightRepairError("Input mesh is empty before volumetric repair.")

    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    voxels = tri_mesh.voxelized(pitch=voxel_pitch)
    filled = voxels.fill()
    reconstructed = filled.marching_cubes
    reconstructed.apply_transform(filled.transform)

    repaired = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(reconstructed.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(reconstructed.faces, dtype=np.int32)),
    )
    _cleanup_mesh_topology(repaired)
    logger.info(
        "Volumetric repair generated mesh: pitch={}, vertices={}, triangles={}",
        voxel_pitch,
        len(repaired.vertices),
        len(repaired.triangles),
    )
    return repaired


def ensure_watertight_mesh(
    mesh: o3d.geometry.TriangleMesh,
    mesh_label: str | Path,
) -> o3d.geometry.TriangleMesh:
    label = str(mesh_label)

    if mesh.is_watertight():
        return mesh

    logger.warning(
        "Mesh {} is not watertight. Starting deterministic volumetric repair before SDF.",
        label,
    )

    repaired = copy.deepcopy(mesh)
    _cleanup_mesh_topology(repaired)
    if repaired.is_watertight():
        repaired.compute_vertex_normals()
        logger.info(
            "Mesh {} became watertight after topology cleanup ({} vertices, {} triangles).",
            label,
            len(repaired.vertices),
            len(repaired.triangles),
        )
        return repaired

    try:
        repaired = _repair_with_voxel_reconstruction(
            repaired,
            voxel_pitch_ratio=1e-2,
        )
    except Exception as exc:
        raise WatertightRepairError(
            f"Failed to repair non-watertight mesh {label} with volumetric reconstruction."
        ) from exc

    _cleanup_mesh_topology(repaired)
    if repaired.is_empty() or len(repaired.triangles) == 0:
        raise WatertightRepairError(f"Mesh repair produced an empty mesh: {label}")
    if not repaired.is_watertight():
        raise WatertightRepairError(
            f"Mesh repair failed. Mesh is still not watertight: {label}."
        )

    repaired.compute_vertex_normals()
    logger.info(
        "Mesh {} repaired to watertight ({} vertices, {} triangles).",
        label,
        len(repaired.vertices),
        len(repaired.triangles),
    )
    return repaired
