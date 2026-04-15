from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import open3d as o3d
from loguru import logger

from collision_resolver.mesh_repair import ensure_watertight_mesh

DEFAULT_CACHE_DIR = Path("data") / "sdf_cache"
SUPPORTED_MESH_SUFFIXES = {
    ".obj",
    ".ply",
    ".stl",
    ".off",
    ".gltf",
    ".glb",
}


@dataclass
class SDFVolume:
    values: np.ndarray
    min_bound: np.ndarray
    max_bound: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError("SDF volume values must be a 3D array.")
        if self.values.shape[0] < 2 or self.values.shape[1] < 2 or self.values.shape[2] < 2:
            raise ValueError("SDF volume shape must be at least 2x2x2.")
        if self.min_bound.shape != (3,) or self.max_bound.shape != (3,):
            raise ValueError("SDF bounds must be 3D vectors.")
        if not np.all(self.max_bound > self.min_bound):
            raise ValueError("SDF max_bound must be larger than min_bound on each axis.")


@dataclass
class CachePaths:
    cache_dir: Path
    mesh_path: Path
    sdf_path: Path
    surface_points_path: Path
    meta_path: Path


@dataclass
class PreprocessedMeshAsset:
    source_path: Path
    cache_key: str
    cache_paths: CachePaths
    mesh: o3d.geometry.TriangleMesh
    sdf_volume: SDFVolume
    surface_points: np.ndarray
    cache_hit: bool
    was_watertight: bool
    repair_applied: bool


SDFQueryFn = Callable[[np.ndarray], np.ndarray]


def mesh_name_to_cache_key(mesh_path: str | Path) -> str:
    stem = Path(mesh_path).stem.strip()
    if not stem:
        raise ValueError(f"Invalid mesh name for cache key: {mesh_path}")
    key = re.sub(r"[^0-9A-Za-z_.-]+", "_", stem)
    return key.strip("._-") or "mesh"


def build_cache_paths(cache_root: str | Path, mesh_path: str | Path) -> CachePaths:
    cache_key = mesh_name_to_cache_key(mesh_path)
    cache_dir = Path(cache_root) / cache_key
    return CachePaths(
        cache_dir=cache_dir,
        mesh_path=cache_dir / "mesh_watertight.ply",
        sdf_path=cache_dir / "sdf_volume.npz",
        surface_points_path=cache_dir / "surface_points.npy",
        meta_path=cache_dir / "meta.json",
    )


def read_mesh(mesh_path: str | Path) -> o3d.geometry.TriangleMesh:
    path = Path(mesh_path)
    if not path.is_file():
        raise FileNotFoundError(f"Mesh file does not exist: {path}")

    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise ValueError(f"Failed to load mesh or mesh is empty: {path}")
    if len(mesh.triangles) == 0:
        raise ValueError(f"Mesh has no triangles: {path}")

    logger.info(
        "Loaded mesh {}: {} vertices, {} triangles",
        path,
        len(mesh.vertices),
        len(mesh.triangles),
    )
    return mesh


def build_raycast_scene(mesh: o3d.geometry.TriangleMesh) -> o3d.t.geometry.RaycastingScene:
    mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_tensor)
    return scene


def query_sdf_from_scene(
    scene: o3d.t.geometry.RaycastingScene,
    points: np.ndarray,
    *,
    batch_size: int = 200_000,
) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points must be shaped as (N, 3).")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if len(points) == 0:
        return np.empty((0,), dtype=np.float32)

    sdf_values = np.empty((len(points),), dtype=np.float32)
    for start in range(0, len(points), batch_size):
        end = min(start + batch_size, len(points))
        points_tensor = o3d.core.Tensor(points[start:end], dtype=o3d.core.Dtype.Float32)
        sdf_values[start:end] = scene.compute_signed_distance(points_tensor).numpy()
    return sdf_values


def build_sdf_volume(
    mesh: o3d.geometry.TriangleMesh,
    *,
    voxel_size_ratio: float = 1e-2,
    padding_ratio: float = 5e-2,
    max_grid_dim: int = 160,
    batch_size: int = 200_000,
) -> SDFVolume:
    if voxel_size_ratio <= 0.0:
        raise ValueError("voxel_size_ratio must be positive.")
    if padding_ratio < 0.0:
        raise ValueError("padding_ratio must be non-negative.")
    if max_grid_dim < 2:
        raise ValueError("max_grid_dim must be >= 2.")

    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    diagonal = float(np.linalg.norm(extent))
    if diagonal <= 0.0:
        raise ValueError("Mesh bbox diagonal must be positive for SDF volume build.")

    voxel_size = diagonal * voxel_size_ratio
    padding = max(diagonal * padding_ratio, voxel_size)
    min_bound = bbox.min_bound - padding
    max_bound = bbox.max_bound + padding

    grid_shape = np.ceil((max_bound - min_bound) / voxel_size).astype(np.int32) + 1
    max_dim = int(np.max(grid_shape))
    if max_dim > max_grid_dim:
        scale = max_dim / max_grid_dim
        voxel_size *= scale
        grid_shape = np.ceil((max_bound - min_bound) / voxel_size).astype(np.int32) + 1

    grid_shape = np.maximum(grid_shape, 2)
    grid_shape_tuple = (int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2]))

    logger.info(
        "Building SDF volume: shape={}, voxel_size={}",
        list(grid_shape_tuple),
        voxel_size,
    )

    axes = [
        np.linspace(min_bound[i], max_bound[i], num=grid_shape_tuple[i], dtype=np.float32)
        for i in range(3)
    ]
    xx, yy, zz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    grid_points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

    scene = build_raycast_scene(mesh)
    sdf_flat = query_sdf_from_scene(scene, grid_points, batch_size=batch_size)
    sdf_values = sdf_flat.reshape(grid_shape_tuple).astype(np.float32)

    return SDFVolume(
        values=sdf_values,
        min_bound=min_bound.astype(np.float64),
        max_bound=max_bound.astype(np.float64),
    )


def save_sdf_volume(volume: SDFVolume, path: str | Path) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path_obj,
        values=volume.values.astype(np.float32),
        min_bound=volume.min_bound.astype(np.float64),
        max_bound=volume.max_bound.astype(np.float64),
    )


def load_sdf_volume(path: str | Path) -> SDFVolume:
    path_obj = Path(path)
    data = np.load(path_obj)
    return SDFVolume(
        values=np.asarray(data["values"], dtype=np.float32),
        min_bound=np.asarray(data["min_bound"], dtype=np.float64),
        max_bound=np.asarray(data["max_bound"], dtype=np.float64),
    )


def build_surface_points(
    mesh: o3d.geometry.TriangleMesh,
    *,
    point_count: int,
) -> np.ndarray:
    if point_count <= 0:
        raise ValueError("surface point_count must be a positive integer.")

    sampled_pcd = mesh.sample_points_uniformly(number_of_points=point_count)
    points = np.asarray(sampled_pcd.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Sampled surface points must be shaped as (N, 3).")
    return points


def save_surface_points(points: np.ndarray, path: str | Path) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    np.save(path_obj, np.asarray(points, dtype=np.float32))


def load_surface_points(path: str | Path) -> np.ndarray:
    path_obj = Path(path)
    points = np.asarray(np.load(path_obj), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Cached surface points must be shaped as (N, 3).")
    return points


def _trilinear_interpolate(values: np.ndarray, index_points: np.ndarray) -> np.ndarray:
    shape = np.array(values.shape, dtype=np.int32)
    lower = np.floor(index_points).astype(np.int32)
    lower = np.clip(lower, 0, shape - 2)
    upper = lower + 1
    frac = index_points - lower

    x0 = lower[:, 0]
    y0 = lower[:, 1]
    z0 = lower[:, 2]
    x1 = upper[:, 0]
    y1 = upper[:, 1]
    z1 = upper[:, 2]
    dx = frac[:, 0]
    dy = frac[:, 1]
    dz = frac[:, 2]

    c000 = values[x0, y0, z0]
    c100 = values[x1, y0, z0]
    c010 = values[x0, y1, z0]
    c110 = values[x1, y1, z0]
    c001 = values[x0, y0, z1]
    c101 = values[x1, y0, z1]
    c011 = values[x0, y1, z1]
    c111 = values[x1, y1, z1]

    c00 = c000 * (1.0 - dx) + c100 * dx
    c10 = c010 * (1.0 - dx) + c110 * dx
    c01 = c001 * (1.0 - dx) + c101 * dx
    c11 = c011 * (1.0 - dx) + c111 * dx
    c0 = c00 * (1.0 - dy) + c10 * dy
    c1 = c01 * (1.0 - dy) + c11 * dy
    return c0 * (1.0 - dz) + c1 * dz


def query_sdf_from_volume(
    volume: SDFVolume,
    points: np.ndarray,
    *,
    translation: np.ndarray | None = None,
    fallback_scene: o3d.t.geometry.RaycastingScene | None = None,
    batch_size: int = 200_000,
) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points must be shaped as (N, 3).")
    if len(points) == 0:
        return np.empty((0,), dtype=np.float32)

    shift = np.zeros(3, dtype=np.float64)
    if translation is not None:
        shift = np.asarray(translation, dtype=np.float64)

    local_points = points.astype(np.float64) - shift[None, :]
    bounds_extent = volume.max_bound - volume.min_bound
    grid_extent = np.array(volume.values.shape, dtype=np.float64) - 1.0
    index_points = ((local_points - volume.min_bound[None, :]) / bounds_extent[None, :]) * grid_extent[None, :]

    inside_mask = np.logical_and(index_points >= 0.0, index_points <= grid_extent[None, :]).all(axis=1)
    sdf_values = np.empty((len(points),), dtype=np.float32)

    if np.any(inside_mask):
        sdf_values[inside_mask] = _trilinear_interpolate(volume.values, index_points[inside_mask])

    if np.any(~inside_mask):
        if fallback_scene is None:
            sdf_values[~inside_mask] = np.inf
        else:
            sdf_values[~inside_mask] = query_sdf_from_scene(
                fallback_scene,
                local_points[~inside_mask],
                batch_size=batch_size,
            )
    return sdf_values


def create_cached_sdf_query(
    volume: SDFVolume,
    mesh_in_cache_frame: o3d.geometry.TriangleMesh,
    *,
    translation: np.ndarray | None = None,
) -> SDFQueryFn:
    scene = build_raycast_scene(mesh_in_cache_frame)
    shift = np.zeros(3, dtype=np.float64)
    if translation is not None:
        shift = np.asarray(translation, dtype=np.float64).copy()

    def _query(points: np.ndarray) -> np.ndarray:
        return query_sdf_from_volume(
            volume,
            points,
            translation=shift,
            fallback_scene=scene,
        )

    return _query


def _cache_files_exist(cache_paths: CachePaths) -> bool:
    return (
        cache_paths.mesh_path.is_file()
        and cache_paths.sdf_path.is_file()
        and cache_paths.surface_points_path.is_file()
        and cache_paths.meta_path.is_file()
    )


def _load_cached_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise ValueError(f"Cached mesh is empty: {path}")
    mesh.compute_vertex_normals()
    return mesh


def _load_cache_meta(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_cache_meta(path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def preprocess_mesh_with_cache(
    mesh_path: str | Path,
    *,
    cache_root: str | Path = DEFAULT_CACHE_DIR,
    prefer_cache: bool = True,
    force_rebuild: bool = False,
    voxel_size_ratio: float = 1e-2,
    padding_ratio: float = 5e-2,
    max_grid_dim: int = 160,
    surface_point_count: int = 20_000,
) -> PreprocessedMeshAsset:
    if surface_point_count <= 0:
        raise ValueError("surface_point_count must be a positive integer.")

    source_path = Path(mesh_path)
    cache_key = mesh_name_to_cache_key(source_path)
    cache_paths = build_cache_paths(cache_root, source_path)

    if prefer_cache and not force_rebuild and _cache_files_exist(cache_paths):
        try:
            meta = _load_cache_meta(cache_paths.meta_path)
            cached_surface_point_count = int(meta.get("surface_point_count", -1))
            if cached_surface_point_count != surface_point_count:
                raise ValueError(
                    "Cached surface_point_count does not match request "
                    f"({cached_surface_point_count} != {surface_point_count})."
                )

            mesh = _load_cached_mesh(cache_paths.mesh_path)
            sdf_volume = load_sdf_volume(cache_paths.sdf_path)
            surface_points = load_surface_points(cache_paths.surface_points_path)
            logger.info("Preprocess cache hit: {} -> {}", source_path.name, cache_paths.cache_dir)
            return PreprocessedMeshAsset(
                source_path=source_path,
                cache_key=cache_key,
                cache_paths=cache_paths,
                mesh=mesh,
                sdf_volume=sdf_volume,
                surface_points=surface_points,
                cache_hit=True,
                was_watertight=bool(meta.get("was_watertight", True)),
                repair_applied=bool(meta.get("repair_applied", False)),
            )
        except Exception as exc:
            logger.warning(
                "Cache load failed for {} ({}). Rebuilding preprocess cache.",
                source_path,
                exc,
            )

    mesh = read_mesh(source_path)
    was_watertight = mesh.is_watertight()
    mesh = ensure_watertight_mesh(mesh, mesh_label=source_path)
    mesh.compute_vertex_normals()

    sdf_volume = build_sdf_volume(
        mesh,
        voxel_size_ratio=voxel_size_ratio,
        padding_ratio=padding_ratio,
        max_grid_dim=max_grid_dim,
    )
    surface_points = build_surface_points(mesh, point_count=surface_point_count)

    cache_paths.cache_dir.mkdir(parents=True, exist_ok=True)
    success = o3d.io.write_triangle_mesh(str(cache_paths.mesh_path), mesh)
    if not success:
        raise RuntimeError(f"Failed to write cached watertight mesh: {cache_paths.mesh_path}")
    save_sdf_volume(sdf_volume, cache_paths.sdf_path)
    save_surface_points(surface_points, cache_paths.surface_points_path)
    _save_cache_meta(
        cache_paths.meta_path,
        {
            "source_path": str(source_path),
            "source_name": source_path.name,
            "source_stem": source_path.stem,
            "cache_key": cache_key,
            "was_watertight": was_watertight,
            "repair_applied": not was_watertight,
            "sdf_grid_shape": list(sdf_volume.values.shape),
            "voxel_size_ratio": voxel_size_ratio,
            "padding_ratio": padding_ratio,
            "max_grid_dim": max_grid_dim,
            "surface_point_count": int(surface_point_count),
        },
    )

    logger.info("Preprocess cache generated: {} -> {}", source_path.name, cache_paths.cache_dir)
    return PreprocessedMeshAsset(
        source_path=source_path,
        cache_key=cache_key,
        cache_paths=cache_paths,
        mesh=mesh,
        sdf_volume=sdf_volume,
        surface_points=surface_points,
        cache_hit=False,
        was_watertight=was_watertight,
        repair_applied=not was_watertight,
    )


def iter_mesh_files(input_dir: str | Path, *, recursive: bool = True) -> list[Path]:
    root = Path(input_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Input mesh directory does not exist: {root}")

    iterator = root.rglob("*") if recursive else root.glob("*")
    files = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_MESH_SUFFIXES
    ]
    return sorted(files)
