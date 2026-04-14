from __future__ import annotations

import hashlib
import json

import numpy as np

from collision_resolver.backends import CoacdAdapter, FclAdapter, default_coacd_adapter, default_fcl_adapter
from collision_resolver.cache import ConvexDecompositionCache, PreprocessCache
from collision_resolver.config import PreprocessingConfig
from collision_resolver.exceptions import ValidationError
from collision_resolver.math_utils import (
    has_consistent_winding,
    mesh_signed_distance,
    sample_surface_points,
    signed_mesh_volume,
    triangle_areas,
)
from collision_resolver.models import (
    AABB,
    CacheKey,
    ConvexPart,
    MeshMetadata,
    MeshResource,
    PreprocessedMesh,
    SdfField,
    SurfaceSamples,
)

UNIT_TO_METERS = {"m": 1.0, "cm": 0.01, "mm": 0.001}


class MeshPreprocessor:
    def __init__(
        self,
        config: PreprocessingConfig,
        *,
        cache: PreprocessCache | None = None,
        convex_cache: ConvexDecompositionCache | None = None,
        coacd_adapter: CoacdAdapter | None = None,
        fcl_adapter: FclAdapter | None = None,
    ) -> None:
        self.config = config
        self.cache = cache or PreprocessCache()
        self.convex_cache = convex_cache
        self.coacd_adapter = coacd_adapter or default_coacd_adapter()
        self.fcl_adapter = fcl_adapter or default_fcl_adapter()

    def prepare(
        self,
        resource: MeshResource,
        *,
        require_sdf: bool | None = None,
        require_samples: bool | None = None,
    ) -> tuple[PreprocessedMesh, bool]:
        want_sdf = self.config.generate_sdf_by_default if require_sdf is None else require_sdf
        want_samples = self.config.generate_surface_samples_by_default if require_samples is None else require_samples
        cache_key = self._cache_key(resource, want_sdf=want_sdf, want_samples=want_samples)
        cached = self.cache.get(str(cache_key))
        if cached is not None:
            return cached, True

        normalized_vertices, faces, metadata = self._validate_and_normalize(resource, require_sdf=want_sdf)
        convex_parts = self._load_or_decompose_convex_parts(resource, normalized_vertices, faces)
        fcl_geometries = tuple(self.fcl_adapter.create_geometry(part) for part in convex_parts)
        sdf_field = self._build_sdf(normalized_vertices, faces) if want_sdf else None
        surface_samples = self._build_surface_samples(normalized_vertices, faces) if want_samples else None
        artifact = PreprocessedMesh(
            resource_id=resource.resource_id,
            cache_key=cache_key,
            normalized_vertices=normalized_vertices,
            faces=faces,
            mesh_aabb=AABB.from_points(normalized_vertices),
            convex_parts=convex_parts,
            fcl_geometries=fcl_geometries,
            sdf_field=sdf_field,
            surface_samples=surface_samples,
            metadata=metadata,
        )
        self.cache.put(str(cache_key), artifact)
        return artifact, False

    def _load_or_decompose_convex_parts(
        self,
        resource: MeshResource,
        normalized_vertices: np.ndarray,
        faces: np.ndarray,
    ) -> tuple[ConvexPart, ...]:
        if self.convex_cache is not None:
            cached = self.convex_cache.load(resource, normalized_vertices, faces, self.config)
            if cached is not None:
                return cached

        convex_parts = self.coacd_adapter.decompose(
            resource.resource_id,
            normalized_vertices,
            faces,
            self.config.coacd_max_convex_parts,
        )
        if self.convex_cache is not None:
            self.convex_cache.save(resource, normalized_vertices, faces, self.config, convex_parts)
        return convex_parts

    def _cache_key(self, resource: MeshResource, *, want_sdf: bool, want_samples: bool) -> CacheKey:
        relevant_config = {
            "target_unit": self.config.target_unit,
            "coacd_max_convex_parts": self.config.coacd_max_convex_parts,
            "sdf_resolution": self.config.sdf_resolution if want_sdf else None,
            "sdf_padding_ratio": self.config.sdf_padding_ratio if want_sdf else None,
            "sdf_sign_method": self.config.sdf_sign_method if want_sdf else None,
            "surface_sample_count": self.config.surface_sample_count if want_samples else None,
            "random_seed": self.config.random_seed if want_samples else None,
        }
        digest = hashlib.sha256()
        digest.update(resource.content_digest().encode("utf-8"))
        digest.update(json.dumps(relevant_config, sort_keys=True).encode("utf-8"))
        return CacheKey(resource_id=resource.resource_id, digest=digest.hexdigest())

    def _validate_and_normalize(
        self,
        resource: MeshResource,
        *,
        require_sdf: bool,
    ) -> tuple[np.ndarray, np.ndarray, MeshMetadata]:
        vertices = np.asarray(resource.vertices, dtype=np.float64)
        faces = np.asarray(resource.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValidationError("preprocess", "INVALID_VERTICES", "mesh 顶点必须是 N x 3 数组。")
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            raise ValidationError("preprocess", "INVALID_FACES", "首版仅支持三角面片。")
        if not np.isfinite(vertices).all():
            raise ValidationError("preprocess", "NONFINITE_VERTICES", "mesh 顶点包含非有限数值。")
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise ValidationError("preprocess", "FACE_INDEX_OUT_OF_RANGE", "mesh 面片索引越界。")

        unit_scale = UNIT_TO_METERS.get(resource.unit)
        target_scale = UNIT_TO_METERS.get(self.config.target_unit)
        if unit_scale is None or target_scale is None:
            raise ValidationError("preprocess", "UNSUPPORTED_UNIT", f"不支持的单位: {resource.unit}")

        normalized = vertices @ resource.axis_transform.T
        normalized = normalized * (unit_scale / target_scale)
        areas = triangle_areas(normalized, faces)
        if np.any(areas <= self.config.min_triangle_area):
            raise ValidationError("preprocess", "DEGENERATE_TRIANGLE", "mesh 存在退化三角形。")
        if require_sdf:
            if not has_consistent_winding(faces):
                raise ValidationError("preprocess", "INCONSISTENT_WINDING", "mesh 不满足闭合且法向一致的高精度 SDF 构建前提。")
            if abs(signed_mesh_volume(normalized, faces)) <= self.config.min_abs_volume:
                raise ValidationError("preprocess", "NON_VOLUME_MESH", "mesh 体积过小或不满足高精度 SDF 构建前提。")

        metadata = MeshMetadata(
            vertex_count=len(vertices),
            face_count=len(faces),
            centroid=np.mean(normalized, axis=0),
            extents=np.max(normalized, axis=0) - np.min(normalized, axis=0),
            normalized_unit=self.config.target_unit,
            used_axis_transform=np.asarray(resource.axis_transform, dtype=np.float64),
        )
        return normalized, faces, metadata

    def _build_sdf(self, vertices: np.ndarray, faces: np.ndarray) -> SdfField:
        mesh_aabb = AABB.from_points(vertices)
        extent = np.maximum(mesh_aabb.extents, 1e-6)
        padding = float(np.max(extent) * self.config.sdf_padding_ratio + 1e-6)
        minimum = mesh_aabb.minimum - padding
        maximum = mesh_aabb.maximum + padding
        padded_aabb = AABB(minimum=minimum, maximum=maximum)

        resolution = self.config.sdf_resolution
        axes = [np.linspace(minimum[index], maximum[index], resolution) for index in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        distances = mesh_signed_distance(
            grid,
            vertices,
            faces,
            sign_method=self.config.sdf_sign_method,
        )
        values = distances.reshape(resolution, resolution, resolution)
        spacing = np.array([(maximum[index] - minimum[index]) / max(resolution - 1, 1) for index in range(3)], dtype=np.float64)
        return SdfField(
            origin=minimum,
            spacing=spacing,
            values=values,
            shape=values.shape,
            mesh_aabb=mesh_aabb,
            padded_aabb=padded_aabb,
            padding=padding,
            sign_method=self.config.sdf_sign_method,
        )

    def _build_surface_samples(self, vertices: np.ndarray, faces: np.ndarray) -> SurfaceSamples:
        rng = np.random.default_rng(self.config.random_seed)
        points, normals = sample_surface_points(vertices, faces, self.config.surface_sample_count, rng)
        return SurfaceSamples(points=points, normals=normals)
