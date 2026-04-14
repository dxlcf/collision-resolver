from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from collision_resolver.math_utils import (
    FloatArray,
    IntArray,
    as_float_array,
    as_int_array,
    identity_matrix,
    inverse_transform_points_numpy,
    transform_points_numpy,
)


class Stage(str, Enum):
    PREPROCESS = "preprocess"
    DETECT = "detect"
    RESOLVE = "resolve"
    VERIFY = "verify"


class DetectionState(str, Enum):
    CLEAR = "clear"
    COLLIDING = "colliding"
    FAILED = "failed"


class ResolutionState(str, Enum):
    SKIPPED = "skipped"
    CANDIDATE = "candidate"
    UNRESOLVED = "unresolved"
    FAILED = "failed"


class FinalState(str, Enum):
    CLEAR = "clear"
    COLLIDING = "colliding"
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    FAILED = "failed"


class RequestMode(str, Enum):
    DETECT = "detect"
    RESOLVE = "resolve"


@dataclass(frozen=True)
class StageDiagnostic:
    stage: Stage
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CacheKey:
    resource_id: str
    digest: str

    def __str__(self) -> str:
        return f"{self.resource_id}:{self.digest}"


@dataclass(frozen=True)
class Pose:
    translation: FloatArray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    rotation: FloatArray = field(default_factory=identity_matrix)

    @classmethod
    def identity(cls) -> "Pose":
        return cls()

    def apply(self, points: FloatArray) -> FloatArray:
        return transform_points_numpy(points, self.rotation, self.translation)

    def inverse_apply(self, points: FloatArray) -> FloatArray:
        return inverse_transform_points_numpy(points, self.rotation, self.translation)


@dataclass(frozen=True)
class AABB:
    minimum: FloatArray
    maximum: FloatArray

    @classmethod
    def from_points(cls, points: FloatArray) -> "AABB":
        return cls(np.min(points, axis=0), np.max(points, axis=0))

    @property
    def center(self) -> FloatArray:
        return (self.minimum + self.maximum) * 0.5

    @property
    def extents(self) -> FloatArray:
        return self.maximum - self.minimum

    def overlaps(self, other: "AABB", padding: float = 0.0) -> bool:
        return bool(
            np.all(self.minimum - padding <= other.maximum)
            and np.all(other.minimum - padding <= self.maximum)
        )

    def intersection_depths(self, other: "AABB") -> FloatArray:
        return np.minimum(self.maximum, other.maximum) - np.maximum(self.minimum, other.minimum)


@dataclass(frozen=True)
class MeshResource:
    resource_id: str
    vertices: FloatArray
    faces: IntArray
    unit: str = "m"
    axis_transform: FloatArray = field(default_factory=identity_matrix)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "vertices", as_float_array(self.vertices))
        object.__setattr__(self, "faces", as_int_array(self.faces))
        object.__setattr__(self, "axis_transform", as_float_array(self.axis_transform, shape=(3, 3)))

    def content_digest(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.vertices.tobytes())
        hasher.update(self.faces.tobytes())
        hasher.update(self.unit.encode("utf-8"))
        hasher.update(self.axis_transform.tobytes())
        return hasher.hexdigest()


@dataclass(frozen=True)
class MeshInstance:
    resource: MeshResource
    pose: Pose = field(default_factory=Pose.identity)


@dataclass(frozen=True)
class MeshMetadata:
    vertex_count: int
    face_count: int
    centroid: FloatArray
    extents: FloatArray
    normalized_unit: str
    used_axis_transform: FloatArray


@dataclass(frozen=True)
class ConvexPart:
    part_id: str
    vertices: FloatArray
    faces: IntArray
    local_aabb: AABB


@dataclass(frozen=True)
class FclGeometryData:
    part_id: str
    vertices: FloatArray
    faces: IntArray
    local_aabb: AABB
    backend: str


@dataclass(frozen=True)
class SdfField:
    origin: FloatArray
    spacing: FloatArray
    values: FloatArray
    shape: tuple[int, int, int]
    mesh_aabb: AABB
    padded_aabb: AABB
    padding: float
    sign_method: str

    def query_numpy(self, points: FloatArray) -> FloatArray:
        coordinates = (points - self.origin) / self.spacing
        max_index = np.array(self.shape, dtype=np.float64) - 1.0
        clipped = np.clip(coordinates, 0.0, max_index)
        lower = np.floor(clipped).astype(np.int64)
        upper = np.minimum(lower + 1, np.asarray(self.shape, dtype=np.int64) - 1)
        weights = clipped - lower

        x0, y0, z0 = lower[:, 0], lower[:, 1], lower[:, 2]
        x1, y1, z1 = upper[:, 0], upper[:, 1], upper[:, 2]
        wx, wy, wz = weights[:, 0], weights[:, 1], weights[:, 2]

        c000 = self.values[x0, y0, z0]
        c001 = self.values[x0, y0, z1]
        c010 = self.values[x0, y1, z0]
        c011 = self.values[x0, y1, z1]
        c100 = self.values[x1, y0, z0]
        c101 = self.values[x1, y0, z1]
        c110 = self.values[x1, y1, z0]
        c111 = self.values[x1, y1, z1]

        c00 = c000 * (1.0 - wx) + c100 * wx
        c01 = c001 * (1.0 - wx) + c101 * wx
        c10 = c010 * (1.0 - wx) + c110 * wx
        c11 = c011 * (1.0 - wx) + c111 * wx
        c0 = c00 * (1.0 - wy) + c10 * wy
        c1 = c01 * (1.0 - wy) + c11 * wy
        return c0 * (1.0 - wz) + c1 * wz

    def query_torch(self, points: torch.Tensor) -> torch.Tensor:
        origin = torch.as_tensor(self.origin, dtype=points.dtype, device=points.device)
        spacing = torch.as_tensor(self.spacing, dtype=points.dtype, device=points.device)
        coords = (points - origin) / spacing
        shape = torch.as_tensor(self.shape, dtype=points.dtype, device=points.device)
        denom = torch.clamp(shape - 1.0, min=1.0)
        normalized = (coords / denom) * 2.0 - 1.0
        grid = torch.stack((normalized[:, 0], normalized[:, 1], normalized[:, 2]), dim=-1).view(1, -1, 1, 1, 3)
        volume = torch.as_tensor(self.values, dtype=points.dtype, device=points.device).permute(2, 1, 0).unsqueeze(0).unsqueeze(0)
        sampled = F.grid_sample(volume, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return sampled.view(-1)

    def summary(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "origin": self.origin.tolist(),
            "spacing": self.spacing.tolist(),
            "padding": self.padding,
            "sign_method": self.sign_method,
        }


@dataclass(frozen=True)
class SurfaceSamples:
    points: FloatArray
    normals: FloatArray


@dataclass(frozen=True)
class PreprocessedMesh:
    resource_id: str
    cache_key: CacheKey
    normalized_vertices: FloatArray
    faces: IntArray
    mesh_aabb: AABB
    convex_parts: tuple[ConvexPart, ...]
    fcl_geometries: tuple[FclGeometryData, ...]
    sdf_field: SdfField | None
    surface_samples: SurfaceSamples | None
    metadata: MeshMetadata
    diagnostics: tuple[StageDiagnostic, ...] = ()


@dataclass(frozen=True)
class Contact:
    pair_id: str
    obstacle_id: str
    movable_part_id: str
    obstacle_part_id: str
    point: FloatArray
    normal: FloatArray
    penetration_depth: float


@dataclass(frozen=True)
class PairCollisionResult:
    pair_id: str
    obstacle_id: str
    status: DetectionState
    is_colliding: bool
    contacts: tuple[Contact, ...]
    max_penetration_depth: float
    representative_normal: FloatArray | None
    representative_contact_point: FloatArray | None
    candidate_pair_count: int
    checked_pair_count: int
    diagnostics: tuple[StageDiagnostic, ...] = ()


@dataclass(frozen=True)
class CollisionDetectionResult:
    status: DetectionState
    is_colliding: bool
    contacts: tuple[Contact, ...]
    max_penetration_depth: float
    representative_normal: FloatArray | None
    representative_contact_point: FloatArray | None
    candidate_pair_count: int
    checked_pair_count: int
    pair_results: tuple[PairCollisionResult, ...]
    diagnostics: tuple[StageDiagnostic, ...] = ()


@dataclass(frozen=True)
class OptimizationSummary:
    iterations: int
    stop_reason: str
    optimality_tolerance: float
    motion_constraints: dict[str, Any]
    search_metadata: dict[str, Any]
    sdf_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanarDelta:
    dx: float
    dy: float
    translation_norm: float


@dataclass(frozen=True)
class OptimizationResult:
    status: ResolutionState
    candidate_pose: Pose | None
    planar_delta: PlanarDelta | None
    summary: OptimizationSummary | None
    diagnostics: tuple[StageDiagnostic, ...] = ()


@dataclass(frozen=True)
class ServiceRequest:
    mode: RequestMode
    movable: MeshInstance
    obstacles: tuple[MeshInstance, ...]


@dataclass(frozen=True)
class ServiceResult:
    mode: RequestMode
    status: FinalState
    initial_detection: CollisionDetectionResult | None
    optimization: OptimizationResult | None
    verification: CollisionDetectionResult | None
    diagnostics: tuple[StageDiagnostic, ...] = ()
