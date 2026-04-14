from __future__ import annotations

import importlib

import numpy as np

from collision_resolver.exceptions import BackendUnavailableError
from collision_resolver.math_utils import FloatArray, IntArray
from collision_resolver.models import AABB, Contact, ConvexPart, FclGeometryData, Pose


class CoacdAdapter:
    def __init__(self, name: str) -> None:
        self.name = name

    def decompose(
        self,
        resource_id: str,
        vertices: FloatArray,
        faces: IntArray,
        max_parts: int,
    ) -> tuple[ConvexPart, ...]:
        raise NotImplementedError


class NativeCoacdAdapter(CoacdAdapter):
    def __init__(self, coacd_module: object) -> None:
        super().__init__(name="coacd")
        self._coacd = coacd_module

    def decompose(
        self,
        resource_id: str,
        vertices: FloatArray,
        faces: IntArray,
        max_parts: int,
    ) -> tuple[ConvexPart, ...]:
        mesh = self._coacd.Mesh(vertices.astype(np.float64), faces.astype(np.int32))
        result = self._coacd.run_coacd(mesh, max_convex_hull=max_parts)
        parts: list[ConvexPart] = []
        for index, (part_vertices, part_faces) in enumerate(result):
            part_vertices_array = np.asarray(part_vertices, dtype=np.float64)
            part_faces_array = np.asarray(part_faces, dtype=np.int64)
            parts.append(
                ConvexPart(
                    part_id=f"{resource_id}:part-{index}",
                    vertices=part_vertices_array,
                    faces=part_faces_array,
                    local_aabb=AABB.from_points(part_vertices_array),
                )
            )
        if not parts:
            raise RuntimeError("CoACD 未返回任何凸分解结果。")
        return tuple(parts)


class FclAdapter:
    def __init__(self, name: str) -> None:
        self.name = name

    def create_geometry(self, part: ConvexPart) -> FclGeometryData:
        return FclGeometryData(
            part_id=part.part_id,
            vertices=part.vertices,
            faces=part.faces,
            local_aabb=part.local_aabb,
            backend=self.name,
        )

    def collide(
        self,
        pair_id: str,
        obstacle_id: str,
        movable: FclGeometryData,
        movable_pose: Pose,
        obstacle: FclGeometryData,
        obstacle_pose: Pose,
    ) -> tuple[Contact, ...]:
        raise NotImplementedError


class NativeFclAdapter(FclAdapter):
    def __init__(self, fcl_module: object) -> None:
        super().__init__(name="python-fcl")
        self._fcl = fcl_module

    def collide(
        self,
        pair_id: str,
        obstacle_id: str,
        movable: FclGeometryData,
        movable_pose: Pose,
        obstacle: FclGeometryData,
        obstacle_pose: Pose,
    ) -> tuple[Contact, ...]:
        movable_model = _build_fcl_bvh(self._fcl, movable.vertices, movable.faces)
        obstacle_model = _build_fcl_bvh(self._fcl, obstacle.vertices, obstacle.faces)
        movable_transform = self._fcl.Transform(movable_pose.rotation, movable_pose.translation)
        obstacle_transform = self._fcl.Transform(obstacle_pose.rotation, obstacle_pose.translation)
        movable_obj = self._fcl.CollisionObject(movable_model, movable_transform)
        obstacle_obj = self._fcl.CollisionObject(obstacle_model, obstacle_transform)
        request = self._fcl.CollisionRequest(enable_contact=True, num_max_contacts=32)
        result = self._fcl.CollisionResult()
        self._fcl.collide(movable_obj, obstacle_obj, request, result)
        contacts: list[Contact] = []
        for item in result.contacts:
            contacts.append(
                Contact(
                    pair_id=pair_id,
                    obstacle_id=obstacle_id,
                    movable_part_id=movable.part_id,
                    obstacle_part_id=obstacle.part_id,
                    point=np.asarray(item.pos, dtype=np.float64),
                    normal=np.asarray(item.normal, dtype=np.float64),
                    penetration_depth=float(item.penetration_depth),
                )
            )
        return tuple(contacts)


def ensure_runtime_dependencies() -> None:
    default_coacd_adapter()
    default_fcl_adapter()


def default_coacd_adapter() -> CoacdAdapter:
    try:
        module = importlib.import_module("coacd")
    except Exception as exc:  # pragma: no cover
        raise BackendUnavailableError("init", "COACD_UNAVAILABLE", "CoACD 未安装或无法导入。") from exc
    return NativeCoacdAdapter(module)


def default_fcl_adapter() -> FclAdapter:
    try:
        module = importlib.import_module("fcl")
    except Exception as exc:  # pragma: no cover
        raise BackendUnavailableError("init", "FCL_UNAVAILABLE", "python-fcl 未安装或无法导入。") from exc
    return NativeFclAdapter(module)


def _build_fcl_bvh(fcl_module: object, vertices: FloatArray, faces: IntArray):  # pragma: no cover
    model = fcl_module.BVHModel()
    model.beginModel(len(vertices), len(faces))
    model.addSubModel(vertices.astype(np.float64), faces.astype(np.int32))
    model.endModel()
    return model
