from __future__ import annotations

import numpy as np

from collision_resolver.backends import CoacdAdapter, FclAdapter
from collision_resolver.config import DetectionConfig, PreprocessingConfig, ResolutionConfig, ServiceConfig
from collision_resolver.detection import CollisionDetector
from collision_resolver.models import AABB, Contact, ConvexPart, MeshResource, Pose
from collision_resolver.preprocess import MeshPreprocessor
from collision_resolver.resolution import PoseResolver
from collision_resolver.service import CollisionResolverService


def cube_mesh(resource_id: str, *, size: float = 1.0, unit: str = "m") -> MeshResource:
    half = size * 0.5
    vertices = np.array(
        [
            [-half, -half, -half],
            [half, -half, -half],
            [half, half, -half],
            [-half, half, -half],
            [-half, -half, half],
            [half, -half, half],
            [half, half, half],
            [-half, half, half],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return MeshResource(resource_id=resource_id, vertices=vertices, faces=faces, unit=unit)


class FakeCoacdAdapter(CoacdAdapter):
    def __init__(self) -> None:
        super().__init__(name="fake-coacd")

    def decompose(
        self,
        resource_id: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        max_parts: int,
    ) -> tuple[ConvexPart, ...]:
        del max_parts
        return (
            ConvexPart(
                part_id=f"{resource_id}:part-0",
                vertices=vertices,
                faces=faces,
                local_aabb=AABB.from_points(vertices),
            ),
        )


class FakeFclAdapter(FclAdapter):
    def __init__(self) -> None:
        super().__init__(name="fake-fcl")

    def collide(
        self,
        pair_id: str,
        obstacle_id: str,
        movable,
        movable_pose: Pose,
        obstacle,
        obstacle_pose: Pose,
    ) -> tuple[Contact, ...]:
        movable_world = AABB.from_points(movable_pose.apply(movable.vertices))
        obstacle_world = AABB.from_points(obstacle_pose.apply(obstacle.vertices))
        if not movable_world.overlaps(obstacle_world):
            return ()
        depths = movable_world.intersection_depths(obstacle_world)
        axis = int(np.argmin(depths))
        direction = obstacle_world.center[axis] - movable_world.center[axis]
        normal = np.zeros(3, dtype=np.float64)
        normal[axis] = -1.0 if direction >= 0.0 else 1.0
        point = (
            np.maximum(movable_world.minimum, obstacle_world.minimum)
            + np.minimum(movable_world.maximum, obstacle_world.maximum)
        ) * 0.5
        return (
            Contact(
                pair_id=pair_id,
                obstacle_id=obstacle_id,
                movable_part_id=movable.part_id,
                obstacle_part_id=obstacle.part_id,
                point=point,
                normal=normal,
                penetration_depth=float(depths[axis]),
            ),
        )


def make_test_service(max_translation_delta: float = 1.0, max_iterations: int = 160) -> CollisionResolverService:
    config = ServiceConfig(
        preprocessing=PreprocessingConfig(surface_sample_count=128, sdf_resolution=20),
        detection=DetectionConfig(),
        resolution=ResolutionConfig(
            learning_rate=0.08,
            max_iterations=max_iterations,
            max_translation_delta=max_translation_delta,
            safety_margin=0.01,
            patience=12,
            translation_weight=0.05,
            rotation_weight=0.02,
        ),
    )
    fake_coacd = FakeCoacdAdapter()
    fake_fcl = FakeFclAdapter()
    preprocessor = MeshPreprocessor(config.preprocessing, coacd_adapter=fake_coacd, fcl_adapter=fake_fcl)
    detector = CollisionDetector(config.detection, fcl_adapter=fake_fcl)
    resolver = PoseResolver(config.resolution)
    return CollisionResolverService(config, preprocessor=preprocessor, detector=detector, resolver=resolver)
