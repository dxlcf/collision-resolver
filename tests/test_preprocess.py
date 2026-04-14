from __future__ import annotations

import numpy as np

from collision_resolver.cache import FilesystemConvexDecompositionCache
from collision_resolver.config import PreprocessingConfig
from collision_resolver.exceptions import ValidationError
from collision_resolver.models import AABB, ConvexPart
from collision_resolver.preprocess import MeshPreprocessor
from tests.helpers import FakeCoacdAdapter, FakeFclAdapter, cube_mesh


def make_preprocessor(**overrides: object) -> MeshPreprocessor:
    config = PreprocessingConfig(**{"surface_sample_count": 64, "sdf_resolution": 10, **overrides})
    return MeshPreprocessor(config, coacd_adapter=FakeCoacdAdapter(), fcl_adapter=FakeFclAdapter())


class CountingCoacdAdapter(FakeCoacdAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def decompose(
        self,
        resource_id: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        max_parts: int,
    ) -> tuple[ConvexPart, ...]:
        self.calls += 1
        return (
            ConvexPart(
                part_id=f"{resource_id}:part-0",
                vertices=vertices,
                faces=faces,
                local_aabb=AABB.from_points(vertices),
            ),
        )


def test_preprocess_normalizes_units_and_builds_cache() -> None:
    preprocessor = make_preprocessor()
    mesh = cube_mesh("cube-cm", size=100.0, unit="cm")

    artifact, cache_hit = preprocessor.prepare(mesh, require_sdf=True, require_samples=True)
    cached_artifact, cached_hit = preprocessor.prepare(mesh, require_sdf=True, require_samples=True)

    assert not cache_hit
    assert cached_hit
    assert artifact.cache_key == cached_artifact.cache_key
    assert np.allclose(artifact.mesh_aabb.extents, np.array([1.0, 1.0, 1.0]))
    assert artifact.sdf_field is not None
    assert artifact.sdf_field.shape == (10, 10, 10)
    assert artifact.surface_samples is not None
    assert artifact.surface_samples.points.shape == (64, 3)


def test_preprocess_sdf_contains_exact_grid_metadata_and_signed_values() -> None:
    preprocessor = make_preprocessor(sdf_resolution=12, sdf_padding_ratio=0.2)
    mesh = cube_mesh("cube")

    artifact, _ = preprocessor.prepare(mesh, require_sdf=True, require_samples=False)

    assert artifact.sdf_field is not None
    assert artifact.sdf_field.shape == (12, 12, 12)
    assert artifact.sdf_field.sign_method == "ray_casting"
    center_value = artifact.sdf_field.query_numpy(np.array([[0.0, 0.0, 0.0]], dtype=np.float64))[0]
    outside_value = artifact.sdf_field.query_numpy(np.array([[1.5, 0.0, 0.0]], dtype=np.float64))[0]
    assert center_value < 0.0
    assert outside_value > 0.0


def test_preprocess_cache_key_changes_when_sdf_parameters_change() -> None:
    mesh = cube_mesh("cube")
    artifact_resolution, _ = make_preprocessor(sdf_resolution=10).prepare(mesh, require_sdf=True, require_samples=False)
    artifact_padding, _ = make_preprocessor(sdf_resolution=12).prepare(mesh, require_sdf=True, require_samples=False)
    artifact_ratio, _ = make_preprocessor(sdf_padding_ratio=0.2).prepare(mesh, require_sdf=True, require_samples=False)

    assert artifact_resolution.cache_key != artifact_padding.cache_key
    assert artifact_resolution.cache_key != artifact_ratio.cache_key


def test_preprocess_reuses_persisted_convex_decomposition_between_preprocessor_instances(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    mesh = cube_mesh("cube")
    mesh = type(mesh)(
        resource_id=mesh.resource_id,
        vertices=mesh.vertices,
        faces=mesh.faces,
        unit=mesh.unit,
        axis_transform=mesh.axis_transform,
        metadata={"source_filename": "cube.ply"},
    )
    config = PreprocessingConfig(surface_sample_count=64, sdf_resolution=10)
    convex_cache = FilesystemConvexDecompositionCache(cache_dir)

    first_coacd = CountingCoacdAdapter()
    first = MeshPreprocessor(
        config,
        convex_cache=convex_cache,
        coacd_adapter=first_coacd,
        fcl_adapter=FakeFclAdapter(),
    )
    second_coacd = CountingCoacdAdapter()
    second = MeshPreprocessor(
        config,
        convex_cache=convex_cache,
        coacd_adapter=second_coacd,
        fcl_adapter=FakeFclAdapter(),
    )

    first_artifact, _ = first.prepare(mesh, require_sdf=False, require_samples=False)
    second_artifact, _ = second.prepare(mesh, require_sdf=False, require_samples=False)

    assert first_coacd.calls == 1
    assert second_coacd.calls == 0
    assert (cache_dir / "cube.ply.npz").is_file()
    assert len(first_artifact.convex_parts) == len(second_artifact.convex_parts) == 1
    assert np.allclose(first_artifact.convex_parts[0].vertices, second_artifact.convex_parts[0].vertices)


def test_preprocess_invalidates_persisted_convex_cache_when_mesh_content_changes(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    config = PreprocessingConfig(surface_sample_count=64, sdf_resolution=10)
    convex_cache = FilesystemConvexDecompositionCache(cache_dir)
    coacd = CountingCoacdAdapter()
    preprocessor = MeshPreprocessor(
        config,
        convex_cache=convex_cache,
        coacd_adapter=coacd,
        fcl_adapter=FakeFclAdapter(),
    )

    base = cube_mesh("cube")
    first_mesh = type(base)(
        resource_id=base.resource_id,
        vertices=base.vertices,
        faces=base.faces,
        unit=base.unit,
        axis_transform=base.axis_transform,
        metadata={"source_filename": "shared-name.ply"},
    )
    second_mesh = type(base)(
        resource_id=base.resource_id,
        vertices=base.vertices * 1.5,
        faces=base.faces,
        unit=base.unit,
        axis_transform=base.axis_transform,
        metadata={"source_filename": "shared-name.ply"},
    )

    preprocessor.prepare(first_mesh, require_sdf=False, require_samples=False)
    second_preprocessor = MeshPreprocessor(
        config,
        convex_cache=convex_cache,
        coacd_adapter=coacd,
        fcl_adapter=FakeFclAdapter(),
    )
    second_preprocessor.prepare(second_mesh, require_sdf=False, require_samples=False)

    assert coacd.calls == 2


def test_preprocess_rejects_inconsistent_winding_for_exact_sdf() -> None:
    preprocessor = make_preprocessor()
    mesh = cube_mesh("bad-cube")
    flipped_faces = mesh.faces.copy()
    flipped_faces[0] = flipped_faces[0][::-1]
    invalid_mesh = cube_mesh("bad-cube")
    invalid_mesh = type(mesh)(
        resource_id=mesh.resource_id,
        vertices=mesh.vertices,
        faces=flipped_faces,
        unit=mesh.unit,
        axis_transform=mesh.axis_transform,
        metadata=mesh.metadata,
    )

    try:
        preprocessor.prepare(invalid_mesh, require_sdf=True, require_samples=False)
    except ValidationError as exc:
        assert exc.code == "INCONSISTENT_WINDING"
    else:
        raise AssertionError("预期高精度 SDF 预处理拒绝法向不一致的网格。")
