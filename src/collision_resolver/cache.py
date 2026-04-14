from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from collision_resolver.config import PreprocessingConfig
from collision_resolver.models import AABB, ConvexPart, MeshResource, PreprocessedMesh


class PreprocessCache:
    def __init__(self) -> None:
        self._entries: dict[str, PreprocessedMesh] = {}

    def get(self, key: str) -> PreprocessedMesh | None:
        return self._entries.get(key)

    def put(self, key: str, value: PreprocessedMesh) -> None:
        self._entries[key] = value


class ConvexDecompositionCache:
    def load(
        self,
        resource: MeshResource,
        normalized_vertices: np.ndarray,
        faces: np.ndarray,
        config: PreprocessingConfig,
    ) -> tuple[ConvexPart, ...] | None:
        raise NotImplementedError

    def save(
        self,
        resource: MeshResource,
        normalized_vertices: np.ndarray,
        faces: np.ndarray,
        config: PreprocessingConfig,
        convex_parts: tuple[ConvexPart, ...],
    ) -> None:
        raise NotImplementedError


class FilesystemConvexDecompositionCache(ConvexDecompositionCache):
    CACHE_VERSION = 1

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)

    def load(
        self,
        resource: MeshResource,
        normalized_vertices: np.ndarray,
        faces: np.ndarray,
        config: PreprocessingConfig,
    ) -> tuple[ConvexPart, ...] | None:
        cache_path = self._cache_path(resource)
        if cache_path is None or not cache_path.is_file():
            return None

        expected_manifest = self._build_manifest(resource, normalized_vertices, faces, config)
        try:
            with np.load(cache_path, allow_pickle=False) as archive:
                manifest = json.loads(str(archive["manifest"].item()))
                manifest_header = dict(manifest)
                manifest_header.pop("part_count", None)
                if manifest_header != expected_manifest:
                    return None

                convex_parts: list[ConvexPart] = []
                for index in range(int(manifest["part_count"])):
                    part_vertices = np.asarray(archive[f"part_{index}_vertices"], dtype=np.float64)
                    part_faces = np.asarray(archive[f"part_{index}_faces"], dtype=np.int64)
                    convex_parts.append(
                        ConvexPart(
                            part_id=f"{resource.resource_id}:part-{index}",
                            vertices=part_vertices,
                            faces=part_faces,
                            local_aabb=AABB.from_points(part_vertices),
                        )
                    )
        except Exception:
            return None

        return tuple(convex_parts)

    def save(
        self,
        resource: MeshResource,
        normalized_vertices: np.ndarray,
        faces: np.ndarray,
        config: PreprocessingConfig,
        convex_parts: tuple[ConvexPart, ...],
    ) -> None:
        cache_path = self._cache_path(resource)
        if cache_path is None:
            return

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = self._build_manifest(resource, normalized_vertices, faces, config)
        manifest["part_count"] = len(convex_parts)
        arrays: dict[str, np.ndarray] = {
            "manifest": np.asarray(json.dumps(manifest, ensure_ascii=False)),
        }
        for index, part in enumerate(convex_parts):
            arrays[f"part_{index}_vertices"] = np.asarray(part.vertices, dtype=np.float64)
            arrays[f"part_{index}_faces"] = np.asarray(part.faces, dtype=np.int64)
        np.savez_compressed(cache_path, **arrays)

    def _cache_path(self, resource: MeshResource) -> Path | None:
        source_filename = resource.metadata.get("source_filename")
        if not isinstance(source_filename, str) or source_filename == "":
            return None
        return self.cache_dir / f"{source_filename}.npz"

    def _build_manifest(
        self,
        resource: MeshResource,
        normalized_vertices: np.ndarray,
        faces: np.ndarray,
        config: PreprocessingConfig,
    ) -> dict[str, object]:
        return {
            "version": self.CACHE_VERSION,
            "resource_digest": resource.content_digest(),
            "normalized_digest": self._normalized_digest(normalized_vertices, faces),
            "target_unit": config.target_unit,
            "coacd_max_convex_parts": config.coacd_max_convex_parts,
        }

    def _normalized_digest(self, normalized_vertices: np.ndarray, faces: np.ndarray) -> str:
        hasher = hashlib.sha256()
        hasher.update(np.asarray(normalized_vertices, dtype=np.float64).tobytes())
        hasher.update(np.asarray(faces, dtype=np.int64).tobytes())
        return hasher.hexdigest()
