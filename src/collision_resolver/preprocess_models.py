from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from collision_resolver.mesh_repair import WatertightRepairError
from collision_resolver.preprocess_cache import (
    DEFAULT_CACHE_DIR,
    iter_mesh_files,
    preprocess_mesh_with_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch preprocess mesh files: watertight repair + SDF cache build."
        ),
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="data/models_eval",
        help="Input directory containing mesh files.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(DEFAULT_CACHE_DIR),
        help="Directory for preprocessed watertight mesh and SDF cache.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild cache for every model.",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only scan the top level of input_dir.",
    )
    parser.add_argument(
        "--voxel-size-ratio",
        type=float,
        default=1e-2,
        help="SDF volume voxel size ratio against mesh bbox diagonal.",
    )
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=5e-2,
        help="SDF volume bbox padding ratio against mesh bbox diagonal.",
    )
    parser.add_argument(
        "--max-grid-dim",
        type=int,
        default=160,
        help="Upper bound of SDF volume grid resolution per axis.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    cache_dir = Path(args.cache_dir)

    mesh_files = iter_mesh_files(
        input_dir,
        recursive=not args.non_recursive,
    )
    if not mesh_files:
        raise ValueError(f"No mesh files found in: {input_dir}")

    cache_hits = 0
    cache_builds = 0
    watertight_failures: list[tuple[Path, str]] = []
    other_failures: list[tuple[Path, str]] = []

    for idx, mesh_path in enumerate(mesh_files, start=1):
        logger.info("Preprocess [{}/{}]: {}", idx, len(mesh_files), mesh_path)
        try:
            asset = preprocess_mesh_with_cache(
                mesh_path,
                cache_root=cache_dir,
                force_rebuild=args.force,
                voxel_size_ratio=args.voxel_size_ratio,
                padding_ratio=args.padding_ratio,
                max_grid_dim=args.max_grid_dim,
            )
        except WatertightRepairError as exc:
            watertight_failures.append((mesh_path, str(exc)))
            logger.error("Watertight repair failed: {} -> {}", mesh_path, exc)
            continue
        except Exception as exc:
            other_failures.append((mesh_path, str(exc)))
            logger.error("Preprocess failed: {} -> {}", mesh_path, exc)
            continue

        if asset.cache_hit:
            cache_hits += 1
        else:
            cache_builds += 1

    any_watertight_failure = len(watertight_failures) > 0
    print("=" * 72)
    print(f"{'MESH PREPROCESS SUMMARY':^72}")
    print("=" * 72)
    print(f"Input directory: {input_dir}")
    print(f"Cache directory: {cache_dir}")
    print(f"Total mesh files: {len(mesh_files)}")
    print(f"Cache hit count: {cache_hits}")
    print(f"Cache build count: {cache_builds}")
    print(f"Watertight repair failures: {len(watertight_failures)}")
    print(f"Other preprocess failures: {len(other_failures)}")
    print(f"Any watertight repair failure: {'YES' if any_watertight_failure else 'NO'}")

    if watertight_failures:
        print("\n[Watertight Repair Failures]")
        for path, reason in watertight_failures:
            print(f"- {path}: {reason}")

    if other_failures:
        print("\n[Other Failures]")
        for path, reason in other_failures:
            print(f"- {path}: {reason}")

    print("=" * 72)
    return 1 if (watertight_failures or other_failures) else 0


def main() -> None:
    args = parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
