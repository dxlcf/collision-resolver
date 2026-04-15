# collision-resolver

[中文说明](./README.zh-CN.md)

`collision-resolver` is a Python tool for detecting collisions between 3D meshes and resolving them with minimal pose adjustment. The core method uses a Signed Distance Field (SDF) for bidirectional penetration sampling and applies translation updates along the SDF gradient direction.

## Features

- Bidirectional SDF collision detection (`B in A` and `A in B`)
- Penetration-tolerance-aware classification to avoid contact false positives
- Penetration depth statistics and collision bounding box output
- Iterative translation-based resolution driven by gradient directions
- Automatic repair for non-watertight meshes before SDF computation
- Optional visualization with `pyvista`

## Automatic Repair for Non-Watertight Meshes

When an input mesh is not watertight, the program automatically performs the following during loading:

1. Topology cleanup, including duplicate and degenerate triangle removal, non-manifold cleanup, and isolated vertex removal
2. Closed-surface reconstruction based on voxelization, solid filling, and marching cubes

The workflow continues to SDF detection and resolution only after the repaired mesh passes the watertight validation. If repair fails, the program exits with an error instead of relying on unstable downstream results.

## Batch Preprocessing and SDF Cache

The preprocessing pipeline can be executed independently:

1. Mesh loading and watertight repair
2. SDF voxel cache construction

The default cache directory is `data/sdf_cache`. Each model uses an independent cache subdirectory named after the source filename without its extension.

Batch preprocessing command with default input directory `data/models_eval`:

```bash
uv run collision-resolver-preprocess
```

Specify both the input directory and cache directory:

```bash
uv run collision-resolver-preprocess data/models_eval --cache-dir data/sdf_cache
```

The summary output reports:

- The number of models whose watertight repair failed
- Whether any watertight repair failures exist (`YES` or `NO`)

## Environment and Dependencies

This project uses `uv` to manage the environment and dependencies:

```bash
uv sync
```

Install the optional visualization dependency if needed:

```bash
uv sync --extra visualize
```

## CLI Usage

```bash
uv run collision-resolver <mesh_a> <mesh_b> [options]
```

At runtime, the program first looks for preprocessing cache entries in `data/sdf_cache` based on the input mesh filenames without extensions. If a cache hit is found, the cached watertight mesh and SDF data are used directly.

Collision classification uses a penetration tolerance `penetration_tolerance`: a sample is considered penetrating only when `SDF < -penetration_tolerance`. By default, this tolerance is auto-derived from cached SDF voxel spacing, which avoids reporting exact contact as penetration due to discretization and floating-point noise.

Example: detection only

```bash
uv run collision-resolver data/a.obj data/b.obj --offset 0.0 0.02 0.0
```

Example: detect and resolve, then export the resolved mesh

```bash
uv run collision-resolver data/a.obj data/b.obj \
    --resolve \
    --output-resolved-mesh data/b_resolved.obj
```

Example: visualize before and after resolution

```bash
uv run collision-resolver data/a.obj data/b.obj \
    --resolve-visualize \
    --samples 30000
```

## Main Options

- `--samples`: explicitly set the number of sampling points per direction
- `--sample-spacing-ratio`: estimate sampling density from the scene bounding box diagonal when `--samples` is not provided
- `--gradient-eps` / `--gradient-eps-ratio`: finite-difference step size for gradient estimation
- `--max-resolve-iters`: maximum number of resolution iterations
- `--safety-margin` / `--safety-margin-ratio`: safety margin for each translation step
- `--penetration-tolerance` / `--penetration-tolerance-ratio`: penetration tolerance (auto-derived from cached SDF voxel spacing when not explicitly set)
- `--resolve`: enable translation-based resolution
- `--visualize` / `--resolve-visualize`: enable visualization
- `--sdf-cache-dir`: specify the preprocessing cache directory, defaulting to `data/sdf_cache`
- `--rebuild-preprocess-cache`: force rebuilding the preprocessing cache for input meshes

For strict zero-tolerance classification, set:

```bash
uv run collision-resolver data/a.obj data/b.obj --penetration-tolerance 0.0
```

## Code Structure

- `src/collision_resolver/sdf_collision.py`: core SDF collision detection and resolution algorithm
- `src/collision_resolver/cli.py`: command-line entrypoint
- `src/collision_resolver/preprocess_cache.py`: mesh preprocessing and SDF cache module
- `src/collision_resolver/preprocess_models.py`: batch preprocessing script entrypoint
