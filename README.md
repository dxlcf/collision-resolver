# collision-resolver

[中文说明](./README.zh-CN.md)

`collision-resolver` is a Python tool for collision detection between 3D meshes. The current workflow follows the symmetric SDF loss formulation and evaluates both `B -> A` and `A -> B` terms.

## Features

- Symmetric bidirectional loss (`B -> A` and `A -> B`)
- Offline cache of watertight mesh, SDF volume, and surface point cloud
- Joint A/B pose optimization (both 4x4 transforms are updated together)
- Three joint optimization modes: translation-only, rotation-only, or full 6DoF
- Runtime inputs: two mesh paths and two optional 4x4 transforms
- Identity transforms are used by default when transforms are omitted
- Penetration statistics, total loss, and collision bbox reporting
- Optional visualization for meshes, penetration points, and collision bbox

## Automatic Repair for Non-Watertight Meshes

When an input mesh is not watertight, the program automatically performs the following during loading:

1. Topology cleanup, including duplicate and degenerate triangle removal, non-manifold cleanup, and isolated vertex removal
2. Closed-surface reconstruction based on voxelization, solid filling, and marching cubes

The workflow continues to SDF detection and resolution only after the repaired mesh passes the watertight validation. If repair fails, the program exits with an error instead of relying on unstable downstream results.

## Batch Preprocessing and Cache

The preprocessing pipeline can be executed independently:

1. Mesh loading and watertight repair
2. SDF voxel cache construction
3. Offline surface point cloud extraction and cache save

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

Install optional visualization dependencies if needed:

```bash
uv sync --extra visualize
```

## CLI Usage

```bash
uv run collision-resolver <mesh_a> <mesh_b> [options]
```

At runtime, the program first looks for preprocessing cache entries in `data/sdf_cache` based on mesh filenames without extensions. If a cache hit is found, cached watertight mesh, SDF volume, and offline surface points are reused.

Example: default identity transforms

```bash
uv run collision-resolver data/a.obj data/b.obj
```

Example: pass a 4x4 transform inline (row-major 16 values)

```bash
uv run collision-resolver data/a.obj data/b.obj \
    --transform-b 1 0 0 0  0 1 0 0  0 0 1 -0.01  0 0 0 1
```

Example: load transforms from files (`.npy` or text)

```bash
uv run collision-resolver data/a.obj data/b.obj \
    --transform-a-file data/T_a.txt \
    --transform-b-file data/T_b.txt
```

Example: joint optimization for A/B transforms

```bash
uv run collision-resolver data/a.obj data/b.obj \
    --optimize \
    --max-opt-iters 20
```

Example: translation-only optimization

```bash
uv run collision-resolver data/a.obj data/b.obj \
    --optimize \
    --optimize-mode translation
```

Example: rotation-only optimization

```bash
uv run collision-resolver data/a.obj data/b.obj \
    --optimize \
    --optimize-mode rotation
```

Example: full 6DoF optimization

```bash
uv run collision-resolver data/a.obj data/b.obj \
    --optimize \
    --optimize-mode 6dof
```

Example: optimize with before/after visualization

```bash
uv run collision-resolver data/a.obj data/b.obj \
    --optimize-visualize \
    --max-opt-iters 20
```

## Main Options

- `--transform-a` / `--transform-b`: 4x4 transform matrices (16 values)
- `--transform-a-file` / `--transform-b-file`: load 4x4 transform matrices from files
- `--sdf-cache-dir`: specify the preprocessing cache directory, defaulting to `data/sdf_cache`
- `--rebuild-preprocess-cache`: force rebuilding the preprocessing cache for input meshes
- `--surface-point-count`: offline surface sample count used during cache build
- `--voxel-size-ratio` / `--padding-ratio` / `--max-grid-dim`: SDF cache parameters
- `--optimize`: enable joint optimization for A/B transforms
- `--optimize-mode`: choose `translation`, `rotation`, or `6dof`
- `--max-opt-iters`: max optimization iterations
- `--opt-grad-eps`: central-difference epsilon for optimization gradients
- `--opt-init-step` / `--opt-backtrack-factor` / `--opt-armijo-c`: backtracking line-search controls
- `--opt-grad-tol` / `--opt-loss-tol` / `--opt-min-step`: optimization stopping controls
- `--visualize`: visualize current/final state
- `--optimize-visualize`: when optimizing, visualize both before and after states

## Code Structure

- `src/collision_resolver/formula_collision.py`: symmetric SDF loss evaluation
- `src/collision_resolver/cli.py`: command-line entrypoint
- `src/collision_resolver/preprocess_cache.py`: mesh preprocessing, SDF cache, and offline surface point cache
- `src/collision_resolver/preprocess_models.py`: batch preprocessing script entrypoint
