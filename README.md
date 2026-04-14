# collision-resolver

`collision-resolver` 是一个基于 Open3D 的三角网格碰撞检测与修复工具。核心方法采用 Signed Distance Field (SDF) 进行双向穿透采样，并通过 SDF 梯度方向执行最小平移修复。

## 功能

- 双向 SDF 碰撞检测（`B in A` 与 `A in B`）。
- 穿透深度统计与碰撞包围盒输出。
- 基于梯度推力向量的迭代平移修复。
- 可选可视化（需要安装 `pyvista` 可选依赖）。

## 环境与依赖（uv）

项目使用 `uv` 管理环境与依赖：

```bash
uv sync
```

如果需要可视化：

```bash
uv sync --extra visualize
```

## 命令行用法

```bash
uv run collision-resolver <mesh_a> <mesh_b> [options]
```

示例：仅检测

```bash
uv run collision-resolver data/a.obj data/b.obj --offset 0.0 0.02 0.0
```

示例：检测并修复，导出修复后网格

```bash
uv run collision-resolver data/a.obj data/b.obj \
	--resolve \
	--output-resolved-mesh data/b_resolved.obj
```

示例：修复前后可视化

```bash
uv run collision-resolver data/a.obj data/b.obj \
	--resolve-visualize \
	--samples 30000
```

## 主要参数

- `--samples`：显式指定每个方向的采样点数量。
- `--sample-spacing-ratio`：未显式设置采样点数时，用场景包围盒对角线比例自动估算采样密度。
- `--gradient-eps` / `--gradient-eps-ratio`：梯度有限差分步长。
- `--max-resolve-iters`：最大修复迭代次数。
- `--safety-margin` / `--safety-margin-ratio`：每步平移安全裕度。
- `--resolve`：启用平移修复。
- `--visualize` / `--resolve-visualize`：启用可视化。

## 代码结构

- `src/collision_resolver/sdf_collision.py`：SDF 检测与修复核心算法。
- `src/collision_resolver/cli.py`：命令行入口。
