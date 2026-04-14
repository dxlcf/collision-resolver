# collision-resolver

`collision-resolver` 是一个基于 Open3D 的三角网格碰撞检测与修复工具。核心方法采用 Signed Distance Field (SDF) 进行双向穿透采样，并通过 SDF 梯度方向执行最小平移修复。

## 功能

- 双向 SDF 碰撞检测（`B in A` 与 `A in B`）。
- 穿透深度统计与碰撞包围盒输出。
- 基于梯度推力向量的迭代平移修复。
- 非闭合网格自动修复（先执行拓扑清理与体素闭合重建，再进入 SDF 计算）。
- 可选可视化（需要安装 `pyvista` 可选依赖）。

## 非闭合网格自动修复

当输入网格不是 watertight 时，程序会在加载阶段自动执行：

1. 拓扑清理（重复/退化三角形、非流形边、孤立顶点）。
2. 基于体素化 + 体填充 + marching cubes 的闭合曲面重建。

修复完成且通过 watertight 校验后，才会继续执行 SDF 检测与修复流程；若修复失败会直接报错，避免不稳定 SDF 结果。

## 批量预处理与SDF缓存

支持将网格预处理步骤独立执行：

1. 网格加载与 watertight 修复。
2. SDF 体素缓存构建。

默认缓存目录为 `data/sdf_cache`，每个模型按“文件名（不含后缀）”建立独立缓存子目录。

批处理命令（默认输入 `data/models_eval`）：

```bash
uv run collision-resolver-preprocess
```

指定输入目录与缓存目录：

```bash
uv run collision-resolver-preprocess data/models_eval --cache-dir data/sdf_cache
```

脚本会在汇总中输出：

- watertight 修复失败模型数。
- 是否存在 watertight 修复失败模型（YES/NO）。

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

运行时会先根据输入网格文件名（不含后缀）在 `data/sdf_cache` 中查找预处理缓存；命中则直接使用缓存的 watertight 网格与 SDF，跳过网格预处理。

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
- `--sdf-cache-dir`：指定预处理缓存目录（默认 `data/sdf_cache`）。
- `--rebuild-preprocess-cache`：强制重建输入网格的预处理缓存。

## 代码结构

- `src/collision_resolver/sdf_collision.py`：SDF 检测与修复核心算法。
- `src/collision_resolver/cli.py`：命令行入口。
- `src/collision_resolver/preprocess_cache.py`：网格预处理与SDF缓存模块。
- `src/collision_resolver/preprocess_models.py`：批量预处理脚本入口。
