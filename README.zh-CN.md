# collision-resolver

[English](./README.md)

`collision-resolver` 是一个基于 Python 的 3D 网格碰撞检测与解碰工具。核心方法采用 Signed Distance Field（SDF）进行双向穿透采样，并沿 SDF 梯度方向执行尽量小的平移修复。

## 功能

- 双向 SDF 碰撞检测（`B in A` 与 `A in B`）
- 支持穿透容差判定，避免贴合接触被误判为碰撞
- 穿透深度统计与碰撞包围盒输出
- 基于梯度方向的迭代平移解碰
- 对非 watertight 网格先自动修复，再执行 SDF 计算
- 支持使用 `pyvista` 进行可选可视化

## 非 Watertight 网格自动修复

当输入网格不是 watertight 时，程序会在加载阶段自动执行：

1. 拓扑清理，包括重复三角形、退化三角形、非流形结构和孤立顶点清理
2. 基于体素化、实体填充与 marching cubes 的闭合曲面重建

只有在修复完成且通过 watertight 校验后，程序才会继续执行 SDF 检测与解碰流程。若修复失败，程序会直接报错，而不是依赖不稳定的后续结果。

## 批量预处理与 SDF 缓存

支持将预处理流程独立执行：

1. 网格加载与 watertight 修复
2. SDF 体素缓存构建

默认缓存目录为 `data/sdf_cache`，每个模型会按源文件名（不含扩展名）建立独立缓存子目录。

默认输入目录为 `data/models_eval` 的批量预处理命令：

```bash
uv run collision-resolver-preprocess
```

指定输入目录和缓存目录：

```bash
uv run collision-resolver-preprocess data/models_eval --cache-dir data/sdf_cache
```

汇总输出会报告：

- watertight 修复失败的模型数量
- 是否存在 watertight 修复失败模型（`YES` 或 `NO`）

## 环境与依赖

项目使用 `uv` 管理环境与依赖：

```bash
uv sync
```

如需可视化依赖，可执行：

```bash
uv sync --extra visualize
```

## 命令行用法

```bash
uv run collision-resolver <mesh_a> <mesh_b> [options]
```

运行时，程序会先根据输入网格文件名（不含扩展名）在 `data/sdf_cache` 中查找预处理缓存。若命中缓存，则直接使用缓存中的 watertight 网格与 SDF 数据。

碰撞判定使用穿透容差 `penetration_tolerance`：仅当 `SDF < -penetration_tolerance` 时才认定为穿透。默认情况下，该容差会根据缓存 SDF 体素步长自动推导，从而避免“完全贴合但被浮点与离散化误判为碰撞”的情况。

示例：仅检测

```bash
uv run collision-resolver data/a.obj data/b.obj --offset 0.0 0.02 0.0
```

示例：检测并解碰，导出解碰后的网格

```bash
uv run collision-resolver data/a.obj data/b.obj \
    --resolve \
    --output-resolved-mesh data/b_resolved.obj
```

示例：解碰前后可视化

```bash
uv run collision-resolver data/a.obj data/b.obj \
    --resolve-visualize \
    --samples 30000
```

## 主要参数

- `--samples`：显式指定每个方向的采样点数量
- `--sample-spacing-ratio`：未显式设置 `--samples` 时，按场景包围盒对角线估算采样密度
- `--gradient-eps` / `--gradient-eps-ratio`：梯度有限差分步长
- `--max-resolve-iters`：最大解碰迭代次数
- `--safety-margin` / `--safety-margin-ratio`：每步平移的安全裕度
- `--penetration-tolerance` / `--penetration-tolerance-ratio`：穿透容差（未显式设置时按缓存 SDF 体素步长自动推导）
- `--resolve`：启用平移解碰
- `--visualize` / `--resolve-visualize`：启用可视化
- `--sdf-cache-dir`：指定预处理缓存目录，默认值为 `data/sdf_cache`
- `--rebuild-preprocess-cache`：强制重建输入网格的预处理缓存

如需严格零容差判定，可显式指定：

```bash
uv run collision-resolver data/a.obj data/b.obj --penetration-tolerance 0.0
```

## 代码结构

- `src/collision_resolver/sdf_collision.py`：SDF 碰撞检测与解碰核心算法
- `src/collision_resolver/cli.py`：命令行入口
- `src/collision_resolver/preprocess_cache.py`：网格预处理与 SDF 缓存模块
- `src/collision_resolver/preprocess_models.py`：批量预处理脚本入口
