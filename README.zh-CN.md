# collision-resolver

[English](./README.md)

`collision-resolver` 是一个基于 Python 的 3D 网格碰撞检测工具。当前版本按照对称 SDF 损失公式计算碰撞项：

$$
\mathcal{L}_{\text{col}} = \mathcal{L}_{B \to A} + \mathcal{L}_{A \to B}, \quad
\mathcal{L}_{B \to A} = \frac{1}{|\mathcal{S}_B|}\sum_{\mathbf{p}\in\mathcal{S}_B}\big[\max(0,-\Phi_A(\mathbf{T}_A^{-1}\mathbf{T}_B\mathbf{p}))\big]^2
$$

其中 `S_A`、`S_B` 为离线提取并缓存的表面点集。

## 功能

- 对称双向损失：`B -> A` 与 `A -> B`
- 离线缓存 watertight 网格、SDF 体素与表面点集
- 支持 A/B 位姿联合优化（同时更新两个 4x4 变换矩阵）
- 运行时输入两个模型路径与两个 4x4 变换矩阵
- 若未提供变换矩阵，默认使用单位矩阵（保持原始位姿）
- 输出双向穿透统计、总损失与碰撞包围盒
- 支持优化前后可视化（网格、穿透点、碰撞包围盒）

## 非 Watertight 网格自动修复

当输入网格不是 watertight 时，程序会在预处理阶段自动执行：

1. 拓扑清理，包括重复三角形、退化三角形、非流形结构和孤立顶点清理
2. 基于体素化、实体填充与 marching cubes 的闭合曲面重建

只有在修复完成且通过 watertight 校验后，程序才会继续执行 SDF 与表面点缓存构建。若修复失败，程序会直接报错。

## 批量预处理与缓存

支持将预处理流程独立执行：

1. 网格加载与 watertight 修复
2. SDF 体素缓存构建
3. 表面点集离线采样并缓存

默认缓存目录为 `data/sdf_cache`，每个模型会按源文件名（不含扩展名）建立独立缓存子目录。

默认输入目录为 `data/models_eval` 的批量预处理命令：

```bash
uv run collision-resolver-preprocess
```

指定输入目录、缓存目录和离线点数：

```bash
uv run collision-resolver-preprocess data/models_eval \
    --cache-dir data/sdf_cache \
    --surface-point-count 20000
```

## 环境与依赖

项目使用 `uv` 管理环境与依赖：

```bash
uv sync
```

如需可视化功能，可执行：

```bash
uv sync --extra visualize
```

## 命令行用法

```bash
uv run collision-resolver <mesh_a> <mesh_b> [options]
```

默认仅输入两个模型路径即可，`T_A` 与 `T_B` 默认都为单位矩阵。

示例 1：默认单位变换

```bash
uv run collision-resolver data/open_box.ply data/inner_cube.ply
```

示例 2：通过命令行直接传入 4x4 矩阵（按行主序 16 个数）

```bash
uv run collision-resolver data/open_box.ply data/inner_cube.ply \
    --transform-b 1 0 0 0  0 1 0 0  0 0 1 -0.01  0 0 0 1
```

示例 3：从文件读取变换矩阵（`.npy` 或文本）

```bash
uv run collision-resolver data/open_box.ply data/inner_cube.ply \
    --transform-a-file data/T_a.txt \
    --transform-b-file data/T_b.txt
```

示例 4：联合优化 A/B 位姿

```bash
uv run collision-resolver data/open_box.ply data/inner_cube.ply \
    --optimize \
    --max-opt-iters 20
```

示例 5：联合优化并可视化优化前后

```bash
uv run collision-resolver data/open_box.ply data/inner_cube.ply \
    --optimize-visualize \
    --max-opt-iters 20
```

## 主要参数

- `--transform-a` / `--transform-b`：网格 A/B 的 4x4 变换矩阵（16 个浮点数）
- `--transform-a-file` / `--transform-b-file`：从文件加载 4x4 变换矩阵
- `--sdf-cache-dir`：预处理缓存目录，默认 `data/sdf_cache`
- `--rebuild-preprocess-cache`：强制重建缓存
- `--surface-point-count`：离线表面点集采样数（缓存构建时使用）
- `--voxel-size-ratio` / `--padding-ratio` / `--max-grid-dim`：SDF 体素缓存参数
- `--optimize`：启用 A/B 联合优化
- `--max-opt-iters`：联合优化最大迭代数
- `--opt-grad-eps`：联合优化数值梯度的中心差分步长
- `--opt-init-step` / `--opt-backtrack-factor` / `--opt-armijo-c`：回溯线搜索参数
- `--opt-grad-tol` / `--opt-loss-tol` / `--opt-min-step`：联合优化停止条件参数
- `--visualize`：可视化当前或最终状态
- `--optimize-visualize`：优化时可视化优化前后两种状态

## 代码结构

- `src/collision_resolver/formula_collision.py`：公式一致版对称损失计算
- `src/collision_resolver/cli.py`：命令行入口
- `src/collision_resolver/preprocess_cache.py`：网格预处理、SDF 缓存与离线表面点缓存
- `src/collision_resolver/preprocess_models.py`：批量预处理脚本入口
