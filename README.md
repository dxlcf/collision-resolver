# collision-resolver

`collision-resolver` 用于检测两个 3D 三角网格之间的碰撞，并在给定位姿初值的前提下，对可动物体求解满足精确无碰撞约束的最小 XY 平移增量。

## 依赖准备

项目使用 `uv` 管理环境与依赖。本地脚本需要额外安装 `trimesh` 用于读取 `ply` 文件；如果要启用可视化，还需要安装 `open3d`。

```bash
uv sync --dev --extra mesh --extra visualization
```

运行时还要求 `coacd`、`python-fcl`、`torch`、`shapely` 可用；如果这些依赖缺失，初始化会直接失败。

## 本地调用脚本

直接使用：

```bash
uv run python scripts/run_local_case.py <mesh_a.ply> <mesh_b.ply> [参数...]
```

脚本会在仓库根目录下创建 `cache/`，并将每个输入模型的 CoACD 凸分解结果保存为按模型文件名命名的缓存文件。再次运行时，会先校验该缓存是否与当前网格内容和预处理配置一致；一致则直接复用，不再重复执行 CoACD。

### 输入协议

- 第一个位置参数：第一个 `ply` 文件路径
- 第二个位置参数：第二个 `ply` 文件路径
- `--translation-a X Y Z`：第一个模型的初始平移，默认 `0 0 0`
- `--translation-b X Y Z`：第二个模型的初始平移，默认 `0 0 0`
- `--euler-a RX RY RZ`：第一个模型的初始 XYZ 欧拉角，单位度，默认 `0 0 0`
- `--euler-b RX RY RZ`：第二个模型的初始 XYZ 欧拉角，单位度，默认 `0 0 0`
- `--unit-a {m,cm,mm}`：第一个模型长度单位，默认 `m`
- `--unit-b {m,cm,mm}`：第二个模型长度单位，默认 `m`
- `--movable {first,second}`：指定哪个模型可动，默认 `second`
- `--output result.json`：可选，将结果写入 JSON 文件；未提供时打印到标准输出
- `--visualize`：可选，使用 `open3d` 先后可视化初始碰撞区域和优化后结果

默认行为是：**第一个模型固定，第二个模型可动**。

欧拉角采用右手系 XYZ 顺序，内部旋转矩阵为 `Rz @ Ry @ Rx`。

### 输出内容

脚本固定执行一次 `resolve` 流程，输出一个 JSON 对象，主要字段如下：

- `final_status`：最终状态，可能为 `resolved`、`failed`
- `initial_detection`：初始碰撞检测结果
- `optimization`：最小平面平移求解结果
- `verification`：候选位姿的精确复检结果
- `diagnostics`：各阶段诊断信息

其中：

- `initial_detection.is_colliding` 表示初始姿态是否碰撞
- `initial_detection.max_penetration_depth` 表示最大穿透深度
- `optimization.candidate_pose` 给出求解后的最终位姿
- `optimization.planar_delta` 给出本次求得的 `dx`、`dy` 与平移范数
- `optimization.summary.optimality_tolerance` 给出本次“公差内全局最小”求解使用的平移公差
- `optimization.summary.motion_constraints` 说明当前求解固定旋转与 Z，仅允许 XY 平移
- `verification.is_colliding` 表示优化后的姿态是否仍然碰撞

启用 `--visualize` 后，脚本会顺序打开两个 `open3d` 窗口：

- 第一个窗口显示初始姿态，并用红色接触点与法线段标出碰撞区域
- 第二个窗口显示优化后的姿态；如果优化后仍有碰撞，也会继续标出碰撞区域

### 使用示例

```bash
uv run python scripts/run_local_case.py ^
  data/fixed.ply ^
  data/movable.ply ^
  --translation-a 0 0 0 ^
  --euler-a 0 0 0 ^
  --translation-b 0.02 0 0 ^
  --euler-b 0 0 15 ^
  --visualize
```

如果要让第一个模型可动：

```bash
uv run python scripts/run_local_case.py a.ply b.ply --movable first
```

如果要写入文件：

```bash
uv run python scripts/run_local_case.py a.ply b.ply --output result.json
```

## 运行时自检

命令：

```bash
uv run collision-resolver
```

这个入口当前只做运行时依赖自检与默认配置打印，不处理 `ply` 输入。真正的本地调用入口是上面的 `scripts/run_local_case.py`。

## 代码结构

- [scripts/run_local_case.py](/e:/project/tools/collision-resolver/scripts/run_local_case.py:1)：本地调用脚本入口
- [local_cli.py](/e:/project/tools/collision-resolver/src/collision_resolver/local_cli.py:1)：本地脚本参数解析、PLY 读取与 JSON 输出
- [preprocess.py](/e:/project/tools/collision-resolver/src/collision_resolver/preprocess.py:1)：网格预处理、高精度 SDF 构建与缓存
- [detection.py](/e:/project/tools/collision-resolver/src/collision_resolver/detection.py:1)：AABB broadphase 与 `CoACD + FCL` narrowphase
- [resolution.py](/e:/project/tools/collision-resolver/src/collision_resolver/resolution.py:1)：固定旋转与 Z 的最小 XY 平移求解
- [service.py](/e:/project/tools/collision-resolver/src/collision_resolver/service.py:1)：端到端求解编排

## 测试

```bash
uv run pytest
```

## 已知限制

- 当前本地脚本仅支持两个 `ply` 文件输入。
- 当前仅支持三角网格，不支持点云，也不支持四边形面自动三角化。
- 自动修正当前只支持单个可动物体相对静态障碍物的最小 XY 平移求解，不支持旋转或 Z 方向位移。
- 高精度 SDF 预处理要求输入网格满足闭合、法向一致和非退化三角面等几何前提；不满足时会直接拒绝处理。
