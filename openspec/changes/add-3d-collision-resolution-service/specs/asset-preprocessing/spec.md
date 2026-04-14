## ADDED Requirements

### Requirement: Mesh preprocessing produces reusable exact geometry artifacts
系统 MUST 对输入的 3D 网格执行预处理，并生成后续精确碰撞检测与自动修正可复用的正式几何工件。预处理产物 MUST 至少包含 mesh 级 AABB、CoACD 凸分解结果、供 FCL 使用的凸块几何数据，以及基于原始三角网格生成的高精度体素化网格 SDF。

#### Scenario: Preprocess a valid mesh asset
- **WHEN** 调用方提交一个格式有效、满足几何约束且可用于高精度 SDF 构建的网格资源进行预处理
- **THEN** 系统返回或持久化一组可复用的正式预处理工件，并为该资源生成可追踪的缓存标识

### Requirement: Preprocessing validates mesh suitability for exact SDF generation
系统 MUST 在预处理阶段检查网格是否满足精确检测与高精度 SDF 构建的基本前提，并在资源不适合处理时返回明确失败原因。检查范围 MUST 包含几何完整性、坐标系/单位元数据可解释性、退化三角形约束，以及构建有符号距离场所需的网格条件。

#### Scenario: Reject an unsuitable mesh
- **WHEN** 输入网格存在严重退化、缺失关键元数据或不满足高精度 SDF 构建前提
- **THEN** 系统拒绝继续处理，并返回可区分的预处理失败状态与原因说明

### Requirement: Required native geometry backends must be available before artifact creation
系统 MUST 在创建预处理工件前确认 CoACD、FCL 和高精度 SDF 构建链路所需依赖可用。任一关键依赖缺失时，系统 MUST 直接失败，而不是退化为单凸块、AABB 或其他近似实现。

#### Scenario: Fail fast when a required backend is missing
- **WHEN** 预处理阶段发现 `coacd`、`python-fcl` 或高精度 SDF 构建依赖不可用
- **THEN** 系统返回明确的依赖失败结果，并且不生成任何近似替代工件

### Requirement: Preprocessing artifacts are keyed by mesh content and exact SDF parameters
系统 MUST 基于网格内容与关键预处理参数生成缓存键，并在相同输入和相同参数下复用已有预处理结果。任何影响凸分解或高精度 SDF 结果的参数变化 MUST 导致不同的缓存标识，这些参数至少包括体素精度、包围盒 padding 和符号判定方式。

#### Scenario: Reuse a preprocessing cache entry
- **WHEN** 调用方再次提交内容相同且预处理参数一致的网格资源
- **THEN** 系统命中已有缓存，而不是重复执行完整预处理流程
