## ADDED Requirements

### Requirement: Collision resolution optimizes a movable object's pose against a volumetric mesh SDF
系统 MUST 通过查询基于障碍物原始三角网格生成的高精度体素化网格 SDF 来构造碰撞损失，并对单个可动物体的位姿执行梯度优化，以尽量小的位姿改变量解除碰撞。优化过程中使用的碰撞信号 MUST 来源于可动物体表面采样点或其等价的表面离散表示。

#### Scenario: Optimize a colliding pose
- **WHEN** 一个可动物体与静态障碍物发生碰撞且存在可用的正式体素化网格 SDF
- **THEN** 系统启动位姿优化，并输出一个新的候选位姿用于后续复检

### Requirement: Resolution must not use box-based or fallback distance fields
系统 MUST 使用正式体素化网格 SDF 作为优化阶段的唯一距离场来源。若障碍物缺少正式 SDF 工件或 SDF 构建依赖不可用，系统 MUST 直接失败，而不是退化为局部 AABB、凸块盒并集或其他替代距离场。

#### Scenario: Fail when exact SDF is unavailable
- **WHEN** 自动修正阶段发现至少一个障碍物缺少正式体素化网格 SDF
- **THEN** 系统返回失败状态，并说明缺失的是正式 SDF 工件

### Requirement: Resolution minimizes pose change while reducing collision loss
系统 MUST 在优化目标中同时考虑碰撞损失与相对初始位姿的改变量代价。该能力 MUST 以“尽量小的位姿调整解除碰撞”为目标，而不是允许任意大幅移动对象来换取更低碰撞损失。

#### Scenario: Prefer a smaller valid adjustment
- **WHEN** 存在多个都能解除碰撞的候选位姿
- **THEN** 系统优先选择相对初始位姿改变量更小的候选结果

### Requirement: Resolution exposes optimization outcome and exact SDF provenance
系统 MUST 返回优化阶段的结果摘要，至少包括是否收敛、迭代次数、最终损失、候选位姿、停止原因，以及本次求解使用的 SDF 缓存键和关键精度参数。若优化无法执行或未找到满足条件的候选位姿，系统 MUST 返回可区分的失败或未解决状态。

#### Scenario: Trace solver configuration and SDF provenance
- **WHEN** 调用方发起一次自动修正请求
- **THEN** 返回结果中包含本次求解所使用的关键优化参数摘要，以及对应正式 SDF 工件的来源信息
