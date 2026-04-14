## ADDED Requirements

### Requirement: Broadphase filters candidate collision pairs using AABB
系统 MUST 在精确碰撞检测前先执行基于轴对齐包围盒的候选碰撞对过滤。只有 mesh 级或凸块级 AABB 存在重叠的对象对，才允许进入后续精确碰撞检测阶段。

#### Scenario: Skip narrowphase for separated objects
- **WHEN** 两个对象的 AABB 不发生重叠
- **THEN** 系统直接判定该对象对不需要进入精确碰撞检测

### Requirement: Narrowphase computes exact contacts from convex decomposition and FCL
系统 MUST 对通过 broadphase 的对象对使用 CoACD 凸分解结果构造候选凸块对，并通过 FCL 执行精确碰撞检测。若对象发生相交，系统 MUST 返回至少一个包含接触法线、接触点或其等价位置描述，以及穿透深度的接触结果。

#### Scenario: Report contact information for a colliding pair
- **WHEN** 两个经过凸分解的对象在精确碰撞检测中发生相交
- **THEN** 系统返回结构化接触结果，并为每个被报告的接触提供法线与穿透深度信息

### Requirement: Exact detection must not degrade to approximate backends
系统 MUST 将 CoACD 与 FCL 视为精确检测的强制后端。若任一后端缺失、初始化失败或所需工件不可用，系统 MUST 返回检测失败，而不是退化为单凸块、AABB 或其他近似碰撞实现。

#### Scenario: Surface backend unavailability as detection failure
- **WHEN** 检测阶段发现 CoACD/FCL 后端不可用或缺少对应正式工件
- **THEN** 系统返回检测失败状态，并明确说明失败原因

### Requirement: Detection returns both raw contacts and summary fields
系统 MUST 将精确碰撞检测结果表示为接触集合与摘要字段的组合。返回结果 MUST 包含碰撞状态，并在发生碰撞时提供接触集合、最大穿透深度，以及可用于上层决策的代表性法线或等价摘要信息。

#### Scenario: Return structured collision summary
- **WHEN** 调用方请求某一对象对的碰撞检测结果
- **THEN** 系统返回既可用于程序消费又可用于调试诊断的结构化碰撞结果，而不是仅返回布尔值

### Requirement: Detection failures are distinguishable from collision-free results
系统 MUST 明确区分“未发生碰撞”和“检测过程失败”这两类结果。任何由第三方库异常、依赖缺失、预处理缺失或输入不合法导致的检测中断 MUST 通过单独状态或错误码表达。

#### Scenario: Surface a narrowphase failure
- **WHEN** 精确碰撞检测阶段因后端异常、依赖缺失或缺少正式工件而无法完成
- **THEN** 系统返回检测失败状态，而不是将该对象对误报为无碰撞
