## ADDED Requirements

### Requirement: Verification rechecks the optimized pose with exact collision detection
系统 MUST 在自动修正输出候选位姿后，使用精确碰撞检测流程对该候选位姿进行复检。复检 MUST 使用与常规精确检测一致的几何工件、后端路径和判定标准，而不能仅依赖优化阶段的 SDF 值判断成功。

#### Scenario: Verify a resolved candidate pose
- **WHEN** 自动修正阶段输出一个候选位姿
- **THEN** 系统对该候选位姿执行精确碰撞复检，并生成最终验证结论

### Requirement: Verification must not degrade when exact backends are unavailable
系统 MUST 将精确检测后端视为复检阶段的强制前提。若 FCL、CoACD 或所需正式工件不可用，系统 MUST 返回 `failed`，而不是退化为 AABB 或其他近似复检实现。

#### Scenario: Fail verification when exact backends are missing
- **WHEN** 复检阶段发现精确后端或正式工件不可用
- **THEN** 系统返回 `failed`，并明确说明复检无法建立在正式几何基础上

### Requirement: Verification returns authoritative final status
系统 MUST 为自动修正请求输出最终权威结论，至少区分 `resolved`、`unresolved` 和 `failed` 三类状态。只有当复检确认无碰撞时，系统才可以返回 `resolved`。

#### Scenario: Keep unresolved when exact collision remains
- **WHEN** 候选位姿经过精确复检后仍检测到碰撞
- **THEN** 系统返回 `unresolved`，而不是将该结果标记为已修正成功

### Requirement: Verification preserves pre- and post-resolution context
系统 MUST 在最终结果中保留足够的前后文，使调用方能够对比修正前后的检测结果。返回信息 MUST 至少能够关联初始碰撞摘要、候选位姿摘要、复检结论，以及本次复检所使用的正式几何工件标识。

#### Scenario: Compare before and after verification
- **WHEN** 调用方查看一次自动修正请求的最终结果
- **THEN** 系统提供修正前检测摘要、修正后复检摘要及其最终状态，支持调用方判断修正是否有效
