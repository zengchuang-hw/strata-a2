# Iteration records

Every optimization iteration is one pull request. The PR description carries
the optimization content, the mechanism, the trial-and-error trail and the
evidence links; the file in this directory is the durable record of the same
iteration inside the tree.

Recorded per iteration (InferRSI S6 template):

```text
candidate_id / parent_id / contract_id
瓶颈：原始 trace、事件和时间归因；说明重叠
假设：什么改动，通过什么机制改善什么指标
修改：源码 diff、guards、依赖/编译参数变化
风险：受影响的精度、状态、负载和硬件边界
验证：必须重跑哪些 S5 检查
实验：确切 argv、冷/热条件、预算、结果路径
结果：质量、性能、内存、原始记录、保留/拒绝原因
```

| Iteration | PR | Candidate | Outcome |
| --- | --- | --- | --- |
| 001 | pending | bootstrap layered runtime | pending |
| 002 | pending | paired predictor rotary kernel | pending |
