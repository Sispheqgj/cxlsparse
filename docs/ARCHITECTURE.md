# NSA KV 池的访问与交接

## 场景和对象

GR 请求包含稳定用户历史和本次候选后缀。NSA 的不同 Query 位置分别选择历史块，
这些选择经常重叠。历史 KV 跨请求保留；候选后缀只在本次 forward 中有效，不写回
持久用户历史。HBM 保存热副本、压缩摘要、当前 KV 和临时工作集，完整历史放在后备池。

本方案用同一套块标识连接“取什么”“先取谁”和“谁可以开始算”：

```text
各 Query 的逻辑选块
    ↓ 保留每个消费者的引用
唯一块身份 → 来源地址 / 有效范围 / HBM 工作位置
    ↓ 根据消费位置排列外读
异步取回 → 目标写入完成 → ready[块]
                              ↓
                依赖该块的 Attention 消费者
```

块身份以同一用户、层和 KV head 为边界。跨 Query 去重只合并物理取回，各 Query 的
重复引用、因果 mask 与 Attention 状态保持独立。正式实现没有把不同用户或层的内容
按裸指针合并，也没有实现跨 Host 版本一致性。

## 规划

`models/nsa/planning/active.py` 解析活动块、热副本、当前 KV 和后备地址；
`occurrences.py` 保留每个消费行的块引用。消费行粒度是 Query × KV head，一个
Attention CTA 内处理共享该 KV head 的一组 Q heads。

`dependencies.py` 利用消费位置和共享消费者形成静态供数优先级，
`read_order.py` 将它写入实际 IO 顺序。正式配置为 `consumption_depth`。这是本次
执行前建立的计划，不是根据消费者实时进度反馈的动态调度。

关闭 `nsa_cross_query_dedup` 时，`query_private.py` 为每个 Query 分配独立工作目标，
相同冷块需要分别读取。这是普通 baseline 和去重消融使用的路径。

## TMA 读取和可见性

CXL 内存经 devdax `mmap`、CUDA 注册获得 GPU 可访问地址。H100 的请求仍经实际的
GPU/CPU 互连、主机桥和 CXL 链路到达内存。TMA 是搬运机制，不是新的 CXL 链路协议。

当前完整块为 `16 tokens × (K64 + V64) × 2 bytes = 4096 bytes`。
`tma/transport.py` 在 GPU 上检查完整性、连续性和对齐：完整冷块直接 TMA；
混合冷热块和尾块先按 token 来源收集；全驻留块不再外读。

`tma/pipeline.py` 使用正式 A/staged 路径：

```text
CXL → IO CTA shared → HBM 工作块 → Attention CTA shared → 计算
                            │
                 写入完成后 release ready
                            ↓
                    消费者 acquire 观察
```

IO 和 Attention 保持 CTA 分工；不同块可以在搬运、交接和计算阶段重叠。
Cooperative launch 的驻留约束保证生产者和消费者共同推进，避免消费者占满 GPU 后
阻止 IO CTA 运行。映射、注册和工作区的生命周期覆盖所有在途访问与 Graph 使用。

当前 HBM 工作区覆盖整次选块需求，没有实现小容量循环槽位回收；也没有消除 HBM
中转。正式配置 Query 分组和 ready 窗口均为 1，仍按每个消费行规定的顺序计算。

## 系统边界

Scheduler 与 Worker 通过 gRPC 交换控制消息；MemoryPool、热缓存和模型位于 Worker。
RDMA 对照另有 Pool 管理进程。GPU 直访 CXL 与 RDMA READ 都可以按地址访问内存，
本实验比较的是整批完成交接与块级可见性驱动计算，以及是否共享跨 Query 读取。

实验结果支持当前 H100、NSA-1B 和输入矩阵下的系统收益，其他 CXL 拓扑和模型的效果
需要单独测量。
