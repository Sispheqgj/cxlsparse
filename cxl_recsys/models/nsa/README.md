# NSA 实现结构

本目录是项目当前主线：**基于 NSA（Native Sparse Attention）实现 NSA-1B 推理，
在其稀疏选块分支上接入内存语义 KV 缓存池。**

模型采用固定版本的 FLA NSA mean-pooling 实现及 NSA-1B checkpoint，保留压缩注意力、
稀疏选块、局部窗口和门控融合结构。KV 池改造负责把选中的历史块转为共享或 Query 私有
的访存计划，再通过相应后端取数、计算和更新热副本。
FLA kernel 的来源与适配说明见 [nsa_runtime](../../../third_party/nsa_runtime/README.md)。

## 目录与职责

```text
nsa/
├── model.py              NSA-1B 模型、权重加载与逐层推理
├── config.py             checkpoint 元数据、版本与缓存几何校验
├── ops.py                压缩、选块、局部窗口等模型运算
├── fla.py                固定版本 FLA 压缩注意力与在线 Top-K 适配
├── attention.py          indexed Triton selected attention
├── dispatch.py           Query 分桶、静态输入缓冲与 CUDA Graph 执行
├── transport.py          KV 工作区、访存规划、后端分派与热副本提交
├── planning/             取哪些块、各 Query 怎么使用、先取哪个块
├── tma/                  来源准备、TMA 搬运和 IO/Attention 流水
├── backends/             其他搬运与 Attention 后端、对照及共用实现
└── experimental/         未通过完整模型验收的扩展
```

模型类的公开入口是 `from cxl_recsys.models.nsa import NSAModel`，
具体实现位于 [model.py](model.py)。Worker 通过上层 `resolve_model("nsa_1b")` 选择它。

### `planning/`：把稀疏选择转为访存与消费关系

| 文件 | 职责 |
| --- | --- |
| [active.py](planning/active.py) | GPU 上建立活动块表，合并跨 Query 重复块，解析热副本、当前 KV 和后备地址 |
| [query_private.py](planning/query_private.py) | 关闭跨 Query 共享的对照；各 Query 使用独立目标，重复冷块分别外读 |
| [occurrences.py](planning/occurrences.py) | 构造各消费行的块引用与计数，保留重复引用、消费位置和有效范围 |
| [read_order.py](planning/read_order.py) | 管理 IO 提交顺序，组织物理顺序、首次使用和 region 等排序方式 |
| [dependencies.py](planning/dependencies.py) | 根据消费深度和近期共享消费者安排供数优先级 |

共享以同一用户、同一层、同一 KV head 内的块身份为边界。
去重改变的是物理取回次数，各 Query 的逻辑引用和 mask 仍单独保留。

### `tma/`：异步取数与块级交接

| 文件 | 职责 |
| --- | --- |
| [transport.py](tma/transport.py) | 准备源地址与 ready 状态，创建工作区并发起融合 kernel |
| [pipeline.py](tma/pipeline.py) | 构建 IO CTA / Attention CTA 流水；正式 Worker 使用 A/staged 路径 |
| [pipeline.cuh](tma/pipeline.cuh) | TMA、同步、内存可见性等设备端基础操作 |
| [launch.py](tma/launch.py) | 保持 cooperative launch 约束，使生产者和消费者能够共同驻留 |

正式路径的数据流为：

```text
已注册的 CXL 地址 → TMA → IO shared → HBM 共享块 + ready
                                               ↓
                                      Attention shared → 计算
```

完整连续冷块由 TMA 直接读取；混合冷热块和尾块先按 token 来源收集。
IO 在目标 HBM 写入完成后以 release 发布 ready，Attention 以 acquire 确认后读取。
这里的 TMA 访问 GPU 可寻址的注册地址，链路经主机桥到达 CXL 内存。

### `backends/` 与 `experimental/`

`backends/` 保留仍可调用的其他路径：[legacy_transport.py](backends/legacy_transport.py)
和 [gpu_transport.py](backends/gpu_transport.py) 管理对应的工作区与搬运；
`tilelang.py`、`indexed.py`、`fused.py`、`gpu_fused.py` 提供 Attention kernel；
[io_plan.py](backends/io_plan.py) 提供分阶段计划及正确性参考。
这些路径有各自的配置和历史结果，使用范围见[实验记录](../../../scripts/kvpool_compare/README.md)。

`experimental/` 中的 [query_tile.py](experimental/query_tile.py) 探索多个 Query 共用计算分组，
[ready.py](experimental/ready.py) 与 [ready.cuh](experimental/ready.cuh) 探索窗口内择就绪消费。
它们保留用于算子研究；正式模型配置使用 **Query 分组 1、ready 窗口 1**。

## 一层计算怎样经过这些模块

1. `model.py` 执行投影，调用 `ops.py` / `fla.py` 得到压缩注意力、窗口注意力和稀疏块选择。
2. `dispatch.py` 为 GR 请求准备 Query 桶和 Graph 输入，`transport.py` 获取对应工作区。
3. `planning/` 解析块的来源、共享关系、消费引用与 IO 顺序。
4. `transport.py` 按选定后端推进取数和 selected attention；TMA 路径进入 `tma/transport.py`。
5. 选块分支结果回到 `model.py`，与压缩、窗口分支门控融合，再完成输出投影与 FFN。

持久历史写入和热副本提交按模型及 transport 的生命周期执行，位于 Graph 预热之外。

历史 KV 的跨请求生命周期由上层 [cached_sparse.py](../cached_sparse.py) 管理；
CXL 映射、注册、页分配和 RDMA 通路由 [mempool](../../mempool/) 提供。

## 配置与阅读入口

主线配置为 [nsa-tma.yml](../../../configs/nsa-tma.yml)，关键选项如下：

| 配置 | 当前主实验使用方式 |
| --- | --- |
| `worker.model` | `nsa_1b` |
| `nsa_kernel_backend` | `fla`，用于压缩注意力与稀疏选块 |
| `nsa_selected_backend` | `tilelang_tma`，用于选块分支的取数与计算流水 |
| `nsa_cross_query_dedup` | 本方案开启；普通对照及去重消融关闭 |
| `nsa_selected_read_order` | 正式五组实验的 TMA 路径使用 `consumption_depth`；样例同样为 `consumption_depth` |
| `nsa_selected_query_group` / `nsa_selected_ready_window` | 均为 `1` |

`triton_gr` 使用整批取回后执行 indexed Triton attention，是普通对照的计算后端。
根目录默认配置仍为模拟模型与 DRAM；运行 NSA/CXL/TMA 应显式选择对应配置。

建议先读 [model.py](model.py) 和 [transport.py](transport.py)，再沿规划与 TMA 的调用阅读。
完整技术说明见[内存语义 KV 池](../../../docs/ARCHITECTURE.md)，
五组对照与复现入口见 [kvpool_compare](../../../scripts/kvpool_compare/README.md)。
