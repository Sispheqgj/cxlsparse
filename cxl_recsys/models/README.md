# 模型代码

统一入口是 [`resolve_model`](./__init__.py)。**当前内存语义 KV 池主线基于 NSA-1B 实现**，
代码集中在 [`nsa/`](nsa/README.md)。公共代码保留在本层。

| 目录 / 文件 | 职责 |
| --- | --- |
| [`nsa/`](nsa/README.md) | NSA-1B 骨干及内存语义 KV 池访问，当前主线；目录内 README 说明具体结构 |
| `simulation/` | 系统联调用模拟骨干与随机权重 Indexer |
| `base.py` | 模型接口与请求缓存上下文 |
| `cached_sparse.py` | 共用的历史 KV 生命周期与稀疏缓存集成 |
| `cuda_graph.py` | CUDA Graph 缓存与回放 |
| `tensor_ops.py` | RMSNorm 和 RoPE 共用数学运算 |

NSA 从 [`nsa/model.py`](nsa/model.py) 进入，当前路径为
`dispatch.py → transport.py → planning/ + tma/`：

- `planning/`：共享/Query 私有块表、消费关系和供数排序。
- `tma/`：TMA 来源准备、A 路径 IO/Attention 流水、设备端实现及启动。
- `backends/`：普通加载、Triton/TileLang 对照及其共用的搬运实现。
- `experimental/`：未通过完整模型验收的 Query 分组、ready 窗口扩展；正式配置使用分组 1、窗口 1。

实验对照、复现脚本和接受范围见[实验说明](../../scripts/kvpool_compare/README.md)。
