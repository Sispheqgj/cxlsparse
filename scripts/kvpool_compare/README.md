# 最终五组系统对照

从当前源码冻结快照，让五组对照使用相同模型、输入和系统代码。运行真实
Scheduler / gRPC / Worker / MemoryPool；不提供用 DRAM 模拟 CXL 性能的模式。
归档结果见 [results/final](../../results/final/README.md)。

## 五组配置

| ID | 方式 | 跨 Query 去重 | 交接粒度 |
| --- | --- | --- | --- |
| `rdma_batch_private` | GPUDirect RDMA，每次注册目标 | 关 | 整批完成后计算 |
| `rdma_optimized_private` | GPUDirect RDMA，目标注册复用 | 关 | 整批完成后计算 |
| `cxl_batch_private` | CXL 普通 GPU 读取 + indexed attention | 关 | 整批完成后计算 |
| `cxl_tma_private` | CXL TMA + 消费深度排序 | 关 | 块级就绪后计算 |
| `cxl_tma_shared` | 本方案，共享块 + CXL TMA + 消费深度排序 | 开 | 块级就绪后计算 |

普通对照的目标空间按 Query 独立，相同冷块会实际重复读取。所有组保留热副本、
按需读取与 CUDA Graph；模型与 attention 逻辑一致。`triton_gr` 是前三组共用的
indexed attention 后端，正式 TMA 路径的 Query 分组与 ready 窗口都为 1。
RDMA Pool 在同一主机启动，结果必须标注 loopback。

## 准备环境

先完成根 README 的环境安装、模型下载和 native 库编译，然后编辑
[configs/nsa-tma.yml](../../configs/nsa-tma.yml)。模型与库路径相对于仓库根目录，
也可填写绝对路径。运行器从该文件读取 DAX 设备、偏移、RDMA 端口等硬件设置。

矩阵运行时按 History 自动设置本次 DAX 长度：H4K 为 32MiB、H8K 为 64MiB、
H16K 为 128MiB，起点使用配置中的偏移。请事先为完整范围预留独占区域。每组会
备份原字节，关闭 Worker 后验证恢复；失败恢复凭据保存在输出目录，不能直接删除。

检查 GPU 无其他任务，并与同机实验共用 `NSA_LOCK_PREFIX` 指定的锁前缀。
例如原实验主机使用 `/tmp/cxlpool-motivation-qiaoguanjia`；其他程序也必须使用相同锁
才能相互协调。设置不同的锁名不能隔离同一 GPU 或 DAX 区域。

```bash
source scripts/nsa_tilelang_environment.sh
export NSA_CPUS=32-47  # 原主机设置；其他机器按实际拓扑选择，未设置则不绑定
export NSA_LOCK_PREFIX=/tmp/cxlpool-motivation-qiaoguanjia
```

`run.sh` 默认使用 `sudo -n`。已具有 DAX/verbs 权限时设置 `NSA_SUDO=0`；
`NSA_PYTHON` 可以指定已安装的 TileLang 环境，`NSA_CACHE_DIR` 指定临时编译缓存。
输出目录必须是新目录，推荐放在仓库外。脚本不会下载模型，也不会修改系统网络配置。

## 先做小规模检查

```bash
bash scripts/kvpool_compare/run.sh \
  --config configs/nsa-tma.yml --output /absolute/path/to/new-smoke \
  --histories 4096 --lengths 128 --rounds 1 \
  --warmups 1 --repeats 2 --skip-mixed --save-outputs

"$NSA_PYTHON" scripts/kvpool_compare/compare.py /absolute/path/to/new-smoke
python3 scripts/kvpool_compare/analyze.py /absolute/path/to/new-smoke
```

这可以检查路径、数值与资源释放，少量样本不能用作正式性能结论。

## 正式矩阵

```bash
bash scripts/kvpool_compare/run.sh \
  --config configs/nsa-tma.yml --output /absolute/path/to/new-formal \
  --histories 4096 8192 16384 --lengths 128 256 \
  --rounds 2 --warmups 3 --repeats 15 --save-outputs

"$NSA_PYTHON" scripts/kvpool_compare/compare.py /absolute/path/to/new-formal
python3 scripts/kvpool_compare/analyze.py /absolute/path/to/new-formal
```

默认输入为 `results/final/request-source.json`。History 从原请求历史按固定规则取出，
候选文本模板固定，仅填充到所需 Extend 长度。这是延迟实验，不是推荐质量评估。
`--cases` 可选择表中部分组；读取量消融与汇总建议保留完整五组。
`--snapshot` 可以重放由本运行器保存且指纹未变化的源码；通常省略，以测试当前代码。

## 文件职责与输出

| 文件 | 职责 |
| --- | --- |
| `run.sh`、`runner.py` | 环境、GPU 空闲检查、持锁、源码冻结与实验调度 |
| `cases.py` | 五组对照及处理变量 |
| `benchmark.py` | Worker/Pool 生命周期、预热、steady/mixed 测量与清理 |
| `evidence.py` | 源码、外部模型、native 库和输入指纹 |
| `compare.py` | 完整 hidden 张量与处理变量匹配检查 |
| `validation.py`、`analyze.py` | 实际执行路径、读取计数和私有/共享关系验证，再汇总性能 |

新运行目录包含源码快照、参数、环境、模型/库指纹、逐请求响应与完整输出。
`summary.csv/json` 给出每点 p10/p50/p90 和计数，`comparisons.csv` 给出加速比，
`correctness.json` 与 `traffic-validation.json` 给出验收证据。
分析器验证数值、后端路径、Graph 捕获与计数后才接受结果。

E2E 取请求往返延迟，Worker 取服务端执行时间。历史 prefill、首次 JIT、Graph 捕获
与预热单独记录。不要用 TMA 共享组与普通组的总加速比代替 TMA 单项收益。
新运行的日志、备份字节、张量、生成 CUDA 源码和缓存均不提交到 Git。
