# 基于 NSA 的内存语义 KV Cache Pool

本项目**基于 NSA（Native Sparse Attention）实现 NSA-1B 推理**，面向生成式推荐（GR）中
多个 Query 共用用户历史的场景，将完整历史 KV 放在 CXL 后备池，在 GPU 上合并重复
取数需求，并通过块级就绪状态连接异步搬运与 Attention 计算。

保留 NSA 的压缩注意力、稀疏选块、局部窗口和门控融合；本项目的改造集中在历史 KV
的存储、共享访问、供数顺序和读算流水。代码也提供 RDMA 整批取回与同 CXL 链路对照。

## 工作方式

```text
Scheduler：GR 请求 → gRPC → Worker：历史复用 + NSA 稀疏选块
                                      ↓
                      共享块表 + 每个 Query 的消费引用
                                      ↓
                热副本 / 当前 KV / CXL 后备地址解析
                                      ↓
                    按消费深度排列缺失块的供数顺序
                                      ↓
 CXL 注册地址 → TMA → IO shared → HBM 工作块 → ready[b]
                                                   ↓
                           Attention CTA → shared → 计算
```

“内存语义”在这里指 KV 的身份、可寻址范围、访问依赖和块级可见性约定：

- **共享访问**：同一用户、层和 KV head 内的相同块只取回一次，各 Query 仍保留自己的引用和 mask。
- **按需读取**：区分 HBM 热副本、当前 KV 和缺失历史；连续完整冷块直接使用 TMA，混合块和尾块按来源收集。
- **依赖驱动供数**：根据消费者的使用位置安排缺失块读取，优先满足较早的计算依赖。
- **块级交接**：IO 完成 HBM 写入后 release 发布 ready，Attention acquire 确认后读取，实现 IO/计算流水。

当前采用带 HBM 中转的 A 路径。H100 通过主机桥访问 CXL 后备地址；TMA 不会使它原生
发出 CXL.mem。本项目没有实现跨 Host 缓存一致性。RDMA READ 本身也按地址访问内存，
实验中的“整批取回”描述的是软件交接方式，不能把它理解为 RDMA 不具备内存语义。

## 目录

| 路径 | 内容 |
| --- | --- |
| [cxl_recsys/models/nsa](cxl_recsys/models/nsa/README.md) | NSA 骨干、访存规划、TMA 与 Attention |
| `cxl_recsys/mempool/` | CXL DAX 映射注册、热缓存、页管理、GPUDirect RDMA |
| `cxl_recsys/scheduler/`、`worker/` | GR 请求、调度、推理与历史复用 |
| [configs/nsa-tma.yml](configs/nsa-tma.yml) | 最终方案的 H100 配置样例 |
| [scripts/kvpool_compare](scripts/kvpool_compare/README.md) | 最终五组对照的运行、数值比较和汇总 |
| [results/final](results/final/README.md) | 最终实验精简归档，含输入和验收报告 |
| [third_party](third_party/README.md) | 固定版本 NSA 内核、参考实现和 GR 输入代码 |

## 安装

运行平台为 Linux、Python 3.12。正式 TMA 实验使用 H100 PCIe 80GB、CUDA 13.2、真实
CXL Type-3 devdax 和可用的 GPUDirect RDMA 端口。依赖版本见两份 `uv.lock`。
需要 CUDA 开发工具、C/C++ 编译器、Python 开发头文件；RDMA 还需要 `libibverbs`
及其开发头文件。模型权重和编译库不随代码发布。

```bash
git clone https://github.com/Sispheqgj/cxlsparse.git
cd cxlsparse
git submodule update --init --depth 1 -- third_party/sglang
uv sync --locked --group nsa
uv sync --locked --project tools/nsa-tilelang
source scripts/nsa_tilelang_environment.sh

# 下载固定版本 NSA-1B；默认使用 hf-mirror.com，也可自行设置 HF_ENDPOINT
"$NSA_PYTHON" scripts/download_nsa.py --output artifacts/models/NSA-1B

# 在 TileLang 环境编译本项目使用的 HiSparse/CXL 库
"$NSA_PYTHON" -m cxl_recsys.cli build-kernels \
  --config configs/nsa-tma.yml --cuda-arch 9.0
```

TMA 环境固定 `tilelang==0.1.13`、`apache-tvm-ffi==0.1.12`。请在该环境编译并加载
相应的 native 库，避免和根环境另一版本 TVM FFI 的库混用。`uv sync` 安装依赖时不会
下载模型，也不会编译 CXL 库；首次实际推理还会编译 Triton/TileLang kernel。

## 运行

先编辑 [configs/nsa-tma.yml](configs/nsa-tma.yml)，核对模型、编译库、DAX 保留区和
RDMA 设备。样例的 `/dev/dax0.0`、`254GiB` 偏移和 `mlx5_1` 是实验主机配置，
迁移主机时需要替换。Worker 会写入所配置的 DAX 范围，应先为它预留独占区域。

两个终端分别启动 Worker 与 Scheduler：

```bash
# 终端 1；每个终端先 source 环境。访问 DAX 需具备对应设备权限。
source scripts/nsa_tilelang_environment.sh
"$NSA_PYTHON" -m cxl_recsys.cli worker --config configs/nsa-tma.yml \
  --max-users 2 --max-history 16384

# 终端 2；使用归档的 GR 请求，不需要另外下载 GR 数据集
source scripts/nsa_tilelang_environment.sh
"$NSA_PYTHON" -m cxl_recsys.cli scheduler \
  --config configs/nsa-tma.yml \
  --source requests --input results/final/request-source.json \
  --output runs/replay.json
```

默认 Worker 地址为 `127.0.0.1:50051`，Scheduler 默认连接该地址。模型对候选执行
单步打分并返回推荐诊断；这不是自回归对话服务。单独启动的 Worker 不执行实验运行器
的 DAX 备份恢复；正式对照请使用下述运行器。

上面的回放为归档输入的两位用户各保留最多 16K 历史；五组实验运行器会另行设置
单用户及每个实验点的 History 容量。不要把服务的用户容量设为 1 后直接回放两位用户。

没有 CXL 时，默认 `config.yml` 为模拟模型 + DRAM，用于系统联调；不能把模拟结果
当作真实 NSA 或 CXL 性能。具体配置与 RDMA Pool 启动见 [运行说明](docs/RUNNING.md)。

## 复现最终五组实验

普通对照全部关闭跨 Query 去重；热副本、按需读取和 CUDA Graph 条件保持一致。

| 对照 | 历史 KV 取回与计算方式 | 跨 Query 共享 |
| --- | --- | --- |
| RDMA 整批取回 | 每次注册目标；整批完成后执行 indexed attention | 关 |
| RDMA 整批取回（注册复用） | 复用目标注册、合并连续读取；整批完成后计算 | 关 |
| CXL 整批读取 | 普通 GPU 搬运到 HBM 后执行 indexed attention | 关 |
| CXL 按块流水（不共享） | TMA、消费深度排序、块级 ready | 关 |
| 本方案 | TMA、消费深度排序、块级 ready、共享取回 | 开 |

`triton_gr` 是前面三组使用的 indexed attention 计算后端名称。RDMA 结果为**同主机
GPUDirect RDMA loopback**，不代表双机网络部署性能。

```bash
# 使用新建的仓库外目录。32-47 是原主机的 CPU 亲和性，按机器调整。
NSA_CPUS=32-47 bash scripts/kvpool_compare/run.sh \
  --config configs/nsa-tma.yml \
  --output /absolute/path/to/new-run \
  --histories 4096 8192 16384 --lengths 128 256 \
  --rounds 2 --warmups 3 --repeats 15 --save-outputs

"$NSA_PYTHON" scripts/kvpool_compare/compare.py /absolute/path/to/new-run
python3 scripts/kvpool_compare/analyze.py /absolute/path/to/new-run
```

运行器默认通过 `sudo -n` 访问硬件；已具备设备权限时可设置 `NSA_SUDO=0`。
它检查 GPU 空闲、取得协作锁、冻结当前代码，并备份和恢复实验用 DAX 字节。详细说明、
小规模验证命令和输出解释见 [实验入口](scripts/kvpool_compare/README.md)。

## 最终结果

归档为 2026-11-03 最后一轮完整复测：NSA-1B、16 层、BF16、batch=1、hot=256，
History=4K/8K/16K、Extend=128/256。两轮共 30 个进程配置、1800 个计时请求；
48 组完整 hidden 输出比较完全一致。

| 对照 | 本方案 E2E 几何平均加速 | Worker 几何平均加速 |
| --- | ---: | ---: |
| RDMA 整批取回（注册复用） | 2.828× | 3.432× |
| CXL 整批读取 | 1.397× | 1.488× |
| CXL 按块流水（不共享） | 1.439× | 1.551× |

共享计划的软件计数读取载荷合计减少 87.82%，该数值不是 PCIe/CXL 物理链路流量。
不共享的 TMA 流水并未稳定超过 CXL 整批读取，因此不能把整体收益全部归因于 TMA。
逐点延迟、计时边界和归档范围见 [结果说明](results/final/README.md)。

## 检查代码

```bash
uv run --locked --group nsa pytest -m 'not gpu and not rdma'
uv run --locked ruff check cxl_recsys scripts tests
```

发布目录已通过 387 项 CPU 检查和代码 lint，并以 H4K/E128 完成五组硬件检查：
四组完整 hidden 对照完全一致，执行路径、读取计数、DAX 恢复和 RDMA 会话释放通过。
该小规模检查用于验证整理后的代码，性能结论仍采用上面的完整矩阵。

正式模型使用 **Query 分组 1、ready 窗口 1**。NSA 目录保留的 `experimental/` 是关联
算子扩展，不属于通过完整模型验收的方案；`backends/` 还包含主路径实际依赖的搬运
基础实现。第三方来源和许可见 [third_party/README.md](third_party/README.md)。
