# 配置与运行

## 配置入口

命令入口为 `python -m cxl_recsys.cli`，支持 `worker`、`scheduler`、`pool`、
`build-kernels`。`config.yml` 提供完整默认值，`--config` 加载局部覆盖，显式命令行
参数最后覆盖。路径以启动时的工作目录为基准，建议统一在仓库根目录执行。

`config.yml` 使用模拟模型与 DRAM；`configs/nsa-tma.yml` 启用真实 NSA-1B、CXL 和
TMA。真实模型需要本地 checkpoint、GPU 和事先编译的 native 库。配置中的
`max_history`、`pages`、`page_size` 应满足模型每层 K/V 的后备容量要求，启动时会校验。

| 选项 | 含义 |
| --- | --- |
| `worker.model_path` | 固定版本 NSA-1B checkpoint 目录 |
| `worker.kernel_dir` | 在当前 TVM FFI 环境编译的库目录 |
| `worker.dax_device/dax_offset/dax_bytes` | 按顺序列出的 DAX 区域；偏移与长度支持 GiB/MiB |
| `worker.rdma_device/rdma_port/gid_index` | verbs 端口；需要和实际网络配置相符 |
| `worker.pool_target` | RDMA Pool 的控制服务地址 |
| `worker.hot` | 每用户每层 HBM 热副本 token 容量，正式实验为 256 |
| `worker.nsa_cross_query_dedup` | 是否合并本次计算内跨 Query 的重复 KV 外读 |
| `worker.nsa_selected_read_order` | 正式 TMA 路径使用 `consumption_depth` |

单独启动 Worker 会使用并修改其配置的后备区域，关闭后释放映射和注册。
备份与恢复既有 DAX 内容是实验运行器的职责，不是普通 Worker 的运行承诺。

## 模拟模型联调

安装根环境并编译默认配置所需的 planner 后，在不同终端启动：

```bash
uv run --locked python -m cxl_recsys.cli build-kernels
uv run --locked python -m cxl_recsys.cli worker
uv run --locked python -m cxl_recsys.cli scheduler \
  --source generated --num-requests 8 --output runs/simulated.json
```

模拟路径仍需 CUDA GPU；它模拟骨干运算，不是 CPU 版 NSA。正式 NSA 运行应使用根
README 的独立 TileLang 环境和对应编译库。

## 独立 RDMA Pool

Pool 与 Worker 是不同进程，gRPC 交换分配和 MR/QP 控制信息，KV 数据由 RC verbs
one-sided READ 取回。正式对照运行器自动启动同机 Pool 并在退出时检查会话释放。
手动联调可在两个终端执行：

```bash
source scripts/nsa_tilelang_environment.sh
"$NSA_PYTHON" -m cxl_recsys.cli pool \
  --listen 127.0.0.1:50052 --rdma-device mlx5_1 --rdma-port 1 --gid-index 1

source scripts/nsa_tilelang_environment.sh
"$NSA_PYTHON" -m cxl_recsys.cli worker --config configs/nsa-tma.yml \
  --max-users 2 --max-history 16384 \
  --backend rdma --pool-target 127.0.0.1:50052 \
  --nsa-selected-backend triton_gr --no-nsa-cross-query-dedup \
  --nsa-selected-read-order physical
```

上述设备名为原实验主机的样例。GPUDirect 注册失败会报错，不会用 CPU 中转掩盖问题。
跨主机部署需要可达的 Pool 控制地址和 verbs 链路；仓库内的性能数据仅验证同机 loopback。

## 计时与正确性

正式实验先建立用户历史，再测量候选 Extend。History prefill、首次 JIT、Graph 捕获和
预热单独记录，不计入 steady/mixed 延迟。E2E 包含 Scheduler 到 Worker 的请求交互；
Worker 延迟取服务端执行字段，两者不能互相替代，也不能当作独立算子耗时。

运行器会将新实验的全部原始材料写入 `--output`，包括冻结源码、请求、逐次响应、
日志和备份恢复凭据；`--save-outputs` 还保存完整 hidden 张量。运行后先比较数值，再
执行分析。只有路径、计数、数值和清理校验通过，汇总才有效。

Git 中的 `results/final` 是精简结果归档。它保留结果与验收证据，完整日志、张量和
冻结源码仍在实验存储目录，不能用精简归档直接重新运行原始张量比较。
