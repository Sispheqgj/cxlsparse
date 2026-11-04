# NSA KV Pool

- 本仓库基于 NSA-1B，实现 GR 场景的 CXL/RDMA KV 后备池与 GPU 取数流水。
- 正式入口为 README.md、docs/RUNNING.md、scripts/kvpool_compare/README.md。
- 正式配置使用 query group=1、ready window=1；experimental 代码不属于已验收方案。
- 新实验输出放在仓库外，不提交日志、权重、编译产物、绘图脚本和历史扫描结果。
- results/final 只保留最终五组实验的精简结果、原始输入和验收报告。
- GPU 实验前检查占用并取得共享实验锁；CXL 实验只使用已预留范围，保留备份恢复逻辑。
- 修改计算或取数路径后，验证完整模型数值、实际执行后端与去重开关，再比较性能。
