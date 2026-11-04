# 第三方来源

| 目录 | 来源与用途 |
| --- | --- |
| `sglang/` | Git 子模块，固定为 `8ae962021dbb5eb8ee408c578e772c13e2c934fe`；编译 HiSparse/CXL 访存库所需头文件。上游为 Apache-2.0，许可随子模块保留。 |
| `nsa_runtime/` | FLA NSA `bd67af59b90afa34b25f61d2922e612d10dba3bd` 的 serving 适配，保留 MIT LICENSE、来源和改动说明。 |
| `nsa_reference/` | 同一版本 FLA 的参考代码与 MIT LICENSE，用于 GPU 数值回归，不加载训练依赖。 |
| `GR/` | 从 cxldsagr `7ffeaeef1409d685c02c0b9150bdf3ea02a176d4` 提取输入生成所需文件，包含交付侧 NSA prompt 适配。具体文件指纹见 `source.json`。 |

SGLang 通过 `git submodule update --init --depth 1 -- third_party/sglang` 准备；构建时检查
固定 commit。此仓库不会初始化 NOSA、DeepSeek 或完整研究仓库等额外子模块。

GR 仅用于请求生成和历史/候选边界转换，没有引入研究仓库的模型与 CUDA 环境。
保留热度曲线 CSV 及来源信息供该接口使用，不包含绘图程序或图片。提取范围中未发现
独立许可证文件；没有为这部分代码另行授予新的开源许可。

NSA-1B checkpoint 独立从 `zen-E/NSA-1B` 下载，固定 revision 为
`3dc3694d2f873820122b094276ae626a44f00a36`，权重不随本仓库分发。
