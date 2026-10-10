# v2.1 本地 CPU 两轮短跑

本次使用全部 474 个相对水深样本检查训练、权重保存和独立 validation 评估。
**这是流程检查，80 轮 GPU 对照尚未运行。**

| 项目 | 记录 |
|---|---|
| 时间 | 2026-10-10 |
| 地形通道 | `relative_depth = h/λ`，有限水深色散关系 |
| 数据 | train 227、validation 121、test 126；与基线使用相同样本身份和划分 |
| 网络 | 原 MONAI BasicUNet，10 通道输入，Hs 输出 |
| 运行 | Windows，Torch 2.10.0+cpu，CPU，AMP 关闭，num_workers=0 |
| 训练 | 2 轮，seed=20261009，batch=8，AdamW，lr=3e-4 |
| 最佳轮次 | 第 2 轮，按 validation 选择 |
| 独立 FP32 validation RMSE | 0.486287 m |
| 独立 FP32 validation Bias | 0.000953 m |
| test | 未运行推理评估 |

[训练记录与权重](run/) 包含配置、数据检查、history、summary、best.pt 与 last.pt。
[验证报告](run/evaluation/validation/report.md) 包含 121 个验证 case 的指标、图和预测文件。
[迁移验证](../../experiments/v2.1/migration_verification.json) 记录其他通道和标签保持一致的检查。

这里的两轮 CPU 运行与原来的 80 轮 GPU 运行在轮数、AMP 和执行环境上不同，不能用其误差证明 h/λ 更好或更差。
正式判断仍需按 [实验方案](../../experiments/v2.1/EXPERIMENT.md) 运行相同设置的对照。
