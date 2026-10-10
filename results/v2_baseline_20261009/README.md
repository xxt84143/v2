# v2 基线：2026-10-09 圆岛 Hs 训练与验证结果

这是 `log_depth` 地形通道版本的已有结果，作为 v2.1 对照实验的基线。

| 项目 | 结果 |
|---|---|
| 数据 | 474 个实际交付样本：train 227、validation 121、test 126 |
| 训练 | 80 轮，MONAI BasicUNet，10 个输入通道，Hs 单通道输出 |
| 选择 | 根据 validation 选择第 65 轮 `best.pt` |
| FP32 验证 RMSE | 0.484239 m |
| 四角 Hs 插值基线 RMSE | 0.331709 m |
| 新岛 / 新海况 / 两者皆新 RMSE | 0.109572 / 0.574728 / 0.595650 m |
| test | 尚未运行推理评估 |

## 文件

- [完整训练结果](toy_runs/hs_unet/)：`best.pt`、`last.pt`、配置、80 轮记录、数据检查与 summary。
- [验证结果](toy_runs/hs_unet/evaluation/validation/report.md)：逐 case / 泛化组指标、PNG 图与 121 份预测 `.npz`。
- [短跑结果](toy_runs/smoke/)：两轮流程验收记录。
- [深入解读](SWAN_SURROGATE_DEEP_REVIEW.md)：数据覆盖、误差分解与模型限制。
- [复算与审计材料](reports/swan_surrogate_review/)：审计脚本、JSON、CSV、诊断图。

![训练与误差诊断](reports/swan_surrogate_review/diagnostic_overview.png)

训练数据压缩包 `dataset_gebco15s.tar.gz` 保留在本地，通过独立文件传输交付。
完整数据内容签名为 `b9758c251cccbc9989c0d94842794ad4ced2b9ea64f19ebf38b6c5ef3597d69a`。
权重与验证预测保持原始内容，没有重新训练或修改旧 checkpoint。
原报告中的本地目录描述是审计时的定位信息；此目录保存的是其公开可复查副本。

这次结果证明了当前人工岛形族中的一部分空间泛化能力。
整体验证成绩仍弱于简单边界插值基线，新海况覆盖和幅值误差仍是主要限制。
