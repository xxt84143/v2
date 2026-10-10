# v2.1：用 h/λ 替换 log_depth 的对照实验

版本：2.1。分支：`v2.1`。基线结果见 [v2 记录](../../results/v2_baseline_20261009/README.md)。

## 改变什么

第一个输入通道由 `log_depth` 改为 `relative_depth = h/λ`。
其余 9 个通道、121×121 原始网格、128×128 padding、Hs 标签、mask、样本身份和划分保持一致。
网络仍为原 MONAI BasicUNet，使用原训练超参数与 seed。

λ 的含义是**输入 Tp 对应的局地线性波长**：

\[
\omega=2\pi/T_p,\qquad \omega^2=gk\tanh(kh),\qquad
\lambda=2\pi/k,\qquad h/\lambda=kh/(2\pi).
\]

四角输入 Tp 先按原规则双线性插值到细网格，再在每个湿点求解色散关系。
数值实现解 `z tanh(z) = ω²h/g`，`z=kh`，使用有界二分法。
陆地和 padding 为 0；相对水深不再按 `depth_cap_m` 截断，也不额外映射到 0–1。
关系依据 [SWAN 技术文档](https://swanmodel.sourceforge.io/download/zip/swantech.pdf) 的线性波传播描述。

这里没有使用 SWAN 输出的 Tp，因此不会把标签信息带入输入。
在复杂波谱中，输入 Tp 不等于实际局地谱峰，本通道是物理特征编码。

## 物理与数学直觉

相同的水深对短波和长波影响不同，`h/λ` 将这一关系直接呈现给网络。
它较小时，波浪运动更容易受海底影响；它较大时，更接近深水传播。
原输入已有 h 与 Tp，因此这不是增加新的观测信息，而是把它们的非线性关系预先计算出来。
能否改善泛化要以同一批 validation 的结果判断。

## 纯风生浪的参考周期

现有 wind_only case 的四角 Hs 全部为 0，其 Tp=8 s 是边界文件的占位值。
这类 case 没有可用于定义入射波长的 Tp。
v2.1 显式采用固定 `zero_boundary_reference_period_s=8` 计算相对水深，
作为可逆的水深编码，并记录在 metadata；它不代表风生成波浪的实际波长。
该编码独立于任意占位 Tp 的更改。风浪生成器与输入 Tp 通道保持原设置。

## 已有数据如何迁移

已有 474 个样本可以直接重编码，不需要重新运行 SWAN：

```bash
python migrate_dataset.py --data /root/autodl-tmp/toy_data/gebco15s --output /root/autodl-tmp/toy_data_v21/gebco15s
python check_dataset.py --data /root/autodl-tmp/toy_data_v21/gebco15s
```

旧 shard 未保存 raw depth 时，迁移脚本反解 `log_depth`，得到原来被截断后的深度。
本次人工圆岛实际水深不超过 100 m，与原 cap 相同，因此适用于当前对照。
若其他数据曾把大于 cap 的真实水深截断，应从原始 case 的 `depth_m` 重新转换；
旧通道无法恢复被丢失的更深地形。
新转换器和迁移结果都保存 `depth_m`，训练数据检查重新求解并核对 `h/λ` 通道。

迁移只修改 `x[0]`，新增 `depth_m`；manifest 按原字节复制，y、mask、wave、wind、raw_shape 保持不变。
新 schema 为 `toy-v2.1-dataset-relative-depth-1`，旧 checkpoint 会因通道、schema 和数据签名不匹配而被拒绝。
v2.1 同时接受旧 baseline 数据，便于复现对照，但正式实验需确认报告中的 `terrain_channel=relative_depth`。

## 训练、评估和判断

```bash
python train.py --data /root/autodl-tmp/toy_data_v21/gebco15s --output /root/autodl-tmp/toy_runs/hs_unet_v21 --epochs 80
python evaluate.py --data /root/autodl-tmp/toy_data_v21/gebco15s --checkpoint /root/autodl-tmp/toy_runs/hs_unet_v21/best.pt --split validation --save-predictions
```

比较整体验证 RMSE、Bias、相对于四角 Hs 插值的 skill，以及 new_island / new_forcing / new_both 三组误差。
两次训练均只根据 validation 选择 checkpoint，test 保留到方案确定后。
单次 seed 对照只能判断本次训练轨迹，不能作为稳定提升的结论；后续应按相同设置比较多个 seed。

当前实验仍使用同一批交付数据。纯风生浪缺失和海况覆盖不足不会因换一个通道自动消失。

## 本地验证与运行记录（2026-10-10）

完整迁移验证见 [migration_verification.json](migration_verification.json)。
474 个样本的其余输入和标签逐字节一致，manifest 完全一致；原地形重建误差最大为 2.29e-5 m。
52 项检查通过，其中训练包有 25 项，覆盖色散关系、地形特征、数据签名、迁移与 CPU 训练／续训／评估。

另外用完整的 474 个样本在 CPU 上短跑两轮，关闭 AMP，验证训练与保存流程。
这次短跑的权重、训练记录和独立 validation 评估归档在
[CPU 短跑结果](../../results/v2.1_cpu_smoke_20261010/README.md)。
CPU 短跑用于检查流程，训练轮数和运行设置不等同于原 80 轮 GPU 基线，不能用于判断精度提升。
80 轮 GPU 对照尚未运行。继续实验时按上面的正式命令使用独立目录，保留原 test。
