# v2.1：有限水深 h/λ 地形通道实验

本版本将 U-Net 第一个输入通道由 `log_depth` 改为 **`relative_depth = h/λ`**。
λ 根据四角输入 Tp 和局地水深，通过线性波色散关系计算：

$$
\omega = 2\pi/T_p,\quad \omega^2=gk\tanh(kh),\quad h/\lambda=kh/(2\pi).
$$

相同水深对短波与长波的影响不同；这个通道把水深与周期的关系直接呈现给网络。
只使用输入信息，不使用 SWAN 输出周期。
完整说明见 [实验方案](experiments/v2.1/EXPERIMENT.md)。

## 本次状态

- 原 80 轮基线及其权重、验证图、预测和检查报告已归档到 [基线结果](results/v2_baseline_20261009/README.md)。
- 已对全部 474 个样本完成重编码检查：其余 9 个通道、标签和原始风浪矩阵不变，manifest 字节一致。
- 根目录、海岛目录和两个上传包共 52 项检查通过，包含色散关系、数据迁移和 CPU 训练／续训／评估检查。
- 完整数据的两轮 CPU 短跑和 121 个 validation case 的独立评估已完成，见 [短跑结果](results/v2.1_cpu_smoke_20261010/README.md)。
- 新的 80 轮 GPU 对照训练尚未运行；当前不能判断模型精度是否提升。

## 接下来如何试验

上传本分支的 [TRAIN_SERVER](TRAIN_SERVER/README.md)，使用原数据生成新的相对水深数据目录：

```bash
cd ~/TRAIN_SERVER_v21
python migrate_dataset.py --data /root/autodl-tmp/toy_data/gebco15s --output /root/autodl-tmp/toy_data_v21/gebco15s
python check_dataset.py --data /root/autodl-tmp/toy_data_v21/gebco15s
python train.py --data /root/autodl-tmp/toy_data_v21/gebco15s --output /root/autodl-tmp/toy_runs/hs_unet_v21 --epochs 80
python evaluate.py --data /root/autodl-tmp/toy_data_v21/gebco15s --checkpoint /root/autodl-tmp/toy_runs/hs_unet_v21/best.pt --split validation --save-predictions
```

当前 AutoDL 已验证的解释器为 `/root/miniconda3/bin/python`；使用它时可以将命令中的 `python` 替换为该绝对路径。
原数据、基线权重和 test 保留；新实验使用新输出目录。
详细设置、比较指标和迁移限制见 [实验方案](experiments/v2.1/EXPERIMENT.md)。

## 数据与模型约定

| 项目 | 约定 |
|---|---|
| SWAN | stationary，人工圆岛，15 arc-sec，121×121 节点 |
| 原始波浪输入 | `wave(3,2,2)`：Hs、Tp、波向 FROM |
| 原始风输入 | `wind(2,3,3)`：直接使用九点 u10、v10 |
| 网络输入 | 10×128×128；第一个通道为 h/λ，其余顺序沿用 v2 |
| 输出 | 当前仍为 Hs；其他 bulk 参数保存在原始 SWAN 文件 |
| 划分 | train 227、validation 121、test 126，保持原 manifest |

新转换器保留原始 `depth_m`，地形特征不再按 `depth_cap_m` 截断。
旧数据若曾截断超过 cap 的真实水深，不能从旧通道恢复，应从原 case 重新转换。
当前人工海岛水深不超过原 cap=100 m；恢复误差最大约 0.000023 m。

纯风生浪输入 Hs 全零时没有入射波长；采用明确记录的 **8 秒参考周期**编码水深。
这个参考 λ 不代表实际风生浪波长。其余样本使用输入 Tp。

## 上传包与其他入口

[TOY_SERVER](TOY_SERVER/README.md) 同步支持新通道；只需重新转换已有 case，无需为这次特征实验重跑 SWAN。
[TRAIN_SERVER](TRAIN_SERVER/README.md) 可读取两个版本的数据，检查 schema、通道与数据签名。
旧基线 checkpoint 不能续训新的相对水深数据。

ERA5 bulk 下载入口继续独立保留，原波谱下载器 `ERA5Downloading.py` 保持原文件。
矩阵与海况设计见 [MATRIX_ISLAND_UPDATE.md](MATRIX_ISLAND_UPDATE.md)。
