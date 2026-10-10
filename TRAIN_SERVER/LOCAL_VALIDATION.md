# 训练包验收记录

日期：2026-10-09。

## 已验证

| 检查 | 结果 |
|---|---|
| 独立环境依赖安装和 `pip check` | 通过 |
| MONAI BasicUNet 前向、反向、优化器更新 | 通过，输入 1×10×128×128，输出 1×1×128×128 |
| 模型参数量 | 497,201 |
| 数据契约与评估、训练回归检查 | 14 项全部通过 |
| 完整训练 → 保存 → 断点续训 | 通过 |
| 续训与连续训练对比 | CPU 上第二轮所有模型权重逐项相同 |
| test 评估、米制指标、预测保存及 PNG 绘图 | 通过 |
| 数据/归一化修改后的续训与评估 | 正确拒绝不一致的 checkpoint |
| 现有真实 15 arc-sec shard 的完整内容检查 | 通过，121×121 有效网格，128×128 padding |
| 现有真实数据能否正式训练 | 本机仅 1 个 train 样本，缺少 validation/test，正确标记不具备训练条件 |

测试环境：Windows 11、Python 3.12.14、PyTorch 2.10.0+cpu、MONAI 1.5.2、
NumPy 2.5.3、Matplotlib 3.11.2。
解释器由 DTP_env 的 Python 创建独立 venv，依赖使用 pip wheels，未修改 DTP_env。

最初共享 Conda site-packages 的测试环境在评估画图时出现 OpenMP DLL 冲突。
使用默认隔离的 venv 后，完整安装与全部检查通过。
README 同时提供独立环境方式与保留已验证云端镜像环境的方式。

训练回归使用 4 个合成 fixture，验证程序行为，不用于模型性能结论。
评估基线继承 v2 的四角 Hs 双线性插值特征，保留原始预测和负值统计。
收敛为 false 的样本保留并计数，陆地与 padding 不参与损失。

## 当前 AutoDL 实例的 GPU 验证

2026-10-09，Ubuntu 22.04、Python 3.12.3、解释器 `/root/miniconda3/bin/python`：

- 镜像预装 Torch 2.12.1+cu130，CUDA runtime 13.0。
- GPU 为 RTX 2080 Ti，计算能力 7.5；驱动 580.105.08。
- `torch.cuda.is_available()` 为 true，基础 CUDA 前向与反向通过。
- 安装 MONAI 1.5.2 后 `pip check` 通过。
- `check_environment.py` 在 CUDA 上完成完整 BasicUNet 前向、反向及 AdamW 更新。
- 参数量 497,201；单样本检查时 peak allocated 25.49 MiB，此值不是正式训练显存预算。
- 停止另一个 venv 中多余的 Torch 2.10.0 下载，保留镜像 Torch。
- `requirements.txt` 已取消 Torch 2.10.0 的强制固定，保持镜像可用版本。

未对该实例进行正式数据训练，原 14 项回归检查的运行环境仍是上面的 CPU 环境。

## 正式训练前仍需验证

上传后执行 `python check_environment.py`，验证该服务器的 GPU、驱动和 CUDA wheel。
当前 AutoDL 的 GPU 环境已验证；更换实例或解释器后需要重新检查。
没有启动正式数据训练，也没有计算模型的泛化性能。
正式数据需由 SWAN 服务器完成转换并单独传输。

回归检查命令：

```bash
python -m unittest discover -s tests -v
```

测试过程中临时生成的数据、权重与图片由测试临时目录管理，不包含在上传包中。
