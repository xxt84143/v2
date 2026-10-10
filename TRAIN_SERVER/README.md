# TRAIN_SERVER v2.1：h/λ 对照实验训练包

上传整个目录或 `TRAIN_SERVER_v2.1.zip`，建议解压为 `~/TRAIN_SERVER_v21`。
网络、损失与训练设置沿用 v2：MONAI BasicUNet、GroupNorm、10 个输入通道、1 个 Hs 输出。
本版本增加有限水深相对水深通道的数据迁移与检查。

## 当前进度

原模型已完成 80 轮训练，最佳轮次 65；独立 FP32 validation RMSE 为 0.48424 m。
现有数据有 474 个样本，train / validation / test 为 227 / 121 / 126。
原结果已归档。v2.1 的 80 轮 GPU 对照尚未运行。
本地已检查完整数据迁移和 25 项训练包测试；它们验证实现和运行流程，不能证明精度提升。

## 1. 使用服务器已有环境

之前验证的 AutoDL 解释器为 `/root/miniconda3/bin/python`，含可用的 Torch 与 MONAI。
进入新包目录，确认当前实例的环境：

```bash
cd ~/TRAIN_SERVER_v21
/root/miniconda3/bin/python -c "import sys,torch,monai; print(sys.executable); print(torch.__version__,monai.__version__,torch.cuda.is_available())"
```

以下命令里的 `python` 应指向这个已验证的解释器。
若提示符处仍激活旧的空 `.venv`，先 `deactivate`，或直接使用上面的 Python 绝对路径。
新服务器先检查环境，再按 `requirements.txt` 安装缺少的依赖：

```bash
python -m pip install -r requirements.txt
python -m pip check
python check_environment.py --output environment_check.json
```

`check_environment.py` 会检查完整网络的前向、反向和优化器更新。
VS Code 的 Python 解释器也选择同一路径。

## 2. 原数据迁移到独立目录

服务器原数据若仍在 `/root/autodl-tmp/toy_data/gebco15s`，直接执行：

```bash
python migrate_dataset.py --data /root/autodl-tmp/toy_data/gebco15s --output /root/autodl-tmp/toy_data_v21/gebco15s
python check_dataset.py --data /root/autodl-tmp/toy_data_v21/gebco15s --output dataset_check_v21.json
```

否则先上传 `dataset_gebco15s.tar.gz`，解压到 `toy_data/`。
检查结果应为 474 个样本，`terrain_channel=relative_depth`，三个集合分别为 227、121、126。
迁移仅改 `x[0]`、增加 `depth_m`，原 manifest、标签和其余通道保持一致；不需要 SWAN 或 Torch。
目标目录必须是空目录，原数据保留。

**旧深度特征有恢复限制：**反解只能恢复被旧 cap 截断后的水深。
本批人工圆岛水深不超过 100 m，符合迁移条件。
若新数据含更深且已被截断的水深，应在 SWAN 服务器用新 `prepare_dataset.py` 从原 case 转换。

## 3. 两轮短跑，再做 80 轮对照

```bash
python train.py --data /root/autodl-tmp/toy_data_v21/gebco15s --check-model
python train.py --data /root/autodl-tmp/toy_data_v21/gebco15s --output /root/autodl-tmp/toy_runs/smoke_v21 --epochs 2 --num-workers 0
```

短跑结束应有 `best.pt`、`last.pt`、`history.csv` 和 `summary.json`。
之后在独立目录运行正式对照：

```bash
mkdir -p /root/autodl-tmp/toy_runs
nohup /root/miniconda3/bin/python -u train.py --data /root/autodl-tmp/toy_data_v21/gebco15s --output /root/autodl-tmp/toy_runs/hs_unet_v21 --epochs 80 > /root/autodl-tmp/toy_runs/hs_unet_v21.log 2>&1 < /dev/null &
tail -f /root/autodl-tmp/toy_runs/hs_unet_v21.log
```

`tail -f` 是查看日志；日志停止增加时，结合 `summary.json` 的 `epoch` 判断是否完成。
`Ctrl+C` 结束日志查看，后台训练继续。
重复完整实验选择新输出目录；续训使用该实验自己的 `last.pt`：

```bash
python train.py --data /root/autodl-tmp/toy_data_v21/gebco15s --output /root/autodl-tmp/toy_runs/hs_unet_v21 --resume /root/autodl-tmp/toy_runs/hs_unet_v21/last.pt --epochs 80
```

不能用原 `hs_unet/best.pt` 续训 h/λ 数据；数据签名和通道检查会拒绝这种组合。

## 4. 只评估 validation，下载结果比较

```bash
python evaluate.py --data /root/autodl-tmp/toy_data_v21/gebco15s --checkpoint /root/autodl-tmp/toy_runs/hs_unet_v21/best.pt --split validation --save-predictions
```

下载整个 `hs_unet_v21/`，本地打开 `evaluation/validation/report.md`、PNG 和指标 CSV。
与原模型比较整体 RMSE、Bias、插值基线 skill，以及 new_island / new_forcing / new_both 三组误差。
保持原 seed=20261009、80 轮、batch=8、AdamW、lr=3e-4、weight_decay=1e-4、CUDA AMP 和网络配置。
本次先保留 test，选定方案后再做最终评估。
单个 seed 的改善需由后续多 seed 试验确认。

## 数据约定与文件职责

波浪矩阵为 `wave(3,2,2)`，风为 `wind(2,3,3)`；空间行从南到北，列从西到东。
新 schema 是 `toy-v2.1-dataset-relative-depth-1`；输入顺序为：

```text
relative_depth, wet_mask, x, y, wave_hs, wave_tp,
wave_dir_sin, wave_dir_cos, wind_u10, wind_v10
```

有效网格 121×121，padding 至 128×128，陆地和 padding 的 h/λ 为 0。
`y` 是 Hs / hs_scale_m，`mask` 排除陆地和 padding。
纯风生浪采用 8 秒参考周期编码水深；这个 λ 不代表实际风生浪波长。
旧 `toy-v2-dataset-matrix-2` / `log_depth` 数据仍可读取，适合复现原基线。

| 文件 | 职责 |
|---|---|
| `config.json` | 路径、网络与训练设置；默认指向 relative_depth 数据 |
| `migrate_dataset.py` | 将原数据重编码到新目录 |
| `wave_geometry.py` | 求解有限水深色散关系，计算 h/λ |
| `forcing_arrays.py` | 原始矩阵与特征转换规则 |
| `training_core.py`、`check_dataset.py` | 数据约定、签名与逐样本检查 |
| `model_factory.py`、`train.py` | 构建 U-Net、训练与续训 |
| `evaluate.py` | 独立评估、输出图与逐样本指标 |
| `check_environment.py` | 环境与模型运行检查 |
| `tests/` | 物理关系、迁移、训练／续训／评估检查 |

各入口独立调用，共享计算函数；训练服务器不需要 SWAN 或 ERA5 认证。
收敛信息继续仅记录，不作为样本筛选条件。
