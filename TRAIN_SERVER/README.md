# TRAIN_SERVER：圆岛实验的独立机器学习训练包

上传整个文件夹或解压 `TRAIN_SERVER.zip`。默认使用 MONAI BasicUNet，
与 v2 当前模型一致：2D U-Net、GroupNorm、10 个输入通道、1 个 Hs 输出通道。
模型库从 pip 安装，配置中没有本机盘符。
验收范围见 [LOCAL_VALIDATION.md](LOCAL_VALIDATION.md)。

## 你现在接下来怎么做（当前 AutoDL 实例）

2026-10-10 当前进度（v2 基线）：

- 多余的 Torch 2.10.0 下载已停止。
- 使用镜像已有的 `/root/miniconda3/bin/python`，Torch 2.12.1+cu130。
- MONAI 1.5.2 已安装，`pip check` 已通过。
- RTX 2080 Ti 上完整 U-Net 的前向、反向、优化器更新已通过。
- 正式训练已完成 80 轮；474 个样本按 227 / 121 / 126 分为 train / validation / test。
- 第 65 轮权重在独立 FP32 validation 上的 RMSE 为 0.48424 m，test 尚未评估。
- 结果见仓库 `results/v2_baseline_20261009/`；下面的命令用于新服务器复现。

### 第一步：把当前终端切回已验证的解释器

如果提示符前有 `(.venv)`，执行：

```bash
deactivate
```

然后执行：

```bash
cd ~/TRAIN_SERVER
python -c "import sys,torch; print(sys.executable); print(torch.__version__, torch.cuda.is_available())"
```

应显示 `/root/miniconda3/bin/python`、`2.12.1+cu130` 和 `True`。
VS Code 使用“Python: Select Interpreter”，选择 `/root/miniconda3/bin/python`。
当前实例已经装好依赖，可以继续下面的数据步骤。

### 第二步：传输正式数据

在 SWAN 服务器运行第 1 节的转换与打包命令。
将 `dataset_gebco15s.tar.gz` 上传到当前 AutoDL 的 `/root/autodl-tmp/` 数据盘，然后解压：

```bash
mkdir -p /root/autodl-tmp/toy_data
tar -xzf /root/autodl-tmp/dataset_gebco15s.tar.gz -C /root/autodl-tmp/toy_data
```

应得到 `/root/autodl-tmp/toy_data/gebco15s/metadata.json`、`manifest.csv` 和各 `.npz`。
本节把数据与训练结果放在 AutoDL 数据盘，代码仍位于 `/root/TRAIN_SERVER`。
如果传输的是整个 `gebco15s` 文件夹，将它放到 `toy_data/`，可以跳过解压。

### 第三步：检查数据、短跑两轮

```bash
cd ~/TRAIN_SERVER
python check_dataset.py --data /root/autodl-tmp/toy_data/gebco15s
python train.py --data /root/autodl-tmp/toy_data/gebco15s --check-model
python train.py --data /root/autodl-tmp/toy_data/gebco15s --output /root/autodl-tmp/toy_runs/smoke --epochs 2 --num-workers 0
```

检查要求 train、validation、test 三个集合都有样本。
如果目前只转换了一个训练样本，继续完成 SWAN 计算与转换，直到三个集合均有样本。
短跑正常结束后，应有 `best.pt`、`last.pt`、`history.csv` 和 `summary.json`。
`smoke` 是一次短跑的独立目录；重复短跑使用新名称，如 `smoke2`。

### 第四步：正式训练并保存日志

```bash
mkdir -p /root/autodl-tmp/toy_runs
cd ~/TRAIN_SERVER
nohup /root/miniconda3/bin/python -u train.py --data /root/autodl-tmp/toy_data/gebco15s --output /root/autodl-tmp/toy_runs/hs_unet > /root/autodl-tmp/toy_runs/hs_unet.log 2>&1 < /dev/null &
tail -f /root/autodl-tmp/toy_runs/hs_unet.log
```

默认训练 80 轮。`tail -f` 用于查看日志，按 `Ctrl+C` 退出日志查看，后台训练会继续。
如果遇到显存不足，新实验降低 `--batch-size`，例如 4。

中断后在同一目录续训：

```bash
python train.py --data /root/autodl-tmp/toy_data/gebco15s --output /root/autodl-tmp/toy_runs/hs_unet --resume /root/autodl-tmp/toy_runs/hs_unet/last.pt --epochs 80
```

若原实验修改了 batch size 等参数，续训时保持这些参数一致。

### 第五步：评估与下载结果

```bash
python evaluate.py --data /root/autodl-tmp/toy_data/gebco15s --checkpoint /root/autodl-tmp/toy_runs/hs_unet/best.pt --split validation --save-predictions
```

评估图、指标和预测位于 `/root/autodl-tmp/toy_runs/hs_unet/evaluation/validation/`。
下载该文件夹即可在本机查看 PNG、CSV 和 Markdown 报告。
确定模型后，将 `--split validation` 改为 `--split test` 做最终评估。

## 1. 两台服务器通过数据文件衔接

SWAN 服务器在 `TOY_SERVER` 中计算并转换：

```bash
cd ~/TOY_SERVER
source .venv/bin/activate
python prepare_dataset.py --profile gebco15s
tar -czf dataset_gebco15s.tar.gz -C dataset gebco15s
```

转换器会检查已保存结果的输入、输出校验和；未收敛结果按当前策略保留并记录。
转换时应等目标 case 的运行结束，避免读到正在写入的状态。
转换器要求新输出目录；若先前只转换了部分样本，应使用 `--output dataset_full/gebco15s`，
相应打包命令改为 `tar -czf dataset_gebco15s.tar.gz -C dataset_full gebco15s`。

将 `dataset_gebco15s.tar.gz` 下载到本机，再上传到训练服务器的 `TRAIN_SERVER`。
训练服务器执行：

```bash
cd ~/TRAIN_SERVER
mkdir -p dataset
tar -xzf dataset_gebco15s.tar.gz -C dataset
```

最终必须是 `dataset/gebco15s/metadata.json`、`manifest.csv` 和全部相对路径指向的 `.npz`。
保持 manifest 的 train / validation / test 与泛化组划分，训练端不会重新随机分组。
数据包只需要转换后的文件；数据规模以实际成功转换数量为准，完整默认计划是 608 个。
本包不附带正式数据。现有独立交付的压缩包有 474 个样本，包含三个划分；生成计划为 608 个 case。

## 2. 先装环境，再占用 GPU 训练

建议 Linux、Python 3.10–3.12。先检查云端镜像已有的 PyTorch，避免重复下载几个 GB 的 CUDA 依赖：

```bash
python -c "import sys,torch; print(sys.executable); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvidia-smi
```

如果镜像已有可用 Torch，可以在该解释器中安装剩余依赖。
`requirements.txt` 保留已满足 MONAI 依赖条件的 Torch，不再强制降级到 2.10.0：

```bash
cd ~/TRAIN_SERVER
python -m pip install -r requirements.txt
python -m pip check
python check_environment.py --output environment_check.json
python -m pip freeze > installed_requirements.txt
```

当前 AutoDL 实例的镜像解释器为 `/root/miniconda3/bin/python`，预装 Torch 2.12.1+cu130。
2026-10-09 已验证 RTX 2080 Ti 上 CUDA 前向与反向正常，完整模型检查见验收记录。
如当前终端已激活之前创建的 `.venv`，先执行 `deactivate` 再检查镜像解释器。
可以直接用 `/root/miniconda3/bin/python` 执行以上命令。

只有镜像缺少合适 Torch，或需要单独隔离实验时，才创建新的环境并安装已验收的 2.10.0：

```bash
cd ~/TRAIN_SERVER
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
nvidia-smi
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip check
python check_environment.py --output environment_check.json
python -m pip freeze > installed_requirements.txt
```

这里的 `cu130` 对应 CUDA 13.0 版 PyTorch，需训练服务器驱动和 GPU 支持。
另一台服务器的环境需要单独确认，SWAN 服务器的 CUDA 状态不能替代这项检查。
PyTorch wheel 自带运行所需 CUDA 库，此项目不编译 CUDA 扩展；通常无需额外安装完整 Toolkit。
若现有驱动/GPU 适合 CUDA 12.8，可按官方命令将索引改为 `cu128`，版本仍保持 `torch==2.10.0`。
Torch 2.10.0 是本包首次 CPU 验收的版本，并非训练算法必须使用的版本。
保留云端镜像的其他版本后，需要执行 `check_environment.py` 验证完整模型。
安装命令见
[PyTorch 官方版本表](https://pytorch.org/get-started/previous-versions/#v2100)，
模型依赖见 [MONAI 1.5.2 安装文档](https://monai.readthedocs.io/en/1.5.2/installation.html)。

如果使用独立 venv，新开 SSH 终端后重新执行 `source ~/TRAIN_SERVER/.venv/bin/activate`。
如果使用当前 AutoDL 镜像，VS Code 远程解释器选择 `/root/miniconda3/bin/python`；
使用独立 venv 时选择该环境的实际绝对路径。

## 3. 先检查，再短跑，再完整训练

完整数据检查不依赖 Torch，也可在 SWAN 服务器或 CPU 上执行：

```bash
python check_dataset.py --data dataset/gebco15s --output dataset_check.json
python train.py --check-model
python train.py --epochs 2 --output runs/smoke --num-workers 0
python train.py --output runs/hs_unet
```

短跑使用独立输出目录，确认读取数据、损失、保存 checkpoint 和验证流程。
默认 80 epochs、batch size 8、AdamW、学习率 3e-4、CUDA AMP；修改 `config.json` 或命令行参数。
正式训练默认要求 CUDA；CPU 检查可显式使用 `--device cpu --no-amp`。
显存不足时首先减小 `--batch-size`，不要改变样本空间网格。

`check_dataset.py` 会扫描所有 shard，检查数据完整性、形状、浮点数、mask、padding、
原始风浪矩阵、15 arc-sec 的 121×121 网格，以及新岛/新海况划分中的数据泄漏。
数据内容 SHA256 写入检查报告和 checkpoint，以便发现传输缺漏和续训期间的数据变化。
若只是检查一次部分交付，可加 `--allow-incomplete-splits`；训练入口仍要求三个集合非空。
收敛信息只计数，不作为训练筛选条件。

每个 epoch 保存 `last.pt`、`history.csv` 和 `summary.json`，验证集变好时保存 `best.pt`。
`last.pt` 包含优化器与 AMP scaler；日志写入和 checkpoint 替换使用临时文件。
每轮按固定 seed 重建随机顺序，便于复现实验；不同 GPU 的浮点执行可能产生小差异。
训练过程只使用 train 和 validation。test 在独立评估步骤运行。

SSH 断线时，可以在服务器上用 `tmux` 保持训练：

```bash
tmux new -s toy_train
# 使用独立 venv 时才需要 source ~/TRAIN_SERVER/.venv/bin/activate
cd ~/TRAIN_SERVER
python -u train.py --output runs/hs_unet > train.log 2>&1
```

用 `Ctrl+B`，再按 `D` 离开会话；`tmux attach -t toy_train` 返回。
在另一个终端执行 `tail -f ~/TRAIN_SERVER/train.log` 查看进度。
如果服务器没有 tmux，可用 `nohup python -u train.py --output runs/hs_unet > train.log 2>&1 < /dev/null &`。

## 4. 断点续训和评估

```bash
python train.py --output runs/hs_unet --resume runs/hs_unet/last.pt --epochs 120
python evaluate.py --data dataset/gebco15s --checkpoint runs/hs_unet/best.pt --split validation --save-predictions
python evaluate.py --data dataset/gebco15s --checkpoint runs/hs_unet/best.pt --split test --save-predictions
```

续训的 `--epochs` 是总轮数；模型、数据、batch size、学习率、seed、AMP 设置需与原实验一致。
续训回到原输出目录，完整新实验使用新目录，避免覆盖已有模型。
验证集用于调整模型；确定模型后再运行 test。

评估恢复米制 Hs，输出 MAE、RMSE、Bias、R²、逐 case 和逐泛化组指标、预测图与训练曲线。
同时比较“将四角边界 Hs 双线性插值到细网格”的简单基线；纯风生浪边界为零，评估报告会显示这一点。
训练/验证损失按全部有效海水格点平均；评估同时给出按像素统计与按 case 等权统计。
默认保留原始模型输出，包括负 Hs，并统计负值比例；如需截断可明确加 `--clip-min-m 0`。
评估产生的 PNG 与 `.npz` 可以下载查看，不需要在服务器上运行浏览器。

## 数据与目标约定

原始输入仍为 `wave(3,2,2)`：Hs、Tp、波向 FROM；`wind(2,3,3)`：u10、v10。
空间行从南到北，列从西到东。转换后的 `x(10,128,128)` 按顺序为：
log_depth、wet_mask、x、y、wave_hs、wave_tp、wave_dir_sin、wave_dir_cos、wind_u10、wind_v10。
原始有效网格为 121×121，其余是零 padding。模型直接读取已转换特征，不再次归一化或改变深度特征。
`y(1,128,128)` 是 Hs / hs_scale_m，`mask` 排除陆地与 padding。

当前标签仅有 Hs。Tp、平均周期、波向与方向展宽仍保留在 SWAN 原始结果中；
多参数训练需先扩展转换器的数据目标和各参数损失，不能直接把本包输出通道数调大。

## 文件职责

| 文件 | 职责 |
|---|---|
| `config.json` | 数据与输出路径、模型宽度、训练超参数 |
| `check_environment.py` | 执行一次真实前向、反向与优化器更新 |
| `check_dataset.py` | 检查交付数据 |
| `train.py` | 训练与断点续训 |
| `evaluate.py` | 加载 checkpoint，评估并输出图表 |
| `training_core.py` | 配置与数据契约库 |
| `model_factory.py` | MONAI U-Net 构造 |
| `tests/` | 数据约定与训练/续训/评估回归检查 |

入口脚本互不导入，计算服务器与训练服务器通过文件衔接。
评估代码基于 v2 的 `evaluate_v2.py` 整理，训练端无需安装 SWAN、ERA5/CDS 客户端或访问开发仓库。
