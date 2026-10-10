# v2：SWAN 海岛实验与机器学习代理

在 0.5°×0.5° 窗口中构建人工圆岛和浅水裙边，使用 stationary SWAN 生成标签，再训练 U-Net 预测整张细网格的 Hs。
SWAN 网格为 **15 arc-sec，121×121 节点**；训练输入补零至 128×128。

## 已完成的基线

2026-10-09 的交付数据有 **474 个样本**：train 227、validation 121、test 126。
MONAI BasicUNet 已训练 80 轮，按 validation 选择第 65 轮权重；独立 FP32 验证 RMSE 为 **0.48424 m**。
同一验证集的四角 Hs 插值基线 RMSE 为 **0.33171 m**，因此当前模型还有明显改进空间。
test 尚未推理评估。

[基线结果索引](results/v2_baseline_20261009/README.md) 包含模型权重、训练曲线、121 个验证样本的预测、评估图与科学检查报告。
原始标签仍来自实际 SWAN 计算，收敛信息按 `record_only` 保存。
完整生成计划为 608 个 case；当前缺少 134 个纯风生浪 case，覆盖问题见归档报告。

## 两个独立上传包

| 包 | 用途 | 下一步 |
|---|---|---|
| [TOY_SERVER](TOY_SERVER/README.md) | 生成海岛 case、运行 SWAN、转换数据、可视化检查；独立请求 ERA5 bulk 数据 | 配置 SWAN 路径，按 README 操作 |
| [TRAIN_SERVER](TRAIN_SERVER/README.md) | 检查数据、训练、续训与评估 | 上传转换后的数据，使用已有可用 Torch 环境 |

入口脚本通过文件衔接。SWAN 包不依赖 Torch，训练包不依赖 SWAN 或 CDS。
数据集和 Python 环境不上传 Git；训练结果以版本目录归档。

## 输入与输出

- `wave(3,2,2)`：四角 Hs、Tp、波向 FROM。
- `wind(2,3,3)`：九点 u10、v10，直接输入九点矩阵。
- 矩阵行从南到北，列从西到东；通道顺序由固定数组约定。
- 基线网络输入依次为 `log_depth, wet_mask, x, y, wave_hs, wave_tp, wave_dir_sin, wave_dir_cos, wind_u10, wind_v10`。
- 当前监督输出为 Hs；其他 bulk 参数保存在原始 SWAN 输出，尚未加入训练损失。

矩阵与海况设计见 [MATRIX_ISLAND_UPDATE.md](MATRIX_ISLAND_UPDATE.md)。
海岛开发入口见 [island_toy/README.md](island_toy/README.md)。

## 新实验

[`v2.1` 分支](https://github.com/xxt84143/v2/tree/v2.1) 将第一个地形通道替换为有限水深的 `h/λ`。
它保持其余输入、标签、样本划分和训练设置一致，用于验证相对水深能否改善泛化。
原基线模型与结果在本分支保留。

## ERA5 与保留的开发入口

`TOY_SERVER/request_era5.py` 请求 ERA5 single levels 的风与波浪 bulk 参数，支持单变量分块、最多五个在途请求、任务恢复和合并。
操作见 [ERA5_DOWNLOAD.md](TOY_SERVER/ERA5_DOWNLOAD.md)。认证文件放在用户主目录。
原 `ERA5Downloading.py` 是波谱下载器，继续保留。

根目录 `build_v2.py`、`run_v2_swan.py`、`prepare_v2_dataset.py` 保留真实地形开发路径；旧 runner 需要仓库外的 `SWAN_YEARLY/swan_batch.py`。
海岛实验和独立上传包使用各自的 runner。

## 本地检查

2026-10-10：根目录 10 项、island_toy 4 项、TOY_SERVER 13 项检查通过。
TRAIN_SERVER 的验证范围和实际基线运行情况见其 README 与 [LOCAL_VALIDATION.md](TRAIN_SERVER/LOCAL_VALIDATION.md)。
运行结果的评估图可直接打开 PNG；海岛 case 检查可生成离线 HTML，或通过 VS Code 转发端口访问。
