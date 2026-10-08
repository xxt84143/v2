# v2：SWAN 代理模型的矩阵输入与圆岛实验

默认实验使用 0.5°×0.5°窗口、圆岛和浅水裙边。
原始波浪为 `wave(3,2,2)`，完整九点风为 `wind(2,3,3)`；不再依靠位置命名列或五点补角。
设计、风浪关系、矩阵顺序和样本划分见 [MATRIX_ISLAND_UPDATE.md](MATRIX_ISLAND_UPDATE.md)。

## 上传服务器

直接上传 [TOY_SERVER](TOY_SERVER/README.md)，在其中配置 SWAN 路径。
包内的生成、运行、ERA5 请求和训练数据转换脚本各有独立入口，通过文件衔接。
默认圆岛生成独立于 ERA5 和 GEBCO；ERA5 请求作为独立功能保留，并申请 mp1 与 mwp。
本次仅做静态检查，未生成 case、下载数据、执行测试、运行 SWAN 或训练。

## 圆岛开发入口

[island_toy/README.md](island_toy/README.md) 提供本地开发目录的接口说明。
沿用 100 m 平底、圆岛、smoothstep 浅水裙边及新岛/新海况/两者皆新的划分。
默认 608 cases、1 km。服务器首例验收后再扩大计算规模。

训练仍使用外部 U-Net（MONAI BasicUNet 或 segmentation_models.pytorch），
配置 `paths.model_repo` 与实际环境一致，或安装所选模型库。
Torch、CUDA 和模型库根据服务器环境单独配置；SWAN 样本生成包不依赖它们。
模型通道改为 10，原始矩阵保留在 shard 中；转换器当前监督 Hs，其他 bulk 参数保留在 SWAN 原始输出。

```text
python train_v2.py --profile 1km --data TOY_SERVER/dataset/1km --output training/island_matrix --device cuda
python evaluate_v2.py --data TOY_SERVER/dataset/1km --checkpoint training/island_matrix/best.pt --split validation --device cuda
```

上述命令供完成服务器计算后使用。本次未执行；旧数据和 checkpoint 需重新生成和训练。

## 保留的 ERA5/真实地形路径

`build_v2.py`、`run_v2_swan.py`、`prepare_v2_dataset.py` 仍保留旧 ERA5/GEBCO 工作流，
但 forcing 已改为矩阵，并使用完整九点风。输入路径在根目录 `config.json` 中。
`build_v2.py all` 生成 `index/forcings/...npz`；索引保存 `forcing_file` 路径。
`run_v2_swan.py` 仍依赖原工程的外部 `SWAN_YEARLY/swan_batch.py`，该依赖不包含在公开仓库；
新的圆岛 runner 与 TOY_SERVER 已独立于这个旧接口。
旧 ERA5 runner 的周期处理需在启用该工作流时结合源字段另行核对；本轮圆岛统一使用 Tp/PEAK。

## 文件处理依赖与检查

根目录 `requirements.txt` 用于生成、转换和可视化，不含 Torch/模型库。
测试使用合成夹具，独立于原项目大数据；本轮只更新测试源码，未执行测试。
以后可以在根目录、island_toy、TOY_SERVER 中分别运行 `python -m unittest discover -s tests -v`。

## 批量 ERA5 下载

`TOY_SERVER/request_era5.py` 支持单变量分块、最多五个在途任务、任务 ID 恢复和逐块合并。
详见 [下载说明](TOY_SERVER/ERA5_DOWNLOAD.md)。本次更新仅做静态检查，未发起 CDS 请求。
