# 圆岛实验（矩阵输入版）

本实验沿用原有圆岛和浅水裙边，改为四角波浪矩阵及九点风矩阵。
风浪条件、公式、物理范围和划分见 [本次更新说明](../MATRIX_ISLAND_UPDATE.md)。
默认 32 个岛、20 套海况、608 个 case，15 arc-sec（121×121 节点）；无需 ERA5 或 GEBCO 文件。

## 开发入口

在 `island_toy` 目录执行（以下命令供服务器使用，本次未执行）：

```text
python build_island_toy.py --overwrite
python run_island_toy_swan.py --profile gebco15s check --limit 1
python run_island_toy_swan.py --profile gebco15s prepare --limit 1
python run_island_toy_swan.py --profile gebco15s run --limit 1 --swan /实际路径/swan
```

`build` 同时生成每个岛的 `terrain.png` 与 `terrain.npz`。
forcing 存在 `index/forcings/Fxxx.npz`，数组为 `wave(3,2,2)` 和 `wind(2,3,3)`。
`prepare` 将地形、forcing、SWAN INPUT 全部复制到独立 case；运行入口读取已准备的目录。
岛屿 runner 已独立于外部 SWAN_YEARLY；矩形 ERA5 的旧 runner 仍使用该旧接口。
默认无流，波浪边界 PEAK，四边共享入射海况；`wind.dat` 直接使用九点风。

服务器试算通过后，可去掉 `--limit 1` 准备完整实验并运行。
正常结束且输出完整的 case 标记为 `completed`；收敛只记录，未知或不足也可保存和转换。
`run_status.json` 和数据集索引保留收敛信息；程序错误、超时、缺失或损坏输出仍排除。
已有 case 更新时，在 prepare 中使用 `--overwrite`。

## 训练和评估

```text
python prepare_island_dataset.py --profile gebco15s
cd ..
python train_v2.py --profile gebco15s --data island_toy/dataset/gebco15s --output island_toy/training/gebco15s --device cuda
python evaluate_v2.py --data island_toy/dataset/gebco15s --checkpoint island_toy/training/gebco15s/best.pt --split validation --device cuda
```

转换器生成 10 通道特征和 Hs 标签；SWAN 原始输出仍保留多个 bulk 参数。
旧数据和 checkpoint 不兼容新通道，需要重新转换和训练。
test 应在模型和分析规则确定后再用于最终评估。

## 环境与检查

根目录 `requirements.txt` 包含生成和文件处理依赖；Torch 与 MONAI 根据训练环境单独安装。
本目录开发入口本轮未运行；共享 SWAN 库已通过 TOY_SERVER 的实际 15 arc-sec 计算，
见 [检查记录](../TOY_SERVER/LOCAL_VALIDATION.md)。
离线测试以后可分别从根目录和岛屿目录运行 `python -m unittest discover -s tests -v`。
