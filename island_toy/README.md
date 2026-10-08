# Circular-island toy

这是 v2 的第一阶段 toy，不使用真实 GEBCO 地形：整个 0.5°×0.5°矩形先设为
100 m 平底，再放入一个圆形岛。岛核为陆地；从岛岸到外侧 skirt，水深用
`smoothstep` 从随机的 3–15 m 平滑增加到 100 m。

## 固定实验规模

- 岛屿地形：32 个（train 24 / validation 4 / test 4）；
- 耦合强迫：12 套（train 8 / validation 2 / test 2）；
- train：192 cases；
- validation：88 cases；
- test：88 cases；
- 总计：368 cases。

validation 和 test 都分别包含：

- `new_island`：未见岛屿 + 已见强迫；
- `new_forcing`：已见岛屿 + 未见强迫；
- `new_both`：未见岛屿 + 未见强迫。

因此不会把同一岛屿或同一强迫随机拆到两边后宣称实现了几何泛化。

## 风、流和波浪

每套 forcing 同时提供：

- 单侧 JONSWAP 波浪边界：`Hs/Tp/DIR_from`；
- 全域均匀风：`u10/v10`；
- 全域均匀流：`u/v`。

流速由 `0.05 + 0.025 × wind_speed` 生成，流向相对下风方向取顺风、左右
45°横风或逆风四类。波向也相对风向偏转。这里的“耦合”是为 toy 构造相关的
联合强迫；SWAN 接收规定的流场和风场，但代码没有求解海洋环流或海气反馈，
因此不能把它表述为动力双向耦合。

波浪只从 `DIR_from` 对应的最近一条矩形边进入。这样能观察岛后波影、绕射和
流致折射；不会重现早期四边同时强迫导致的平均化。

## 地形可视化

沿用 v1.3 的生成时可视化操作。每个 `terrains/1km/Txxx/` 目录都会同时生成
`terrain.png`（固定 0--100 m 色标、陆地轮廓和岛心）与 `terrain.npz`，并保留
`grid.nc`、`bottom.dat` 和 `metadata.json`。

## 接口

生成地形、forcing 和固定 case plan：

```powershell
cd F:\TXG\CODE_WNE\TOY\v2\island_toy
conda run -n python310 python build_island_toy.py --overwrite
```

只读检查：

```powershell
conda run -n python310 python run_island_toy_swan.py --profile 1km check
```

准备全部 SWAN 目录：

```powershell
conda run -n python310 python run_island_toy_swan.py --profile 1km prepare
```

运行和状态接口：

```powershell
conda run -n python310 python run_island_toy_swan.py --profile 1km run --workers 1
conda run -n python310 python run_island_toy_swan.py --profile 1km status
```

完成后转换为 16 通道、与 `train_v2.py` 兼容的 shard：

```powershell
conda run -n python310 python prepare_island_dataset.py --profile 1km

cd ..
conda run -n python310 python train_v2.py --profile 1km `
  --data island_toy\dataset\1km `
  --output island_toy\training\1km `
  --device cuda --check-model

conda run -n python310 python train_v2.py --profile 1km `
  --data island_toy\dataset\1km `
  --output island_toy\training\1km `
  --device cuda
```

16 个模型通道为：深度、wet mask、x/y、波浪 Hs/Tp/方向 sin/cos、入射边界
四通道 one-hot、风 u/v、流 u/v。岛屿中心和半径不会作为捷径特征直接输入。

MONAI 已通过 `F:\TXG\MONAI-1.3.2` 源码适配器完成静态构造检查：
`BasicUNet(16 -> 1)` 能保持 64×64 输出尺寸。

## Validation 与最终测试

训练脚本中的 `validation_loss` 仅用于选择 checkpoint，不足以作为正式结果分析。
独立评估入口位于上级目录的 `evaluate_v2.py`。先做只读契约检查：

```powershell
cd F:\TXG\CODE_WNE\TOY\v2
conda run -n python310 python evaluate_v2.py `
  --data island_toy\dataset\1km `
  --checkpoint island_toy\training\1km\best.pt `
  --split validation --device cpu --check
```

正式 validation（用于分析和模型选择）：

```powershell
conda run -n python310 python evaluate_v2.py `
  --data island_toy\dataset\1km `
  --checkpoint island_toy\training\1km\best.pt `
  --output island_toy\evaluation\1km\validation `
  --split validation --device cuda --batch-size 16 `
  --bootstrap-samples 2000 --save-predictions
```

模型、阈值和报告规则冻结后，才单独运行 test：

```powershell
conda run -n python310 python evaluate_v2.py `
  --data island_toy\dataset\1km `
  --checkpoint island_toy\training\1km\best.pt `
  --output island_toy\evaluation\1km\test `
  --split test --device cuda --batch-size 16 `
  --bootstrap-samples 2000 --save-predictions
```

默认评估原始模型输出，不会静默把负值截成零。如需将非负截断作为明确的部署后处理，
增加 `--clip-min-m 0`，并将这次结果保存到不同目录。可选质量门限
`--max-rmse-m`、`--min-r2`、`--min-skill` 在不满足时返回退出码 2。

每次评估会输出：

- `summary.json`：数据/checkpoint 哈希、运行环境、总体指标、分组指标和吞吐；
- `metrics_by_case.csv`：逐 case 物理单位指标；
- `metrics_by_group.csv`：整体及 `new_island/new_forcing/new_both` 指标；
- `report.md`：可直接阅读的主结果与统计定义；
- `parity.png`、`residual_distribution.png`、`rmse_by_generalization.png`、
  `case_rmse.png`、`spatial_examples.png`、`training_history.png`；
- `predictions/*.npz`：仅在指定 `--save-predictions` 时保存。

报告同时给出 pooled-pixel（micro）与逐 case 等权（macro）结果；95% CI 在 case
层面 bootstrap，不把空间像素误当成独立样本。核心指标为 Hs 的 MAE、RMSE、
bias、centered RMSE、scatter index、Pearson r、R²，以及相对“边界 Hs 常数场”
基线的 MSE skill。Hs 接近零时不报告无保护百分比误差。

## 静态检查

```powershell
conda run -n python310 python -m py_compile `
  island_core.py build_island_toy.py run_island_toy_swan.py prepare_island_dataset.py ..\evaluate_v2.py
conda run -n python310 python -m unittest discover -s tests -v
cd ..
conda run -n python310 python -m unittest tests.test_evaluate_v2 -v
```

静态检查验证地形边界为 100 m、岛核存在、裙边向外加深、风流耦合公式、
split 无泄漏，以及 SWAN INPUT 同时包含单侧波浪、WIND 和 CURRENT。
