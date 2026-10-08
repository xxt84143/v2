# v2：0.5° × 0.5° ERA5 局地矩形

v2 不再把整个台湾 polygon 当成一个训练样本。一个空间样本是一个固定的
0.5° × 0.5° 经纬度矩形，其四个顶点严格落在 ERA5 0.5°波浪格点上；四条边
的中点和矩形中心严格落在 ERA5 0.25°风格点上。

## 已固定的数据契约

```text
四角 SW/SE/NE/NW:
  ERA5 swh, mwp, mwd                 4 × (3 个量)
  mwd 在模型中编码为 sin/cos        -> 16 个波浪条件通道

南/东/北/西边中点 + 中心:
  ERA5 u10, v10                     5 × (2 个量)
                                      -> 10 个风条件通道

局地空间通道:
  log(depth), wet mask, x, y          -> 4 个通道

模型输入共 30 通道；目标为 SWAN HSIGN。
```

这些 case 级条件被广播成空间通道，与地形一起送入外部 U-Net。这样地形通过
卷积编码器进入模型，而不是压成一个会丢失海岸/浅滩结构的全局标量。

SWAN 需要规则风网格，因此 `run_v2_swan.py` 以五个已声明风点为唯一信息源，
用 IDW 补齐 3×3 网格的四个角。五个原始点保持精确值，未把 ERA5 的四角风
偷偷加入标签生成过程。

## 两档空间分辨率

| profile | 原始节点 | 模型张量 | 说明 |
|---|---:|---:|---|
| `gebco15s` | 121×121 | 128×128 | 120 个 15″间隔，四角精确；GEBCO 双线性采样到节点 |
| `1km` | 默认块为 57×52 | 64×64 | 按块中心纬度将东西/南北边各自取最接近 1 km 的整数间隔，四角仍精确 |

1 km 档的默认块实际约为 `dx=998 m, dy=993 m`。不同纬度的原始东西向节点数
可能相差少量，但在进入模型时统一向北、向东补零到 16 的倍数，并由 mask
排除补零区。两个 profile 分别训练，不把不同物理分辨率混在一个 batch 中。

GEBCO 原始值位于 15″像元中心，而 SWAN 网格需要把 ERA5 四角包含为节点，
所以 `gebco15s` 保持 15″间隔并插值到含边界的 121×121 节点；它不会创造
比 GEBCO 更细的地形信息。

## 生成目录与检查

默认配置选择台湾东岸近岸块 `E121p5_N23p5`（121.5–122.0°E，
23.5–24.0°N）。先构建 catalog、两档地形网格和 400 个已有 MDA 时刻的 forcing：

```powershell
cd F:\TXG\CODE_WNE\TOY\v2
python build_v2.py all --overwrite
```

`index/tile_catalog.csv` 列出当前 ERA5 文件中四角波浪量在全部小时均有效的
173 个矩形。要换块，只需修改 `config.json` 的 `tiles` 后重建。多个块也可
同时列出；同一 ERA5 时刻在所有块和两个分辨率中始终属于同一个 split，避免
条件泄漏。

在不落盘 case 前先做完整几何/ERA5 检查：

```powershell
python run_v2_swan.py --profile 1km --tile E121p5_N23p5 check --limit 1
python run_v2_swan.py --profile gebco15s --tile E121p5_N23p5 check --limit 1
```

准备与运行：

```powershell
python run_v2_swan.py --profile 1km --tile E121p5_N23p5 prepare
python run_v2_swan.py --profile 1km --tile E121p5_N23p5 run --workers 1
python run_v2_swan.py --profile 1km --tile E121p5_N23p5 status
```

15″档只需把 `--profile` 改为 `gebco15s`。默认近岸块的西侧 ERA5 海陆掩膜与
GEBCO 不一致：ERA5 有波浪值而 GEBCO 角点为陆地。准备器会把该角点控制值
映射到相邻的最近有效开边界；具体映射保存在每个 case 的
`case_metadata.json` 和 `boundary_forcing.csv`，不能在分析时忽略。

当前沿用已有工程的参数化边界：ERA5 `mwp` 作为 SWAN JONSWAP `MEAN` 周期，
并非峰值周期；这也不是 ERA5 二维方向谱。波向沿用 `SET NAUTICAL` 下的
ERA5 mean-wave-direction 约定，正式批量运行前应继续用一个 case 核对传播方向。

## 数据转换与训练

SWAN case 完成后：

```powershell
python prepare_v2_dataset.py --profile 1km
python prepare_v2_dataset.py --profile gebco15s
```

转换器只接收存在 `output/compgrid.tab` 的完成 case，读取 `HSIGN`，排除陆地、
负 sentinel 和补零像素，并写成逐 case NPZ shard。训练/验证/测试的默认比例是
80/10/10，split 由源 `case_id` 的稳定哈希确定。

### 外部 U-Net（需要你下载）

v2 没有内嵌自写 U-Net。默认适配
[Project MONAI](https://github.com/Project-MONAI/MONAI) 的 `BasicUNet`。
当前官方 dev 分支要求较新的 PyTorch，而项目记录的环境是
PyTorch 1.13.1+cu116；建议下载官方 release tag `1.3.2`，它仍覆盖这一代环境，
再把 `config.json -> paths.model_repo` 指向仓库根目录。工作区现有
`F:\TXG\MONAI-dev` 是 dev 分支，不作为旧 CUDA 环境的默认依赖。

另一个可选适配器是热门的
[segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)，
将 `model.backend` 改为 `smp_unet` 即可；同样应由你下载并选择与本机 PyTorch
兼容的版本。这里优先 MONAI，因为 `BasicUNet` 无需图像预训练 backbone，适合
30 通道连续物理场回归。

下载完成后先只检查外部模型接口和张量形状：

```powershell
python train_v2.py --profile 1km --device cuda --check-model
```

再训练：

```powershell
python train_v2.py --profile 1km --device cuda --batch-size 4 --epochs 80
```

训练 checkpoint 只按 validation loss 选择，test 只在选定 checkpoint 后评估。

独立评估接口为 `evaluate_v2.py`。它适用于 `train_v2.py` 产生且通道/schema
一致的数据与 checkpoint；必须显式给出数据和权重路径：

```powershell
python evaluate_v2.py --data dataset\1km --checkpoint training\1km\best.pt `
  --split validation --device cuda --batch-size 16
```

冻结模型与报告规则后，将 `--split validation` 改为 `--split test` 并指定独立
输出目录。完整指标、分组、case-bootstrap 置信区间、基线对照、图表和质量门限
说明见 [`island_toy/README.md`](island_toy/README.md)。

目录中同时保留了本轮开发前已有的单矩形草案：`common.py`、
`make_v2_cases.py`、`rectangle.json`、`swan_batch_config.json`，以及直接位于
`grids/15arcsec`、`grids/1000m` 下的旧网格。这些文件没有被删除，但旧默认块
存在 ERA5 陆点波浪缺测，且没有“五风点唯一输入”和多 tile/split 契约；正式
v2 入口以上文的 `config.json`、`build_v2.py` 和嵌套的
`grids/<profile>/<tile_id>/` 为准。

## 静态回归检查

```powershell
python -m py_compile v2_core.py build_v2.py run_v2_swan.py prepare_v2_dataset.py model_factory.py train_v2.py evaluate_v2.py
python -m unittest discover -s tests -v
```

测试覆盖 ERA5 0.5°/0.25°格点契约、173 个有效矩形、两档尺寸、四角精确性、
五点风场重建、模型 padding 和 case 级 split 稳定性。静态检查不会代替 SWAN
物理结果检查。

## 本轮已完成的 smoke

`C0001`（2024-10-30 21:00 UTC）已经在默认块实际运行，不只是生成 INPUT：

| profile | SWAN 状态 | 本机运行时间 | 模型输入 | 有效目标节点 | Hs 范围 |
|---|---|---:|---:|---:|---:|
| `1km` | completed | 约 4 s | 30×64×64 | 2497 | 0.795–8.432 m |
| `gebco15s` | completed | 约 20 s | 30×128×128 | 12399 | 0.609–8.433 m |

对应的单 case shard 已写入 `dataset/1km` 和 `dataset/gebco15s`。它们只用于
链路验证，单一样本不足以训练；完整训练前仍需准备、运行并转换其余 case。

## 圆形海岛 toy

第一阶段的 100 m 平底、随机圆岛、风流联合强迫实验位于
[`island_toy/README.md`](island_toy/README.md)。它拥有独立的地形、forcing、
case plan 和 SWAN runner，不会混入真实 GEBCO/ERA5 数据集。
