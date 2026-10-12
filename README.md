# v2.1.1：圆岛/椭圆与广覆盖单峰海况

此目录只保留合成海况的生成、SWAN 运行、数据转换及独立训练/评估。
下载入口、下载配置和相关依赖已移除。旧根目录入口、island_toy 副本与历史结果不再参与当前项目。
原文件已在本机归档至 `../_archive/v2.1.1_before_cleanup_20261012/`；云端历史保留在 Git 提交中。
整理不改变广覆盖采样、地形、谱、特征算法或有效训练目标。

## 目录与依赖

```text
v2.1.1/
├─ README.md                 唯一维护的文字说明
├─ shared/
│  ├─ wave_features.py       矩阵契约、色散关系、10 通道特征
│  └─ io_utils.py            JSON/CSV 读写与文件哈希
├─ TOY_SERVER/
│  ├─ config.json            合成实验、SWAN 和配额配置
│  ├─ project.py             配置、路径与配额规则
│  ├─ generation.py          圆岛/椭圆、海况、稀疏计划
│  ├─ swan_inputs.py         网格、INPUT 和案例序列化
│  ├─ swan_runtime.py        求解执行与输入输出检查
│  ├─ effective_cases.py     有效标签筛选与精确配额
│  ├─ generate_cases.py      准备输入与实验快照
│  ├─ run_swan.py            批量运行、复用、替补和达标停止
│  ├─ prepare_dataset.py     有效案例预检与训练 shard 转换
│  ├─ inspect_cases.py       离线可视化检查
│  └─ tests/                 求解接口/文件契约测试源码
├─ TRAIN_SERVER/
│  ├─ config.json            网络、路径与训练超参数
│  ├─ training_core.py       数据、签名与目标计数校验
│  ├─ model_factory.py       MONAI BasicUNet 构建
│  ├─ train.py               训练和续训
│  ├─ evaluate.py            独立评估、图像和指标
│  ├─ check_dataset.py       数据检查
│  ├─ check_environment.py   环境与模型运行检查
│  ├─ migrate_dataset.py     旧深度特征迁移
│  └─ tests/                 特征、数据和评估测试源码
├─ scripts/
│  ├─ static_check.py        源码/配置/依赖图静态检查
│  └─ package_servers.py     独立服务器 ZIP 打包
└─ checks/                   静态检查和上传包校验 JSON
```

`shared` 不依赖任何服务器模块。生成器依赖配置规则；运行和有效配额逻辑不导入生成器。
求解运行模块不依赖 INPUT 渲染模块，训练端不导入求解端；入口脚本通过文件交换数据，互不导入。
合并了原 island_core/coverage_generator，去掉旧生成分支与循环导入；两端的 forcing_arrays/wave_geometry 统一为一份 shared/wave_features。
JSON 写入统一使用 shared/io_utils；数据清单和状态文件保持原有契约。

源码 checkout 运行时保留完整目录。需要只上传一个服务器时使用生成的 ZIP：打包器会把 shared 和本 README 自动带入该包；不要只复制不含 shared 的服务器源目录。
两个小型 _bootstrap.py 只负责定位 checkout 或 ZIP 内的 shared，不包含业务逻辑。

## 安装与上传

在仓库根目录执行以下静态工具；不需要导入 NumPy、Torch 或项目模块：

```bash
python scripts/static_check.py
python scripts/package_servers.py
```

输出为父目录的 `TOY_SERVER_v2.1.1.zip` 与 `TRAIN_SERVER_v2.1.1.zip`，可用 `--output /目标目录` 修改。
每个 ZIP 都包含一份自动复制的 README、shared 和 CHECKSUMS.sha256；这些副本由脚本生成，不在源目录重复维护。

求解服务器需要 Python 3.10+、串行/OpenMP SWAN、NumPy、Matplotlib：

```bash
cd TOY_SERVER
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

设置 config.json 的 paths.swan_executable，或 runner 使用 `--swan /实际路径/swan`。
生成及求解不依赖 Torch。配置仅包含合成实验，窗口起点无需对齐外部数据网格，仍限制经纬度有效范围及固定 0.5°窗口。

## 数据与物理范围

维持 `wave(3,2,2)` = Hs/Tp/FROM、`wind(2,3,3)` = u10/v10，及 10 通道 U-Net 输入、h/λ 地形特征、Hs 标签。
仍为 stationary、无规定流、JONSWAP gamma=3.3、方向标准差 30°；没有加入可变谱形、交叉浪、双峰谱或新增输入通道。
数据 schema 继续为 `toy-v2.1-dataset-relative-depth-1`，生成器和 metadata 版本为 2.1.1；数据内容签名不同，旧 checkpoint 不用于直接续训新数据。

## 地形扩充

- 原水深/尺度范围保持 100 m 平底、3–8 km 等面积半径、5–12 km 裙边、岸侧 3–15 m。
- 每个 split 内交替生成圆岛和椭圆，默认椭圆长短轴比 1.2–2.5，朝向从东起逆时针 0–180°；半轴 a=r√q、b=r/√q。
- 在满足开放边界净空的可行区域内分层采样岛心位置，包括中心、偏边与偏角区域；不允许岛体或裙边越界。
- 旋转椭圆的包围盒加上裙边与 1 km 净空限制半径的条件上限，采样仍处于原半径范围内。
- 裙边按“从岛心沿射线到岸线的距离”定义，使用原 smoothstep 向 100 m 过渡；这是径向裙边，不是椭圆法向等距裙边。
- 默认地形 train/validation/test=512/64/64，各集合使用独立随机流和 ID。没有引入群岛、礁环、海槽或背景水深变化。

## 海况扩充

`generation.py` 使用 Latin hypercube 分层采样；海况类型交替排列，类型内覆盖全方向。默认 forcing 数 train/validation/test=1024/64/64。

| 类型 | 默认条件 |
|---|---|
| calm | 无风、无入射能量 |
| near_calm | 入射 Hs 0.01–0.2 m，本地风 0–2 m/s |
| swell_no_wind | 无风涌浪 |
| young_windsea | 波龄 0.6–0.9，同向青年风浪 |
| mature_windsea | 波龄 0.9–1.3，同向成熟风浪 |
| swell_weak_wind | 涌浪与 0.5–5 m/s 风 |
| swell_moderate_wind | 涌浪与 5–12 m/s 风 |
| swell_strong_wind | 涌浪与 12–25 m/s 风 |
| wind_only | 零边界能量，1–25 m/s 本地风 |

全局非零入射周期范围 3–20 s，涌浪背景周期 6–20 s，涌浪背景 Hs 0.2–8 m；每个角点 Hs≤8 m、九点风速≤25 m/s。
同向风浪满足 Tp=2πaU/g、Hs=s·gTp²/(2π)，风速、波龄和波陡按条件生成；不是把极值任意组合。
各非零波系满足 Hs/Lp≤0.04；梯度作用后的周期与波高仍检查上限。风浪与入射涌浪的背景夹角连续覆盖 -180–180°。
风速与 Hs 每轴相对梯度最多 15%（两轴合计最多 30%），方向每轴最多 8°。这些条件由现有四角波浪和九点风表达。
纯风生浪及静海的边界 Tp=8 s 是零能量占位值，h/λ 仍使用既有 8 s 参考水深编码。

这些是合成覆盖包络，未拟合气候发生频率，不宣称覆盖全球 95% 海况。空间网格、谱网格及极端场景的数值质量需 Pilot 后确认。

## 案例规模与配置

在 `TOY_SERVER/config.json` 的 `experiment.case_sampling` 中配置：

```json
{
  "target_effective_train_cases": 16384,
  "training_reserve_fraction": 0.25,
  "evaluation_cases_per_group": 512,
  "require_converged": false
}
```

目标作用于每个 profile 的最终训练数据集（其中包含所选 tiles），不是 train/validation/test 总数。默认单 tile、单 profile。
训练候选默认 ceil(target×1.25)=20,480；验证/测试各包含 new_island、new_forcing、new_both 三组，各组 512，共 3,072 个评测候选。
完整默认输入准备规模为 23,552 cases；全部候选通过且训练足额时，转换器导出 16,384 train + 1,536 validation + 1,536 test = 19,456 样本。
评测有效数量按实际结果报告，未通过的评测不会计为有效。当前没有新增 OOD 组。

稀疏配对不展开完整地形×海况乘积。每一轮覆盖较大的池一次并轮转较小池，确保不重复的地形/forcing 对。扩大候选前缀保持已有配对与 ID；独立的 split 随机流、评测组 ID 使仅修改目标/候选数量不会改变评测。
修改岛数、海况数、随机种子、物理配方或收敛要求时必须使用新的 cases 目录；生成器会拒绝在已有快照上混用不同配方。

若 25% 备用不够，可配置 `training_candidate_cases` 或使用 `--training-candidates`，它覆盖 reserve 公式。超过唯一组合容量时明确报错，需增加库大小并使用新目录。

## 服务器运行

以下命令是交付后的执行说明，本次未执行。

```bash
cd TOY_SERVER
# 仅写覆盖规划 JSON，不写 case、也不运行 SWAN：
python generate_cases.py --plan-only
# 完整准备（默认 23,552 个候选）：
python generate_cases.py
python run_swan.py --check
python run_swan.py --workers 4
python prepare_dataset.py --profile gebco15s --output dataset_v211/gebco15s
```

自定义有效训练目标：

```bash
python generate_cases.py --target-train-cases 8192
```

生成器将 CLI 目标保存在 `cases/experiment.json`，runner 和 converter 读取该快照，无需在后续命令重复目标参数。
训练不足时在相同目录追加更长候选前缀：

```bash
python generate_cases.py --split train --training-candidates 24576
python run_swan.py --split train --workers 4
```

已有目录会记住上一次的 CLI 配额；追加候选时无需重复目标。显式 CLI 参数优先，配置文件中被修改的配额项也会生效。
旧版快照没有保存配置基准时，首次续接会优先保留快照目标；需要改变目标可显式使用 `--target-train-cases`。
成功且输入、输出一致的 case 会复用；失败 case 在后续调用重试。达到有效目标后 runner 不再启动其余训练候选，评测候选不消耗训练配额。
### 增量扩充

数量表示希望达到的累计总量，而非本次追加量。在同一个 cases 目录、同一 seed、地形/海况库与物理设置下，依次执行：

```bash
python generate_cases.py --target-train-cases <首次有效训练总量>
python run_swan.py --workers 4
python generate_cases.py --target-train-cases <更大的有效训练总量>
python run_swan.py --workers 4
```

第二次保留已有案例编号、配对、输入与求解结果，只为缺少的案例写入输入；runner 校验后跳过已有成功结果，只计算尚未完成或失败的案例。
日志显示 `created` / `reused` 数量；重复相同目标时完整输入全部复用。减小目标不删除已有文件，清单也不会出现重复 sample ID。
训练候选仍包含配置中的替补比例，因此新增候选数量不一定等于有效训练目标的差值。验证与测试配对保持原样。
`--limit` 同样选择一个稳定的累计候选前缀，可从较小值增加到较大值，但它是输入诊断上限，不改变有效训练目标。
不要同时使用 `--overwrite` 或 runner 的 `--rerun`，它们分别显式重写输入和重算结果。

`--case-id`/`--limit` 是诊断子集，runner 不要求其达到完整训练目标，也不产生足额声明；`--check` 只检查输入及可执行文件。

仅准备一个输入检查示例：

```bash
python generate_cases.py --limit 1
python run_swan.py --case-id BC00000001 --check
```

## 有效案例的定义

`TOY_SERVER/effective_cases.py` 中的 `saved_case` 为 runner 和 converter 提供同一判据：

- 保存状态有效，执行 return_code=0，日志无 SWAN error。
- 输入 hash、sample/case/profile/split/地形/forcing/海况身份一致。
- 输出存在、形状/坐标/顺序正确，湿点 Hs 有限且非负，输入与输出签名匹配。
- 默认沿用 `record_only`：未收敛和收敛未知不因此自动排除；元数据与 manifest 继续保存真实收敛记录。
- `require_converged=true` 时必须有识别出的收敛记录且达标，未达标候选不能填补配额。

“有效”不表示已经证明数值收敛或真实海况准确性。开启严格收敛要求应选择新 cases 目录，以免混淆配额统计。
runner 在 `quota_status.json` 写明选中范围的有效计数、目标与是否达到。候选耗尽仍不足时返回非零状态，不宣称完成目标。
converter 先检查再写 shard：默认不足目标直接报错；足额时按稳定 manifest 顺序只导出目标数量的训练案例，避免备用案例悄悄改变 N。
`--allow-partial` 显式允许小规模诊断导出，metadata 会记录不足和 partial 标记；不能作为默认目标完成的数据集。
训练服务器校验 metadata 的有效计数/目标状态与 manifest 一致。

## 可视化检查

在 TOY_SERVER 中执行：

```bash
python inspect_cases.py --output inspection
```

打开 inspection/index.html；复制报告时同时带上 assets。报告区分规划候选、已准备输入与实际求解结果，不把计划数量当有效标签数量。
源码不内置现有案例或图像，运行前的规划预览不代表求解结果。

## 训练、续训与评估

将转换后的 dataset_v211/gebco15s 单独传到 GPU 服务器。新服务器检查既有 Torch/CUDA，再安装缺少的依赖；已验证过的实例可使用 `/root/miniconda3/bin/python`，具体以当前环境为准。

```bash
cd TRAIN_SERVER
python -m pip install -r requirements.txt
python -m pip check
python check_environment.py --output environment_check.json
python check_dataset.py --data /实际路径/dataset_v211/gebco15s
python train.py --data /实际路径/dataset_v211/gebco15s --output /实际路径/runs/hs_v211 --epochs 80
python evaluate.py --data /实际路径/dataset_v211/gebco15s --checkpoint /实际路径/runs/hs_v211/best.pt --split validation --save-predictions
```

check_environment 和 train 的 --check-model 会执行模型计算，属于运行时检查，和 scripts/static_check.py 不同。
原设置为 BasicUNet、GroupNorm、batch=8、AdamW、lr=3e-4、weight_decay=1e-4、seed=20261009、CUDA AMP。
CLI 可调整 epochs、batch-size、learning-rate、num-workers、device 和 amp；CPU 需显式 `--device cpu`。
输出 best.pt、last.pt、history.csv、summary.json。新实验选择新输出目录，同一实验续训：

```bash
python train.py --data /实际路径/dataset_v211/gebco15s --output /实际路径/runs/hs_v211 --resume /实际路径/runs/hs_v211/last.pt --epochs 80
```

checkpoint 校验数据签名与特征定义，不能用旧数据权重直接续训新数据。
根据 validation 选择模型，保留 test 到方案确定后；评估输出逐案例/泛化组指标、PNG、预测及运行生成的报告。
源码中的单一 README 要求不禁止评估程序为实际结果生成 report.md。

旧 log_depth 数据的迁移功能保留：

```bash
python migrate_dataset.py --data /旧数据目录 --output /新的relative_depth数据目录
```

迁移只重编码地形特征并保存 depth_m，不运行求解器；旧 cap 截断后的深水无法恢复，遇到这类数据需从原 case 重新转换。

## 静态检查与验证范围

```bash
python scripts/static_check.py
```

检查 Python AST/编译、JSON、单一 README、旧文件和依赖移除、默认目标与配额容量、共享模块导入符号、生产依赖图无环及服务器独立性。
本机可用整理前归档时，还比较特征、地形与广覆盖采样函数的 AST，确认合并没有改变算法。
结果保存 checks/static_checks.json；包内容、版本、shared 与校验和验证保存 checks/upload_packages.json。
工具不会 import 项目模块、生成案例、运行单元测试、调用 SWAN 或训练。
本次整理只做静态检查与 ZIP 内容校验；16,384 是运行后要求达到的有效目标，尚未在本次整理中生成标签。
