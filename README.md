# v2.1.1：广覆盖海况生成器

在 v2.1 的有限水深 h/λ 特征基础上扩充合成单峰海况，地形只增加圆岛位置覆盖与旋转椭圆。
四角波浪、九点风、10 通道输入、固定 JONSWAP 谱形与 Hs 标签保持原契约。

完整设计、配额定义与服务器命令见 [V2_1_1.md](V2_1_1.md)。

| 默认项 | 数量/范围 |
|---|---|
| 有效训练目标 | 16,384，可通过配置或 `--target-train-cases` 调整 |
| 训练候选 | 20,480，包含 25% 替补 |
| 验证 / 测试候选 | 各 1,536，包含新岛、新海况、两者皆新 |
| 地形 | train/validation/test=512/64/64，圆岛和椭圆 |
| 海况 | train/validation/test=1024/64/64，九类单峰条件 |
| 网格 | 15 arc-sec，121×121 节点；网络 padding 到 128×128 |

## 生产入口

上传整个 [TOY_SERVER](TOY_SERVER/README.md)，生成器不依赖 ERA5 或 GEBCO。

```bash
cd TOY_SERVER
python generate_cases.py --plan-only
python generate_cases.py
python run_swan.py --check
python run_swan.py --workers 4
python prepare_dataset.py --profile gebco15s --output dataset_v211/gebco15s
```

runner 达到有效训练目标后停止剩余训练候选；converter 不足目标会报错，足额时只导出目标数量。
有效案例沿用 record_only 策略，并检查执行、输入输出、签名和湿点 Hs；如需排除未收敛案例，可配置 `require_converged=true`。
[TRAIN_SERVER](TRAIN_SERVER/README.md) 独立训练，新增目标/计数一致性检查。

## 验证状态

本版本只做静态检查，没有执行案例生成、SWAN、ERA5 请求、单元测试或训练。
静态检查命令为 `python scripts/static_check_v211.py`，结果见 [static_checks.json](experiments/v2.1.1/static_checks.json)。
16,384 是运行后要求达到的有效训练目标，本次没有生成这些标签；运行时数值质量需要服务器 Pilot 验证。

## 历史对照

- [v2.1 特征实验](experiments/v2.1/EXPERIMENT.md)
- [v2 基线结果](results/v2_baseline_20261009/README.md)
- [v2.1 CPU 流程验收](results/v2.1_cpu_smoke_20261010/README.md)
- [历史矩阵输入设计](MATRIX_ISLAND_UPDATE.md)
- [岛屿开发入口](island_toy/README.md)：帮助库与配置同步，配额生产流程使用 TOY_SERVER。
