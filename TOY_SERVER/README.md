# TOY_SERVER v2.1.1

自包含的圆岛/椭圆、广覆盖单峰海况生成与 SWAN 运行包。无需 ERA5 或真实地形。
固定谱输入与 v2.1 相同，默认有效训练目标 16,384；完整设计见 [V2_1_1.md](V2_1_1.md)。

## 安装

上传整个目录到服务器，安装 Python 3.10+ 和串行/OpenMP SWAN：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

在 config.json 中设置 `paths.swan_executable`，或运行时使用 `--swan /实际路径/swan`。
本阶段使用 CPU 上的 SWAN，不依赖 Torch/CUDA。

## 顺序

```bash
# 规划文件，不生成 case：
python generate_cases.py --plan-only
# 默认准备 20,480 train + 1,536 validation + 1,536 test 候选：
python generate_cases.py
python run_swan.py --check
python run_swan.py --workers 4
python prepare_dataset.py --profile gebco15s --output dataset_v211/gebco15s
```

`--check` 不运行 SWAN。runner 到有效训练目标后停止剩余训练候选，另运行评测候选。
converter 预检有效标签，不足目标报错；`--allow-partial` 仅显式导出诊断用部分数据，记录 partial 标记。
默认收敛仅记录；如配置 `require_converged=true`，未识别或未达标的收敛记录不能填充配额。

修改目标：`python generate_cases.py --target-train-cases 8192`。
修改配置种子、库大小、地形/海况参数时使用新 cases 目录，避免旧案例身份混用。
同一配方下备用不足：`python generate_cases.py --split train --training-candidates 24576`，再运行 `python run_swan.py --split train --workers 4`。
如果初次使用 CLI 改过目标，追加时提供相同 `--target-train-cases` 或先修改配置。

小规模输入诊断：

```bash
python generate_cases.py --limit 1
python run_swan.py --case-id BC00000001 --check
```

`--limit`/`--case-id` runner 调用是诊断子集，不强制其达到整个训练目标。
每个 case 保留 INPUT、底床、九点风、四角波浪、输入 hash、状态与收敛记录；失败与超时不计入有效案例。

## 文件

| 文件 | 职责 |
|---|---|
| config.json | 广覆盖参数、库规模、有效目标与替补预算 |
| coverage_generator.py、island_core.py | 分层条件、圆岛/椭圆、稀疏配对 |
| generate_cases.py | 输入准备、快照、候选清单 |
| effective_cases.py | runner/converter 共用有效案例与目标判据 |
| run_swan.py、swan_runtime.py | 运行、检查、续跑、配额状态 |
| prepare_dataset.py | 预检、目标截取、训练数据转换 |
| inspect_cases.py | 候选与现有结果的离线图像检查 |

模型输入仍为 10×128×128，第一通道 h/λ，原始 wave(3,2,2)/wind(2,3,3)/depth_m 保留在 shard。
数据格式与 v2.1 训练包兼容，数据内容与签名不同，训练使用新的输出目录。

独立 ERA5 下载入口保留，见 [ERA5_DOWNLOAD.md](ERA5_DOWNLOAD.md)，它不会参与广覆盖生成器。
当前版本只做源码/配置静态检查，未运行这些命令，也没有实际生成有效训练标签。
