# TOY_SERVER：独立圆岛实验上传包

上传整个目录即可运行。默认采用与仓库 `island_toy` 相同的圆岛、浅水裙边和海况定义。
生成海岛样本无需真实地形或 ERA5。九点风直接输入，四角波浪采用 3×2×2 矩阵。
详细设计见同目录 [DESIGN.md](DESIGN.md)。本次仅做静态检查，未运行测试或计算。

## 文件职责

| 文件 | 职责 |
|---|---|
| `config.json` | 圆岛数量、海况范围、网格和 SWAN 路径 |
| `generate_cases.py` | 生成自包含圆岛 case、原始矩阵和 manifest |
| `run_swan.py` | 读取已有 case，执行并检查输出和收敛 |
| `prepare_dataset.py` | 将已完成 case 转为训练 shard |
| `request_era5.py`、`era5_config.json` | 独立 ERA5 请求入口与下载配置 |
| `era5_jobs.py`、`era5_merge.py` | 请求拆分、五槽调度、任务恢复及逐块合并 |
| `island_core.py` | 同源圆岛几何、风浪条件和 split 库 |
| `forcing_arrays.py` | 矩阵契约与模型特征 |
| `swan_inputs.py`、`swan_runtime.py` | SWAN 输入文件和运行结果检查库 |

入口脚本互不导入；case 文件是生成、运行和转换阶段之间的接口。

## 服务器操作顺序

进入上传后的 `TOY_SERVER`，使用 Python 3.10 或更新版本：

```text
python -m pip install -r requirements.txt
```

在 `config.json` 中设置 `paths.swan_executable`，指向实际计算程序或 PATH 上的名称。
当前 runner 直接调用计算程序，适用于串行/OpenMP；MPI 版本需要适合服务器的启动器。
此阶段使用 CPU 上的 SWAN，不依赖 Torch 或 CUDA。

先试一个 case：

```text
python generate_cases.py --profile 1km --limit 1
python run_swan.py --profile 1km --limit 1 --check
python run_swan.py --profile 1km --limit 1
```

这些命令供你在服务器执行，本次没有在本机运行。
`--check` 检查 case 文件和程序路径，不运行 SWAN。
`generate_cases --limit` 是每个窗口和分辨率的 case 上限；runner 的 limit 是筛选后的总上限。
默认完整规模为 608 cases；15 arc-sec 可用 `--profile gebco15s` 单独生成。

首例通过并确认 PRINT 和输出后，再生成全部：

```text
python generate_cases.py
python run_swan.py --check
python run_swan.py --workers 2
python prepare_dataset.py --profile 1km
```

生成目录为 `cases_island/profile/tile/ITxxxxxx/`，每个 case 自带 INPUT、bottom.dat、wind.dat、
swaninit、inputs.npz、case.json，计算结果在 `output/compgrid.tab`。
输出包含 Hs、Tm01、Tm02、Tp、平均/峰值方向、方向展宽、TMM10 和风。
转换器当前生成 Hs 标签，兼容仓库的 `train_v2.py`；其他 bulk 参数仍留在原始 SWAN 文件。

重复执行会复用输入一致的 case。输入变化需要 `generate_cases.py --overwrite`，
已通过 case 有意重算时使用 `run_swan.py --rerun`。修改实验数量时建议使用新的 cases 目录。
只有 `completed` 状态的输出进入数据转换；失败、未收敛、日志未识别的结果保留诊断文件。

## 矩阵方向

`wave(3,2,2)` 通道为 Hs、Tp、波浪 FROM 方向；`wind(2,3,3)` 为 u10、v10。
空间行从南到北，列从西到东。`wind.dat` 先写完整 u 矩阵，再写 v 矩阵，IDLA=3。
九点均为原始输入，角点不由五点插值补出。
机器学习特征放大使用完整九点矩阵；周期和方向不是依靠空间位置名称索引。

## 独立 ERA5 请求

请求入口保留，但圆岛实验不会调用它。数据集为
[ERA5 hourly data on single levels from 1940 to present](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download)，
API 标识 `reanalysis-era5-single-levels`。风 0.25°、波浪 0.5°分别请求。

服务器用户主目录的 `.cdsapirc` 按 [CDS 官方说明](https://cds.climate.copernicus.eu/how-to-api)填写：

```text
url: https://cds.climate.copernicus.eu/api
key: 你的个人token
```

在数据集页面接受使用条款后，可以单独执行：

```text
python request_era5.py --config era5_config.json --dry-run
python request_era5.py --config era5_config.json --download-only
python request_era5.py --config era5_config.json --merge-only
```

输出 `data/wind.nc` 与 `data/waves.nc`；默认波浪字段为 swh、mp1、mwd、mwp。
mp1 是 Tm01，mwp 是 Tm-1,0，二者分别保留。已有旧 waves.nc 时用 `--overwrite`。
dry-run 不验证 token。账户文件无需放入共享目录，也不会进入 Git 或上传 ZIP。

大量下载策略、三年逐小时示例和恢复方法见 [ERA5_DOWNLOAD.md](ERA5_DOWNLOAD.md)。
