# TOY_SERVER：独立圆岛实验上传包

上传整个目录即可运行。默认采用与仓库 `island_toy` 相同的圆岛、浅水裙边和海况定义。
生成海岛样本无需真实地形或 ERA5。九点风直接输入，四角波浪采用 3×2×2 矩阵。
默认 SWAN 网格为 **15 arc-sec**：0.5° 窗口含 120×120 网格单元、121×121 节点。
`gebco15s` 只是分辨率配置名，地形仍是人工圆岛。
详细设计见 [DESIGN.md](DESIGN.md)，本机实际试跑结果见 [LOCAL_VALIDATION.md](LOCAL_VALIDATION.md)。

## 文件职责

| 文件 | 职责 |
|---|---|
| `config.json` | 圆岛数量、海况范围、网格和 SWAN 路径 |
| `generate_cases.py` | 生成自包含圆岛 case、原始矩阵和 manifest |
| `run_swan.py` | 读取已有 case，检查输出并记录收敛 |
| `prepare_dataset.py` | 将已完成 case 转为训练 shard |
| `inspect_cases.py`、`inspection_template.html` | 读取配置和已有文件，生成离线检查页及 PNG 图 |
| `request_era5.py`、`era5_config.json` | 独立 ERA5 请求入口与下载配置 |
| `era5_jobs.py`、`era5_merge.py` | 请求拆分、五槽调度、任务恢复及逐块合并 |
| `island_core.py` | 同源圆岛几何、风浪条件和 split 库 |
| `forcing_arrays.py` | 矩阵契约与模型特征 |
| `swan_inputs.py`、`swan_runtime.py` | SWAN 输入文件和运行结果检查库 |

入口脚本互不导入；case 文件是生成、运行和转换阶段之间的接口。

## 服务器操作顺序

进入上传后的 `TOY_SERVER`，使用 Python 3.10 或更新版本。
Ubuntu 上先用 `python3` 创建并激活项目虚拟环境，激活后就有 `python` 命令：

```text
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

每次新开 SSH 终端都需要重新执行 `source ~/TOY_SERVER/.venv/bin/activate`。
也可以直接用 `~/TOY_SERVER/.venv/bin/python` 运行脚本。
VS Code 远程 Python 解释器应选择 `/root/TOY_SERVER/.venv/bin/python`（当前实例使用 root）。

在 `config.json` 中设置 `paths.swan_executable`，指向实际计算程序或 PATH 上的名称。
当前 runner 直接调用计算程序，适用于串行/OpenMP；MPI 版本需要适合服务器的启动器。
此阶段使用 CPU 上的 SWAN，不依赖 Torch 或 CUDA。

先试一个 case：

```text
python generate_cases.py --profile gebco15s --limit 2
python run_swan.py --profile gebco15s --case-id IT000002 --check
python run_swan.py --profile gebco15s --case-id IT000002
```

第二例是非零风与波浪方向一致的海况；本机已在隔离目录验证同一生成、运行、转换流程。
`--check` 检查 case 文件和程序路径，不运行 SWAN。
`generate_cases --limit` 是每个窗口和分辨率的 case 上限；runner 的 limit 是筛选后的总上限。
默认完整规模为 608 cases，使用 15 arc-sec。

首例通过并确认 PRINT 和输出后，再生成全部：

```text
python generate_cases.py
python run_swan.py --check
python run_swan.py --workers 2
python prepare_dataset.py --profile gebco15s
```

生成目录为 `cases_island/profile/tile/ITxxxxxx/`，每个 case 自带 INPUT、bottom.dat、wind.dat、
swaninit、inputs.npz、case.json，计算结果在 `output/compgrid.tab`。
输出包含 Hs、Tm01、Tm02、Tp、平均/峰值方向、方向展宽、TMM10 和风。
转换器当前生成 Hs 标签，兼容仓库的 `train_v2.py`；其他 bulk 参数仍留在原始 SWAN 文件。

重复执行会复用输入一致的 case。输入变化需要 `generate_cases.py --overwrite`，
已通过 case 有意重算时使用 `run_swan.py --rerun`。修改实验数量时建议使用新的 cases 目录。
当前收敛策略为 **仅记录**（`convergence_policy: record_only`）。正常退出且输出完整的 case
标记为 `completed`，即使没有达到 SWAN 日志中的收敛比例，也会保存结果并进入数据转换。
`run_status.json` 和数据集 `manifest.csv` 保留收敛比例、日志中的要求比例、是否收敛、迭代次数、日志识别状态。
日志未知时 JSON 的 `converged` 为 `null`，CSV 对应字段留空；未收敛时为 `false` / `False`。
旧策略的 `unconverged`、`unverified` 输出可在输入和输出校验通过后直接转换，重复运行 runner 会复用并更新状态。
程序报错、超时、输出缺失或损坏仍记为 `failed`。批量运行后可按记录分析各海况。
转换器默认使用配置中唯一的实验分辨率；配置多个分辨率时必须显式指定 `--profile`。
单个 case 的 `.swan.lock` 防止两个 runner 同时写入；异常终止后需确认进程已停止，再清理遗留锁。

## 本机环境

在 PowerShell 中激活 `DTP_env` 后进入本目录。没有激活时，也可使用绝对路径调用 Python：

```powershell
& 'C:\Users\15507\.conda\envs\DTP_env\python.exe' generate_cases.py --profile gebco15s --limit 2
& 'C:\Users\15507\.conda\envs\DTP_env\python.exe' run_swan.py --swan D:/swan/build/bin/swan.exe --profile gebco15s --case-id IT000002
& 'C:\Users\15507\.conda\envs\DTP_env\python.exe' prepare_dataset.py --profile gebco15s
```

上面的普通入口使用 `config.json`；此次验收另用 `local_check/20261009/swan_15arcsec_config.json`，
避免试跑结果混入正式样本。上传包保留可移植的 `swan` 默认程序名，服务器上按实际安装路径设置。

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

## 可视化检查

激活 `DTP_env` 后，在本目录执行。本机现有 15 arc-sec 试跑目录：

```text
python inspect_cases.py --cases local_check/20261009/cases_15arcsec --output inspection
```

双击生成的 [inspection/index.html](inspection/index.html)，或用浏览器打开它。
整个 `inspection` 文件夹可复制到本机离线查看；里面的 PNG 可以独立使用。
此入口只读取配置和已有文件，不生成 case，不执行 SWAN，也不请求 ERA5。
Windows 上 Matplotlib 需要 Conda 环境的 DLL 路径，应先 `conda activate DTP_env`，再执行命令。

服务器正式 case 使用默认配置路径时：

```text
python inspect_cases.py --output inspection
```


### 在本地浏览器查看服务器报告

在 VS Code Remote SSH 的服务器终端中执行：

```bash
cd ~/TOY_SERVER
source .venv/bin/activate
python -m http.server 8000 --bind 127.0.0.1 --directory inspection
```

在 VS Code 的 Ports 面板转发服务器端口 8000，点击转发后的本地地址打开报告。
HTTP 服务和报告目录均在服务器，浏览器在本机。退出服务时按 `Ctrl+C`。
也可以下载整个 `inspection/` 文件夹后打开 `index.html`；同时保留 `assets/`。
操作见 [VS Code Remote SSH 端口转发说明](https://code.visualstudio.com/docs/remote/ssh#_forwarding-a-port-creating-ssh-tunnel)。

检查页提供：

- 全部计划 case 的海况、岛屿、划分和实际进度；32 个圆岛的地形预览。
- 已准备输入的哈希检查，以及实际四角波浪和原始九点风矩阵。
- 波浪传播方向与风矢量、四角风浪方向差。文件行从南到北，表格按北在上显示。
- SWAN 多个 bulk 参数空间分布、迭代收敛曲线、状态、INPUT 与日志摘要。

没有计算输出时明确显示未计算。收敛仅记录，未收敛输出仍可显示。
重新执行此命令即可更新文件快照；报告不读取账户文件。
