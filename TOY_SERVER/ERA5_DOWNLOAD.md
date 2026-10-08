# ERA5 大量数据下载：分块、五槽调度、恢复

本次阅读了 [v2 的 ERA5Downloading.py](https://github.com/xxt84143/v2/blob/main/ERA5Downloading.py)，
采用其每个任务独立客户端、临时文件完成后改名的做法。本次修改的入口是 `request_era5.py`，
申请风和 bulk 波浪数据，数据集仍为 `reanalysis-era5-single-levels`。
圆岛 case 生成器独立运行，不会自动请求 ERA5。

## 1. 先理解限制

对当前 single levels、reanalysis 产品，一张二维网格场对应一个变量、一个时刻。
因此一个请求的字段数近似为：

```text
字段数 = 变量数 × 日期数 × 每天的时刻数
浮点数据体积估计 = 字段数 × 网格点数 × 4 字节
```

变量数和时间长度在字段计数中是相乘关系，没有严格的优先级排名。
区域缩小能减少传输、内存和转换工作量，通常不会改变字段数。
本项目默认先分变量，因为当前需要的字段仅有六个，拆开后可以各自排队、恢复与校验。
这是实施策略，并非声称单变量一定比双变量更快。

官方说明指出，限制随系统负载变化；推荐按月组织 ERA5 请求。
其表格中的 single levels 字段上限为 120000，但表格审查日期较早，
不能据此保证当前 NetCDF 请求会通过。[CDS 文档](https://confluence.ecmwf.int/spaces/CKB/pages/174856258/Climate+Data+Store+CDS+documentation)

2025 年官方公告另行限制了 NetCDF 转换成本；NetCDF 与 GRIB 的可申请规模不同。
不能将参考脚本中完整波谱 GRIB 的四个月/2500000 字段假设搬来使用。
这里保留 NetCDF，便于后续处理；遇到明确的成本超限，继续自动拆分。
[官方 NetCDF 公告](https://forum.ecmwf.int/t/limitation-change-on-netcdf-era5-requests/12477)

风 0.25°和波浪 0.5°分别请求，符合官方关于不同网格数据分开转换的建议。
完整波谱属于 ERA5-complete/MARS 请求，其排队、分块和字段计数应单独考虑。
[官方 ERA5 下载说明](https://confluence.ecmwf.int/spaces/CKB/pages/129135000/How+to+download+ERA5+from+the+Climate+Data+Store+CDS)

## 2. 当前策略

1. 按年、月和相同的每日小时集合分组。边界日期只有部分小时，就单独分组。
   不会因为把日期列表和时刻列表相乘，额外下载未要求的时刻。
2. 默认每个请求一个变量，同一月份尽量保留完整的已选日期。
3. 用字段预算和体积估计继续拆分；服务器明确拒绝请求过大时，依次拆变量、日期、时刻。
   区域保持共同的空间网格，方便直接合并。
4. 最多五个任务同时处于提交、排队、计算、下载或校验阶段。
   一个文件下载并校验完成，马上补入下一个。无需等待整个五任务批次结束。
5. 网络/传输重试使用已保存的远端 ID，不重新创建任务；完成文件按请求哈希复用。
6. 合并时逐请求、逐时间块写入 `wind.nc` 和 `waves.nc`，避免多年字段全部加载进内存。

五是本项目按你的使用约束设定的上限，可降低到 1–4；实际服务器是否同时执行五个由 CDS 决定。
启动时检查账号上是否有本队列以外的排队/运行任务，存在时停止，避免叠加请求。
运行期间请保持这个账号只运行一个下载队列；其他电脑、网页和脚本也共享账号额度。

## 3. 文件职责与配置

| 文件 | 工作 |
|---|---|
| `request_era5.py` | 命令行入口，依次调用规划、下载、合并 |
| `era5_jobs.py` | 纯请求规划、状态记录、限额调度、CDS 客户端 |
| `era5_merge.py` | ZIP/NetCDF 校验、变量规范化、逐块写文件 |
| `era5_config.json` | 保留原来的短时段配置，便于小规模起步 |
| `era5_bulk.example.json` | 2022–2024 年、逐小时、N27/W117/S20/E125 的批量示例 |

入口脚本不互相导入；公共函数在库文件中。下载配置位于 `era5.download`：

| 选项 | 默认值 | 含义 |
|---|---:|---|
| `workers` | 5 | 同时在途任务数，范围 1–5 |
| `variables_per_request` | 1 | 一个请求包含的变量数 |
| `max_days_per_request` | 31 | 月内一个请求最多包含多少日期 |
| `max_fields_per_request` | 10000 | 本地规划预算，不是官方永久上限 |
| `max_estimated_mib` | 256 | float32 原始数据估计预算，非实际 ZIP 大小或精确转换成本 |
| `http_timeout_seconds` | 120 | 单次 HTTP 超时，整个任务排队没有此时间限制 |
| `http_retries` | 10 | 查询/传输阶段的客户端 HTTP 重试预算 |
| `poll_seconds` | 30 | 远端任务查询间隔 |
| `download_attempts` | 3 | 对同一任务的网络/传输重试轮数 |
| `retry_seconds` | 30 | 外层重试基础间隔，逐次增长 |
| `merge_time_chunk` | 24 | 每次写入最多多少个已选时刻，不要求时刻连续 |

若想比较双变量请求，只需改 `variables_per_request` 为 2。
比较单位时间完成的数据量，不能仅比较请求数量。

三年示例在默认策略下初始为 36 月 × 6 变量 = **216 个请求**；若 CDS 拒绝过大请求，数量会增加。
短时段默认配置保留原有采样范围，并未擅自扩大到三年。

## 4. 使用顺序

服务器安装：

```text
python -m pip install -r requirements.txt
```

按 [CDS 设置说明](https://cds.climate.copernicus.eu/how-to-api)将个人 token 放在服务器用户主目录的 `.cdsapirc`，
并在数据集网页接受使用条款。仓库、ZIP 和请求配置不保存 token。

先查看计划：

```text
python request_era5.py --config era5_config.json --dry-run
python request_era5.py --config era5_bulk.example.json --dry-run
```

`--dry-run` 不联网、不读取 token、不写文件。
确认采样范围和磁盘空间后，开始批量拉取：

```text
python request_era5.py --config era5_bulk.example.json --download-only
```

断线、进程退出后再次运行同一命令，即可恢复。建议在服务器的 `tmux` 中运行，
避免 VS Code SSH 断开终止进程。只调整 `--workers` 不改变请求哈希。

全部下载完成后，按需合并：

```text
python request_era5.py --config era5_bulk.example.json --merge-only
```

也可省略 `--download-only`，在下载完成后自动合并。
已有合并文件需加 `--overwrite`；此参数不清空已完成 raw 文件。
严格检查原始文件可加 `--verify-cache`，会逐个重新计算 SHA256，耗费额外磁盘读取。

## 5. 恢复与失败处理

`data/raw/plan.json` 保存初始计划；`manifest.json` 保存已成功完成的最终分块清单。
每个请求有 `.status.json`，记录完整请求、远端任务 ID、完成字节数和 SHA256。
拆分父任务保存 `split` 状态，重启和 `--merge-only` 会沿同样的拆分恢复。

完成标记写入之前，会核对 CDS 结果大小、ZIP CRC、NetCDF 变量、时刻和空间网格。
只检查“文件非空”不足以证明下载完整。默认复用检查完成标记和大小；可额外启用 SHA256 复查。
多线程下载互相独立，NetCDF 校验加锁串行执行，避免底层 HDF5 读库的线程问题。

恢复的是任务和已完成文件块。官方客户端是否在一次传输内续传由它决定；
本脚本不保证进程重启后的字节级续传。

以下情况会停止补入新任务，并保留诊断：

- token、许可或请求内容错误；网络重试预算耗尽；下载内容校验失败。
- 已保存的远端任务失败或结果过期。先在 CDS 的 Your Requests 检查原因；
  只有确认原任务不再活动后，才重置对应 `.status.json` 以重新申请。
- POST 发出后丢失响应、来不及保存 ID：状态留为 `submitting`。
  先到 CDS 查明有无对应任务；已有任务时补上其 `request_id` 并改为 `submitted`，
  确认未提交时才移除该状态。不会盲目重发而占用额外额度。
- 异常退出遗留 `.download.lock`：确认同目录的下载进程停止后，才删除锁文件。

不要在一轮下载尚未完成时改变采样、变量分组或区域；这些会改变请求身份，
旧任务可能变成当前计划以外的活动任务。先完成或查清旧队列，再开始新计划。

## 6. 本次验证范围

按你的要求只做静态检查：源文件语法、接口引用、配置、请求划分逻辑、调度和恢复流程、上传包内容。
未执行项目脚本、单元测试或真实 CDS 请求，也未运行 SWAN。
因此五并发的实际吞吐量、当前 NetCDF 限制和真实下载文件兼容性仍需服务器首轮小请求验证。
