# 本机运行检查：2026-10-09

## 结论

圆岛样本的生成、SWAN 实际计算、结果验收和 Hs 训练数据转换已在 **15 arc-sec** 网格上通过。
此次验证包含一例非零九点风与边界波浪方向一致的海况；尚未验收全部 608 个 case，也未训练神经网络。

## 环境与实际结果

- Python：`DTP_env`，Python 3.12.14。
- SWAN：`D:/swan/build/bin/swan.exe`，41.51AB；CPU、1 个 worker、每 case 1 线程。
- 本次在该环境补装 scipy、netCDF4、cftime，用于实际 NetCDF 读写检查。
- 验收配置：`local_check/20261009/swan_15arcsec_config.json`。
- 计算 case：`gebco15s__E121p5_N23p5__IT000002`，人工圆岛、浅水裙边、aligned_windsea。
- 网格：经纬度间隔 `15/3600°`；121×121 节点，14,463 个湿节点。
- 结果：7 次迭代，100% 满足精度要求（要求 99.5%）；耗时 145.116 秒。
- 湿点 Hs 范围：0.039320618–2.4401982 m。
- 检查包括输出坐标与网格一致、有效水深、Hs、周期和方向取值以及收敛日志。
- 重复执行返回 `skipped`，输出哈希一致。

转换产物在 `local_check/20261009/dataset_15arcsec/`：

| 内容 | 形状或检查 |
|---|---|
| 原始波浪 | `(3, 2, 2)`，Hs、Tp、FROM 方向 |
| 原始九点风 | `(2, 3, 3)`，u10、v10 |
| 模型输入 x | `(10, 128, 128)`，原始 121×121 右侧和上侧补零 |
| Hs 标签 y、掩膜 | `(1, 128, 128)` |
| 数值检查 | 输入和标签有限、湿点掩膜一致、补齐区域无效、反归一化标签与 SWAN Hs 一致 |

原始 SWAN 文件保留多个 bulk 参数；当前转换器只生成 Hs 监督标签。

## 试跑发现并修复的问题

1. `PROJECT` 的第二个字段原为六字符 `ISLAND`，超过 SWAN 的四字符限制，改为 `ISLD`。
   依据：[SWAN PROJECT 文档](https://swanmodel.sourceforge.io/online_doc/swanuse/node23.html)。
2. 默认 BLOCK 输出保留的数字不足以核验细网格经纬度，增加 `OUTPUT OPTIONS BLOCK 8 6`。
   保留坐标检查要求，依据：[SWAN 输出设置文档](https://swanmodel.sourceforge.io/online_doc/swanuse/node32.html)。
3. 每个 case 增加独占 `.swan.lock`，防止两个运行进程同时覆盖 PRINT 和计算输出。
4. 服务器配置和圆岛开发配置默认改为 `gebco15s`；转换器和圆岛 runner 按唯一配置分辨率选择默认值。

## 已知范围与后续验收

改用 15 arc-sec 前，1 km 探索试跑的无风涌浪、同向风浪、弱风涌浪三例通过。
纯风生浪一例在默认 50 次迭代后只有 92.49% 节点达到精度要求，状态为 `unconverged`，没有进入训练集。
以上是旧策略下的试跑记录。按后续要求，当前收敛只记录、不拦截保存或转换；
该例可经校验后直接复用并标记为 `completed`，仍保留 `converged=false`，无需重算。
这说明纯风生浪仍需单独检查迭代设置；不能把本次通过的一例视为所有海况均已验收。
15 arc-sec 批量运行时应先统计每类海况的收敛、失败和耗时，再扩大规模。

此前 Python 文件契约、下载调度和实际 NetCDF 读写回归检查共 **10 项**，全部通过。
测试覆盖 15 arc-sec 网格、九点风写入、输入损坏检查、同 case 并发锁、日期拆分、mwp 保留、
五槽滚动调度、超限拆分持久化、断线续用任务 ID，以及逐块合并数值。
运行命令：`python -m unittest discover -s tests -v`。
收敛策略调整增加两项回归检查，覆盖未收敛/未知日志的保存与旧结果复用，以及程序错误/缺失输出仍失败。
另修复 Windows 状态文件替换可能遇到的短暂权限占用，最多有限重试，并增加对应回归检查。
更新后共 **13 项测试全部通过**，日志在 `local_check/20261009/record_only_tests.log`。

当前策略实际验收：

- 旧 1 km 纯风生浪输出已校验复用，状态更新为 `completed`，记录仍是 `converged=false`、92.49%、50 次迭代。
- `dataset_record_only_1km/` 已包含四例，其中含该未收敛输出；索引的收敛记录已核验。
- 15 arc-sec 的现有输出已复用，重新转换到 `dataset_15arcsec_record_only/`，索引和元数据保留收敛记录。
- 以上均复用已有计算输出，没有重新执行 SWAN 计算；默认正式网格仍是 15 arc-sec。

环境仍出现 NumPy/netCDF4 的兼容或弃用提示，以及可选 cfgrib 后端缺 ecCodes 提示；
此次 NetCDF 读写和数值断言均通过。GRIB/完整波谱读取未在本次检查范围内。

## ERA5 小规模实际请求

用 `local_check/20261009/era5_config.json` 请求了 2023-07-01 00:00 UTC 的六个 bulk 变量，
最多同时占用五个槽位。已下载并校验 u10；完成一个下载后自动提交第六个请求。
其余请求曾处于 CDS 排队状态；按用户要求不等待其完成，已有下载继续运行并保存任务状态。
实际六变量全部下载与合并尚未验收；离线合成 NetCDF 的校验和合并已通过。

本机系统代理导致 Python 访问 CDS 出现 TLS EOF；此次只对下载进程设置 `NO_PROXY=*` 后连接成功。
未修改系统代理，未关闭证书验证。服务器按实际网络设置处理，不需照搬这个本机覆盖。
任务和请求 ID 保存在 `local_check/20261009/data/raw/`；下载进程结束后重复同一命令可恢复。
不要同时对同一 raw 目录启动第二个下载器。

原仓库根目录 `ERA5Downloading.py` 未修改，它仍用于完整波谱请求。

## 整理后的路径

旧 1 km 的配置、case、转换数据已归档到 `Learn/_archive/20261009/previous_local_checks/`。
旧 `dataset_15arcsec/` 已归档为其中的 `dataset_15arcsec_before_record_only/`。
当前运行目录保留 15 arc-sec 的输入、结果与 `dataset_15arcsec_record_only/`。
可视化入口是 `inspect_cases.py`；检查页与 PNG 位于 `inspection/`，可查看实际结果及完整计划预览。

## 可视化检查验收

基于当前配置及实际 15 arc-sec case，生成了完整计划索引和 58 张 PNG。
计划 608 cases（train 288、validation 160、test 160），四类海况各 152 cases；
实际准备 2 例，已读取且通过数值检查的 SWAN 输出 1 例。
浏览器中验证了状态/海况/划分筛选、case 切换、矩阵北西方向、零波浪边界、输出标签页与图片加载。
确认计划 case 不会显示上一例的实际 SWAN 输出。

现有同向风浪例由东偏北向西偏南传播，岛西侧有明显低波高区，输入与输出方向箭头一致。
该观察只涉及现有一例；全部海况的计算和数值效果仍待后续检查。
