# 数据与验证误差定量审计

本审计直接只读打开压缩包，不解压、不训练、不运行预测，也不读取 test 的 y 数组。计划来自当前确定性生成器，验证误差来自已有 metrics_by_case.csv。缺失表示没有进入已交付数据集，不能由此单独判定运行失败或超时。

计划 608 例，实得 474 例，缺失 134 例。原始网格 [(121, 121)]；已补零至 128 × 128。
实际 forcing 数 17，地形数 32。已存在样本与当前生成器：wave 最大差 0.0，wind 最大差 0.0，x 最大差 1.1920928955078125e-07，mask 不一致 0 例，ID/划分不一致 0 例。

按 TRAIN_SERVER/training_core.py 的字节哈希算法复算 dataset signature：`b9758c251cccbc9989c0d94842794ad4ced2b9ea64f19ebf38b6c5ef3597d69a`。与 toy_runs/hs_unet/dataset_check.json 一致：True。

## 划分覆盖

| split | generalization | planned | actual | unconverged | missing |
| --- | --- | --- | --- | --- | --- |
| train | seen_island_seen_forcing | 288 | 227 | 0 | 61 |
| validation | new_island | 48 | 37 | 0 | 11 |
| validation | new_forcing | 96 | 72 | 0 | 24 |
| validation | new_both | 16 | 12 | 0 | 4 |
| test | new_island | 48 | 38 | 0 | 10 |
| test | new_forcing | 96 | 72 | 0 | 24 |
| test | new_both | 16 | 16 | 4 | 0 |

## 每组 forcing 的覆盖和物理条件

| forcing_id | forcing_split | family | planned | actual | missing | unconverged | mean_hs_m | mean_tp_s | mean_direction_deg | mean_wind_speed_mps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F001 | train | swell_no_wind | 32 | 32 | 0 | 0 | 2.3427 | 10.7438 | 159.8003 | 0.0000 |
| F002 | train | aligned_windsea | 32 | 32 | 0 | 0 | 2.1669 | 7.2037 | 74.1854 | 10.3514 |
| F003 | train | swell_weak_wind | 32 | 32 | 0 | 0 | 1.1194 | 9.5936 | 253.6108 | 3.0200 |
| F004 | train | wind_only | 32 | 0 | 32 | 0 | 0.0000 | 8.0000 | 337.7798 | 9.2861 |
| F005 | train | swell_no_wind | 32 | 32 | 0 | 0 | 1.2567 | 12.1965 | 270.3097 | 0.0000 |
| F006 | train | aligned_windsea | 32 | 32 | 0 | 0 | 2.0746 | 8.1095 | 185.2167 | 13.9358 |
| F007 | train | swell_weak_wind | 32 | 32 | 0 | 0 | 0.8058 | 13.2302 | 237.8606 | 1.9964 |
| F008 | train | wind_only | 32 | 14 | 18 | 0 | 0.0000 | 8.0000 | 313.6673 | 10.7135 |
| F009 | train | swell_no_wind | 32 | 32 | 0 | 0 | 1.2864 | 9.0791 | 4.7772 | 0.0000 |
| F010 | train | aligned_windsea | 32 | 32 | 0 | 0 | 1.3080 | 5.6582 | 135.3119 | 10.6373 |
| F011 | train | swell_weak_wind | 32 | 32 | 0 | 0 | 2.3502 | 9.5728 | 102.7073 | 3.0018 |
| F012 | train | wind_only | 32 | 0 | 32 | 0 | 0.0000 | 8.0000 | 35.8612 | 12.1889 |
| F013 | validation | swell_no_wind | 28 | 28 | 0 | 0 | 0.9539 | 10.6904 | 208.5225 | 0.0000 |
| F014 | validation | aligned_windsea | 28 | 28 | 0 | 0 | 1.8716 | 7.8859 | 148.7926 | 11.1918 |
| F015 | validation | swell_weak_wind | 28 | 28 | 0 | 0 | 2.1165 | 10.4377 | 10.7797 | 1.7866 |
| F016 | validation | wind_only | 28 | 0 | 28 | 0 | 0.0000 | 8.0000 | 358.8277 | 6.7465 |
| F017 | test | swell_no_wind | 28 | 28 | 0 | 0 | 2.0317 | 11.4399 | 266.6435 | 0.0000 |
| F018 | test | aligned_windsea | 28 | 28 | 0 | 0 | 1.2839 | 6.1985 | 345.6646 | 8.9657 |
| F019 | test | swell_weak_wind | 28 | 28 | 0 | 0 | 1.7434 | 11.8968 | 126.0530 | 1.4906 |
| F020 | test | wind_only | 28 | 4 | 24 | 4 | 0.0000 | 8.0000 | 38.6952 | 11.3269 |

## 按 split 和族的缺失

| split | family | planned | actual | unconverged | missing |
| --- | --- | --- | --- | --- | --- |
| train | swell_no_wind | 72 | 72 | 0 | 0 |
| train | aligned_windsea | 72 | 72 | 0 | 0 |
| train | swell_weak_wind | 72 | 72 | 0 | 0 |
| train | wind_only | 72 | 11 | 0 | 61 |
| validation | swell_no_wind | 40 | 40 | 0 | 0 |
| validation | aligned_windsea | 40 | 40 | 0 | 0 |
| validation | swell_weak_wind | 40 | 40 | 0 | 0 |
| validation | wind_only | 40 | 1 | 0 | 39 |
| test | swell_no_wind | 40 | 40 | 0 | 0 |
| test | aligned_windsea | 40 | 40 | 0 | 0 |
| test | swell_weak_wind | 40 | 40 | 0 | 0 |
| test | wind_only | 40 | 6 | 4 | 34 |

## 未收敛但进入数据集的样本

| case_id | terrain_id | forcing_id | split | family | convergence_percent | iterations |
| --- | --- | --- | --- | --- | --- | --- |
| IT000596 | T029 | F020 | test | wind_only | 95.5800 | 50.0000 |
| IT000600 | T030 | F020 | test | wind_only | 95.3200 | 50.0000 |
| IT000604 | T031 | F020 | test | wind_only | 97.0900 | 50.0000 |
| IT000608 | T032 | F020 | test | wind_only | 95.5100 | 50.0000 |

## 验证按 forcing 汇总

RMSE 从每例 SSE 与有效像元数重构，不是简单平均每例 RMSE；fraction_total_sse 是该 forcing 对全部验证 SSE 的贡献比例。

| forcing_id | family | n_cases | rmse_m | bias_m | centered_rmse_m | baseline_rmse_m | skill | fraction_total_sse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F001 | swell_no_wind | 4 | 0.1251 | 0.0165 | 0.1241 | 0.4627 | 0.9268 | 0.0022 |
| F002 | aligned_windsea | 4 | 0.1141 | 0.0775 | 0.0838 | 0.3029 | 0.8581 | 0.0019 |
| F003 | swell_weak_wind | 4 | 0.1147 | 0.1046 | 0.0471 | 0.2080 | 0.6959 | 0.0019 |
| F005 | swell_no_wind | 4 | 0.0871 | 0.0665 | 0.0563 | 0.2407 | 0.8691 | 0.0011 |
| F006 | aligned_windsea | 4 | 0.0999 | 0.0436 | 0.0899 | 0.4335 | 0.9469 | 0.0014 |
| F007 | swell_weak_wind | 4 | 0.0942 | 0.0657 | 0.0675 | 0.1644 | 0.6716 | 0.0013 |
| F008 | wind_only | 1 | 0.1484 | 0.1300 | 0.0717 | 1.0330 | 0.9794 | 0.0008 |
| F009 | swell_no_wind | 4 | 0.1127 | 0.0931 | 0.0636 | 0.2235 | 0.7455 | 0.0018 |
| F010 | aligned_windsea | 4 | 0.0963 | 0.0729 | 0.0629 | 0.2353 | 0.8325 | 0.0013 |
| F011 | swell_weak_wind | 4 | 0.1236 | 0.0659 | 0.1046 | 0.2965 | 0.8261 | 0.0022 |
| F013 | swell_no_wind | 28 | 0.2150 | 0.0634 | 0.2054 | 0.1726 | -0.5511 | 0.0454 |
| F014 | aligned_windsea | 28 | 0.5497 | -0.4908 | 0.2476 | 0.4527 | -0.4744 | 0.2969 |
| F015 | swell_weak_wind | 28 | 0.8082 | -0.7787 | 0.2164 | 0.2933 | -6.5941 | 0.6418 |

## 验证按泛化类别和 forcing 汇总

| generalization | forcing_id | n_cases | rmse_m | bias_m | baseline_rmse_m | skill |
| --- | --- | --- | --- | --- | --- | --- |
| new_both | F013 | 4 | 0.2139 | 0.0451 | 0.1435 | -1.2212 |
| new_both | F014 | 4 | 0.5708 | -0.5154 | 0.4800 | -0.4140 |
| new_both | F015 | 4 | 0.8323 | -0.8069 | 0.2399 | -11.0397 |
| new_forcing | F013 | 24 | 0.2152 | 0.0666 | 0.1771 | -0.4765 |
| new_forcing | F014 | 24 | 0.5461 | -0.4866 | 0.4479 | -0.4861 |
| new_forcing | F015 | 24 | 0.8040 | -0.7739 | 0.3014 | -6.1170 |
| new_island | F001 | 4 | 0.1251 | 0.0165 | 0.4627 | 0.9268 |
| new_island | F002 | 4 | 0.1141 | 0.0775 | 0.3029 | 0.8581 |
| new_island | F003 | 4 | 0.1147 | 0.1046 | 0.2080 | 0.6959 |
| new_island | F005 | 4 | 0.0871 | 0.0665 | 0.2407 | 0.8691 |
| new_island | F006 | 4 | 0.0999 | 0.0436 | 0.4335 | 0.9469 |
| new_island | F007 | 4 | 0.0942 | 0.0657 | 0.1644 | 0.6716 |
| new_island | F008 | 1 | 0.1484 | 0.1300 | 1.0330 | 0.9794 |
| new_island | F009 | 4 | 0.1127 | 0.0931 | 0.2235 | 0.7455 |
| new_island | F010 | 4 | 0.0963 | 0.0729 | 0.2353 | 0.8325 |
| new_island | F011 | 4 | 0.1236 | 0.0659 | 0.2965 | 0.8261 |

## 验证按收敛状态汇总

| converged | n_cases | n_valid | rmse_m | bias_m | centered_rmse_m | baseline_rmse_m | skill | fraction_total_sse | target_mean_m | prediction_mean_m | macro_rmse_m | negative_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | 121 | 1719482 | 0.4842 | -0.2566 | 0.4107 | 0.3317 | -1.1311 | 1.0000 | 1.6575 | 1.4009 | 0.3961 | 0.0008 |

## 解释边界

当前配置与已存在数组匹配，只能验证输入的来源一致，不能追回缺失样本的运行原因。压缩包没有全部 608 个 run_status.json/PRINT，因此不要将 missing 自动称为 unconverged。有效独立 forcing 数应按训练中出现的 forcing_id 计数，不能把同一 forcing 在 24 个地形上的重复组合算成 24 组新海况。

复现：`C:/Users/15507/.conda/envs/DTP_env/python.exe reports/swan_surrogate_review/audit_dataset.py`。所有 CSV/JSON/Markdown 均写入本报告目录。
