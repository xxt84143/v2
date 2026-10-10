# Validation evaluation

- Checkpoint: `C:\Users\15507\AppData\Local\Temp\toy_v21_release_20261010\cpu_smoke_relative_depth\best.pt`
- Cases / valid pixels: 121 / 1719482
- Inference: 2.069 s (58.486 cases/s)

## Primary results

| Metric | Model | Boundary-Hs baseline |
| --- | ---: | ---: |
| MAE (m) | 0.398368 | 0.232631 |
| RMSE (m) | 0.486287 | 0.331709 |
| Bias (m) | 0.000953 | -0.027880 |
| Pearson r | 0.673599 | 0.863966 |
| R2 | 0.449858 | 0.744021 |
| Scatter index | 0.293393 | 0.199423 |

MSE skill relative to the bilinearly interpolated boundary-Hs baseline: **-1.149170**.

## Statistical convention

Micro metrics pool all valid wet pixels. Macro metrics first score each case and then give cases equal weight. The reported 95% confidence intervals are percentile bootstraps over cases, not over spatial pixels. Scatter index is centered RMSE divided by mean SWAN Hs. Bias is prediction minus SWAN. R2 is evaluated against the pooled SWAN mean. Percentage errors are intentionally omitted near calm-water values.

Use validation for iteration and checkpoint/model choices. Run the frozen choice on test once for final reporting.
