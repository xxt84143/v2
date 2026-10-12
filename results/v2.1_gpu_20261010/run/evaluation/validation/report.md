# Validation evaluation

- Checkpoint: `/root/autodl-tmp/toy_runs/hs_unet_v21/best.pt`
- Cases / valid pixels: 121 / 1719482
- Inference: 1.642 s (73.709 cases/s)

## Primary results

| Metric | Model | Boundary-Hs baseline |
| --- | ---: | ---: |
| MAE (m) | 0.255759 | 0.232631 |
| RMSE (m) | 0.399201 | 0.331709 |
| Bias (m) | -0.190386 | -0.027880 |
| Pearson r | 0.850845 | 0.863966 |
| R2 | 0.629257 | 0.744021 |
| Scatter index | 0.211696 | 0.199423 |

MSE skill relative to the bilinearly interpolated boundary-Hs baseline: **-0.448333**.

## Statistical convention

Micro metrics pool all valid wet pixels. Macro metrics first score each case and then give cases equal weight. The reported 95% confidence intervals are percentile bootstraps over cases, not over spatial pixels. Scatter index is centered RMSE divided by mean SWAN Hs. Bias is prediction minus SWAN. R2 is evaluated against the pooled SWAN mean. Percentage errors are intentionally omitted near calm-water values.

Use validation for iteration and checkpoint/model choices. Run the frozen choice on test once for final reporting.
