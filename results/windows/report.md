# Shorter observation windows: neural versus conventional estimation

## Main findings

All 30 new neural models trained and all 97,500 new conventional fits completed. The six models and 19,500 fits at 50 ms were verified and reused, not rerun. All new fits reported convergence; actual parameter errors and tails remain the accuracy criteria.

- Conventional fitting remains substantially more accurate than the current networks under the known ideal physics.
- Fitting accuracy improves with observation duration and largely plateaus by 10–20 ms in this tested range.
- The current matching-input MLPs often perform better on shorter recordings than the 50 ms baseline. This is architecture/training-dependent: additional measurements do not inherently remove information, and smaller inputs also reduce first-layer parameter counts.
- These are separately trained networks, not the unchanged 50 ms model accepting missing observations.
- No test-derived architecture change or extra tuning was performed. Window comparisons are descriptive; the best observed test window should not be treated as a validation-selected deployment choice.

## At 10% measurement noise: median parameter error (%)

Each cell is R / L / C. Neural values average per-condition medians across three training seeds and three noise realizations. Fitting values average per-realization medians. Same 1,500 test circuits and exact physical-noise prefixes; repeated observations are not independent circuits.

| Window | Points/channel | Clean-trained MLP | Noise-trained MLP | Fitting |
|---|---:|---:|---:|---:|
| 1 ms | 101 | 3.659 / 1.104 / 1.227 | 2.933 / 0.816 / 0.928 | 0.830 / 0.370 / 0.382 |
| 2 ms | 201 | 2.966 / 1.225 / 1.146 | 2.310 / 0.983 / 0.869 | 0.305 / 0.242 / 0.207 |
| 5 ms | 501 | 2.522 / 1.512 / 1.405 | 1.614 / 0.967 / 1.017 | 0.192 / 0.190 / 0.171 |
| 10 ms | 1001 | 2.710 / 1.655 / 1.794 | 1.825 / 1.314 / 1.303 | 0.168 / 0.180 / 0.165 |
| 20 ms | 2001 | 3.835 / 2.143 / 2.736 | 2.077 / 1.417 / 1.407 | 0.163 / 0.178 / 0.162 |
| 50 ms | 5001 | 6.913 / 3.162 / 4.421 | 2.745 / 1.971 / 2.012 | 0.162 / 0.178 / 0.162 |

## At 10% noise: mean per-condition 95th-percentile error (%)

| Window | Noise-trained MLP R / L / C | Fitting R / L / C |
|---|---:|---:|
| 1 ms | 12.044 / 2.540 / 3.082 | 5.760 / 1.222 / 1.551 |
| 2 ms | 7.429 / 3.056 / 2.675 | 1.480 / 0.825 / 0.713 |
| 5 ms | 4.861 / 3.017 / 3.037 | 0.685 / 0.748 / 0.626 |
| 10 ms | 5.737 / 4.102 / 4.012 | 0.618 / 0.743 / 0.629 |
| 20 ms | 6.671 / 4.537 / 4.784 | 0.615 / 0.742 / 0.629 |
| 50 ms | 8.885 / 6.329 / 6.635 | 0.615 / 0.742 / 0.629 |

## Clean inputs: mean per-seed median error (%)

| Window | Clean-trained MLP R / L / C | Noise-trained MLP R / L / C |
|---|---:|---:|
| 1 ms | 0.562 / 0.264 / 0.352 | 0.809 / 0.433 / 0.498 |
| 2 ms | 0.486 / 0.311 / 0.362 | 0.827 / 0.481 / 0.492 |
| 5 ms | 0.594 / 0.408 / 0.461 | 0.748 / 0.511 / 0.649 |
| 10 ms | 0.679 / 0.478 / 0.550 | 0.846 / 0.696 / 0.731 |
| 20 ms | 0.938 / 0.597 / 0.790 | 0.878 / 0.657 / 0.723 |
| 50 ms | 0.791 / 0.683 / 0.678 | 0.958 / 0.677 / 0.815 |

Clean fitting errors are near numerical precision; inspect raw JSON for non-rounded values.

## Processing latency and observation time

Neural values are ranges of single-circuit CPU medians across three augmented models, including preprocessing and decoding. Fitting medians span all 13 actual estimation conditions per window. CPU thread budget is one. Neural benchmarks are warmed repeated requests; fitting measurements are serial solves. These are not identical benchmark workloads.

| Window | Neural CPU processing (ms) | Fitting processing median (ms) | Neural earliest estimate, approximately (ms) | Fitting earliest estimate, approximately (ms) |
|---|---:|---:|---:|---:|
| 1 | 0.022–0.022 | 1.504 | 1.022 | 2.504 |
| 2 | 0.023–0.023 | 1.667 | 2.023 | 3.667 |
| 5 | 0.025–0.026 | 2.306 | 5.026 | 7.306 |
| 10 | 0.028–0.028 | 2.962 | 10.028 | 12.962 |
| 20 | 0.033–0.034 | 3.979 | 20.033 | 23.979 |
| 50 | 0.048–0.048 | 6.926 | 50.048 | 56.926 |

Acquisition-plus-processing excludes sensor/communication overhead. All 50 ms timings are historical and were not remeasured. CPU remains faster than MPS for these small neural inference requests; device-specific benchmarks are preserved in raw results.

## All noise levels: noise-trained MLP versus fitting median errors (%)

| Window | Noise | Noise-trained MLP R / L / C | Fitting R / L / C |
|---|---|---:|---:|
| 1 ms | 0% | 0.809 / 0.433 / 0.498 | 0.000 / 0.000 / 0.000 |
| 1 ms | 1% | 0.877 / 0.438 / 0.508 | 0.083 / 0.036 / 0.038 |
| 1 ms | 2% | 1.002 / 0.455 / 0.521 | 0.170 / 0.072 / 0.077 |
| 1 ms | 5% | 1.679 / 0.553 / 0.653 | 0.413 / 0.181 / 0.188 |
| 1 ms | 10% | 2.933 / 0.816 / 0.928 | 0.830 / 0.370 / 0.382 |
| 2 ms | 0% | 0.827 / 0.481 / 0.492 | 0.000 / 0.000 / 0.000 |
| 2 ms | 1% | 0.859 / 0.493 / 0.498 | 0.030 / 0.025 / 0.020 |
| 2 ms | 2% | 0.953 / 0.515 / 0.511 | 0.061 / 0.048 / 0.041 |
| 2 ms | 5% | 1.397 / 0.662 / 0.618 | 0.155 / 0.121 / 0.101 |
| 2 ms | 10% | 2.310 / 0.983 / 0.869 | 0.305 / 0.242 / 0.207 |
| 5 ms | 0% | 0.748 / 0.511 / 0.649 | 0.000 / 0.000 / 0.000 |
| 5 ms | 1% | 0.763 / 0.520 / 0.656 | 0.018 / 0.020 / 0.016 |
| 5 ms | 2% | 0.832 / 0.541 / 0.675 | 0.037 / 0.040 / 0.033 |
| 5 ms | 5% | 1.089 / 0.664 / 0.764 | 0.095 / 0.096 / 0.085 |
| 5 ms | 10% | 1.614 / 0.967 / 1.017 | 0.192 / 0.190 / 0.171 |
| 10 ms | 0% | 0.846 / 0.696 / 0.731 | 0.000 / 0.000 / 0.000 |
| 10 ms | 1% | 0.864 / 0.706 / 0.733 | 0.016 / 0.019 / 0.016 |
| 10 ms | 2% | 0.913 / 0.745 / 0.759 | 0.033 / 0.037 / 0.032 |
| 10 ms | 5% | 1.198 / 0.905 / 0.915 | 0.083 / 0.093 / 0.082 |
| 10 ms | 10% | 1.825 / 1.314 / 1.303 | 0.168 / 0.180 / 0.165 |
| 20 ms | 0% | 0.878 / 0.657 / 0.723 | 0.000 / 0.000 / 0.000 |
| 20 ms | 1% | 0.911 / 0.681 / 0.733 | 0.016 / 0.018 / 0.016 |
| 20 ms | 2% | 0.990 / 0.717 / 0.780 | 0.032 / 0.036 / 0.031 |
| 20 ms | 5% | 1.372 / 0.937 / 0.983 | 0.082 / 0.090 / 0.080 |
| 20 ms | 10% | 2.077 / 1.417 / 1.407 | 0.163 / 0.178 / 0.162 |
| 50 ms | 0% | 0.958 / 0.677 / 0.815 | 0.000 / 0.000 / 0.000 |
| 50 ms | 1% | 1.022 / 0.718 / 0.845 | 0.016 / 0.018 / 0.016 |
| 50 ms | 2% | 1.137 / 0.805 / 0.921 | 0.032 / 0.036 / 0.031 |
| 50 ms | 5% | 1.660 / 1.208 / 1.280 | 0.081 / 0.090 / 0.080 |
| 50 ms | 10% | 2.745 / 1.971 / 2.012 | 0.162 / 0.178 / 0.162 |

## Verification and limitations

- Verified 42 prediction archives: hashes, finite outputs, 19,500 rows each, original IDs/truths/condition layouts and full-noise hashes. New-window prefix-noise hashes agree across all seven estimators.
- Source/data/reference hashes match; earlier raw experiment results were not modified.
- Noise sigma uses the original full-training channel RMS at every window: 10% is approximately 124.3 µA current and 0.1003 V voltage, not per-waveform peak noise.
- Fixed sampling interval is 10 µs. Both channels and known 1 V input are retained. This is not yet a sampling-density study.
- All new checkpoints were selected before new test evaluation using fixed noisy validation prefixes. Training augmentation is independent by prefix shape, not exactly paired across windows.
- Matched hidden widths do not mean matched total parameter count. Fixed training budgets and optimizer settings can favor some input shapes. No information-theoretic claim follows from the neural window ranking.
- Three seeds/realizations give descriptive replication, not population confidence intervals. The test split still has only six overdamped and one near-critical circuit.
- Known ideal series-RLC model only; no real hardware, sensor bandwidth effects, colored noise, uncertain input or parasitic components.

## Files

- Protocol: `docs/observation_windows.md`
- Reproduce: `.venv/bin/python -m scripts.window_experiment` (fresh output required)
- Local outputs: `runs/windows/results.json`, per-window checkpoints and predictions
- Committed record: `results/windows/results.json` and `results/windows/report.md`
- Next stage, after review: fewer measurement points at a fixed 50 ms duration.
