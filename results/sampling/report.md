# Fixed-duration sampling density: neural versus conventional estimation

## Main conclusion

Reducing measurement density exposes both learned-estimator errors and occasional severe errors in the fixed-start fitting baseline. This is **not a general neural victory**: fitting retains better median accuracy and a lower rate of moderate (>10%) parameter errors, while the current augmented networks reduce the frequency and magnitude of extreme errors in the sparsest case.

The observation duration remains 50 ms. New counts are 1001, 501 and 101 points/channel; 5001-point results are verified historical reuse. All 18 new networks and 58,500 new fitting problems completed. All solves reported convergence, which did not guarantee accurate estimates.

## At 10% noise: median absolute relative error (%)

R / L / C in each cell. Values average per-condition circuit medians over three training seeds and three noise realizations for neural controls, and over three realizations for fitting. These are not pooled medians or confidence intervals.

| Points/channel | Clean-trained MLP | Noise-trained MLP | Fixed-start fitting |
|---|---:|---:|---:|
| 5001 | 6.913 / 3.162 / 4.421 | 2.745 / 1.971 / 2.012 | 0.162 / 0.178 / 0.162 |
| 1001 | 14.362 / 6.566 / 9.360 | 2.856 / 2.280 / 2.222 | 0.360 / 0.398 / 0.366 |
| 501 | 21.053 / 8.976 / 11.516 | 2.611 / 2.349 / 2.213 | 0.493 / 0.552 / 0.502 |
| 101 | 19.088 / 18.959 / 12.903 | 3.709 / 4.221 / 3.510 | 1.327 / 1.850 / 1.569 |

## At 10% noise: tails and means (%)

| Points/channel | Neural mean R / L / C | Fitting mean R / L / C | Neural p95 R / L / C | Fitting p95 R / L / C |
|---|---:|---:|---:|---:|
| 5001 | 3.422 / 2.462 / 2.528 | 0.221 / 0.252 / 0.221 | 8.885 / 6.329 / 6.635 | 0.615 / 0.742 / 0.629 |
| 1001 | 3.687 / 2.843 / 2.950 | 0.485 / 0.572 / 0.489 | 9.960 / 7.400 / 8.340 | 1.387 / 1.702 / 1.364 |
| 501 | 3.333 / 2.974 / 2.912 | 0.690 / 0.791 / 0.680 | 8.909 / 7.903 / 8.092 | 1.969 / 2.304 / 1.943 |
| 101 | 6.093 / 7.541 / 6.089 | 22.891 / 15.412 / 6.128 | 19.981 / 27.074 / 22.947 | 47.734 / 55.398 / 33.015 |

At 101 points, fitting has lower medians but substantially larger R/L means and p95 errors. The C means are very similar; there is no demonstrated material mean-C advantage. Means can be dominated by a small fraction of large relative errors.

## Incidence of inaccurate estimates at 10% noise

Percent of observations for which **any** R/L/C parameter exceeds the indicated error. Reporting thresholds (10%, 50%) are descriptive post-run diagnostics; they were not used for checkpoint selection, fitting settings, or any follow-up tuning. A >10% error is not an optimizer failure. Rates average per-condition circuit proportions across seeds/realizations; repeats are not independent circuits.

| Points/channel | Neural >10% | Fitting >10% | Neural >50% | Fitting >50% |
|---|---:|---:|---:|---:|
| 5001 | 4.36% | 0.00% | 0.00% | 0.00% |
| 1001 | 7.67% | 0.00% | 0.00% | 0.00% |
| 501 | 6.61% | 0.00% | 0.00% | 0.00% |
| 101 | 27.97% | 17.44% | 1.81% | 6.73% |

This distinction matters: at 101 points the network has **more moderate-error cases**, but **fewer extreme-error cases** than this fitting baseline. No claim that it is more reliable under every error definition is supported.

## Clean sparse measurements

At 101 points, fitting exceeds 10% error on at least one parameter in **52 of 1,500 clean circuits (3.47%)**, although all solves report convergence. Its clean medians and p95 errors are nearly zero, hiding these outliers; mean errors are R 18.81%, L 9.77%, C 0.99%. The noise-trained network has >10% errors in 8.67% of clean cases on average across seeds, but >50% in only 0.31%, versus fitting 3.40%.

For the 52 large-error clean fits, median weighted residual cost is 33.1, versus 2.36e-14 for the other clean fits. The known generating parameters reproduce the clean measurements to storage precision. Thus these observed clean failures include poor optimizer solutions; they are not evidence that parameter recovery is mathematically impossible.

## Aliasing and scope of the baseline

At 101 points, 0.5 ms spacing gives a 2 kHz sampling rate and 1 kHz Nyquist frequency. Some allowed circuits oscillate above 1 kHz. These are instantaneous selected points with **no anti-alias filtering**; aliasing and a difficult nonlinear objective are plausible contributors, but this experiment does not isolate their causal contributions.

The conventional baseline uses a single fixed geometric-center initialization. Results do not establish failure of all conventional estimation: multistart, global optimization or better initialization could change the outliers and runtime. No such methods were added or tuned after seeing these test results.

## Processing latency

Single-circuit neural CPU median range across three augmented models; fitting median over all 13 actual conditions. One numerical thread. Neural timing includes preprocessing/inference/decoding; fitting includes conversion/scaling/solve/decoding. Noise generation and disk are excluded. Warm repeated neural requests and full-run fitting timings are differently structured benchmarks.

| Points/channel | Neural CPU median (ms) | Fitting median (ms) |
|---|---:|---:|
| 5001 | 0.048–0.048 | 6.926 |
| 1001 | 0.039–0.039 | 2.862 |
| 501 | 0.030–0.031 | 2.305 |
| 101 | 0.023–0.024 | 1.623 |

The 50 ms acquisition requirement is unchanged: fewer points reduce measurement volume and processing, not recording duration. Historical 5001-point timings were not remeasured. CPU is faster than MPS for these small inference workloads; full device results are in JSON.

## All noise levels: median error (%)

| Points | Noise | Noise-trained MLP R / L / C | Fitting R / L / C |
|---|---|---:|---:|
| 5001 | 0% | 0.958 / 0.677 / 0.815 | 0.000 / 0.000 / 0.000 |
| 5001 | 1% | 1.022 / 0.718 / 0.845 | 0.016 / 0.018 / 0.016 |
| 5001 | 2% | 1.137 / 0.805 / 0.921 | 0.032 / 0.036 / 0.031 |
| 5001 | 5% | 1.660 / 1.208 / 1.280 | 0.081 / 0.090 / 0.080 |
| 5001 | 10% | 2.745 / 1.971 / 2.012 | 0.162 / 0.178 / 0.162 |
| 1001 | 0% | 1.016 / 0.656 / 0.829 | 0.000 / 0.000 / 0.000 |
| 1001 | 1% | 1.051 / 0.715 / 0.873 | 0.036 / 0.041 / 0.037 |
| 1001 | 2% | 1.212 / 0.829 / 0.947 | 0.072 / 0.080 / 0.072 |
| 1001 | 5% | 1.748 / 1.314 / 1.367 | 0.181 / 0.206 / 0.181 |
| 1001 | 10% | 2.856 / 2.280 / 2.222 | 0.360 / 0.398 / 0.366 |
| 501 | 0% | 1.007 / 0.655 / 0.844 | 0.000 / 0.000 / 0.000 |
| 501 | 1% | 1.042 / 0.702 / 0.872 | 0.051 / 0.058 / 0.051 |
| 501 | 2% | 1.136 / 0.810 / 0.953 | 0.105 / 0.113 / 0.103 |
| 501 | 5% | 1.625 / 1.318 / 1.349 | 0.259 / 0.275 / 0.247 |
| 501 | 10% | 2.611 / 2.349 / 2.213 | 0.493 / 0.552 / 0.502 |
| 101 | 0% | 1.483 / 1.388 / 1.269 | 0.000 / 0.000 / 0.000 |
| 101 | 1% | 1.531 / 1.471 / 1.328 | 0.143 / 0.195 / 0.161 |
| 101 | 2% | 1.658 / 1.671 / 1.482 | 0.270 / 0.380 / 0.323 |
| 101 | 5% | 2.287 / 2.494 / 2.127 | 0.683 / 0.950 / 0.790 |
| 101 | 10% | 3.709 / 4.221 / 3.510 | 1.327 / 1.850 / 1.569 |

## Verification and limitations

- Verified all 28 prediction archive hashes, finite 19,500-row predictions, paired IDs/truth/conditions and full-noise hashes; retained-noise hashes match across all seven estimators on each new grid.
- Exact selected indices and time grids checked; historical artifacts copied without modification. Source/data/reference hashes and original checkpoint hash remain unchanged.
- Three matched seeds, same circuit splits; all new checkpoint selections completed before any new test evaluation.
- Fixed original full-training RMS noise scaling, training-only selected-point preprocessing; no test-based tuning.
- Changing input dimension changes parameter count and optimization difficulty. Networks are retrained per grid, not one universal variable-input model.
- Ideal known series-RLC physics, independent white Gaussian noise only, sparse overdamped/near-critical test groups. Real hardware, model mismatch and anti-alias filtering remain outside scope.

## Files

- Protocol: `docs/sampling_density.md`
- Local outputs: `runs/sampling/results.json`, `error_rates.json`, `report.md`, count-specific checkpoints and prediction files
- Committed measured record: `results/sampling/`
- Reproduce: `.venv/bin/python -m scripts.sampling_experiment` (requires fresh output directory and original local reference artifacts).

This completes the agreed core experiment sequence. The evidence supports a nuanced accuracy–speed–outlier trade-off, not a claim that neural estimation universally outperforms physical fitting.
