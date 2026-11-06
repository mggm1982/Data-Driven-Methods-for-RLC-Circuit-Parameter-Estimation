# Final unseen-amplitude Gaussian stress test

Fullgrid5001, 50 ms, original 1,500 test circuits; six fixed checkpoints.
All errors/rates below are percent: mean +/- sample SD over three seed
averages, each averaging three realization statistics (not pooled errors).

| Noise | Model | Statistic | R | L | C |
|---|---|---|---|---|---|
| 15% | clean | median | 10.417 +/- 0.153 | 4.709 +/- 0.145 | 6.643 +/- 0.224 |
| 15% | clean | mean | 15.072 +/- 0.152 | 6.211 +/- 0.167 | 10.033 +/- 0.170 |
| 15% | clean | p95 | 44.264 +/- 0.698 | 17.147 +/- 0.436 | 31.010 +/- 0.475 |
| 15% | clean | max | 190.077 +/- 24.655 | 50.842 +/- 4.068 | 86.650 +/- 2.344 |
| 15% | clean | any_gt10 | 58.652 +/- 1.000 | — | — |
| 15% | clean | any_gt50 | 4.341 +/- 0.269 | — | — |
| 15% | augmented | median | 3.926 +/- 0.379 | 2.824 +/- 0.340 | 2.887 +/- 0.374 |
| 15% | augmented | mean | 4.909 +/- 0.479 | 3.559 +/- 0.414 | 3.618 +/- 0.419 |
| 15% | augmented | p95 | 12.718 +/- 1.349 | 9.280 +/- 1.004 | 9.631 +/- 1.034 |
| 15% | augmented | max | 30.239 +/- 2.361 | 27.729 +/- 6.399 | 20.141 +/- 2.228 |
| 15% | augmented | any_gt10 | 16.037 +/- 4.880 | — | — |
| 15% | augmented | any_gt50 | 0.000 +/- 0.000 | — | — |
| 20% | clean | median | 13.681 +/- 0.127 | 6.164 +/- 0.186 | 8.762 +/- 0.303 |
| 20% | clean | mean | 20.608 +/- 0.284 | 8.166 +/- 0.210 | 13.156 +/- 0.245 |
| 20% | clean | p95 | 62.447 +/- 1.314 | 22.520 +/- 0.570 | 40.331 +/- 0.598 |
| 20% | clean | max | 325.056 +/- 43.723 | 71.247 +/- 7.698 | 113.351 +/- 1.290 |
| 20% | clean | any_gt10 | 69.652 +/- 0.584 | — | — |
| 20% | clean | any_gt50 | 9.059 +/- 0.228 | — | — |
| 20% | augmented | median | 5.133 +/- 0.485 | 3.715 +/- 0.415 | 3.720 +/- 0.460 |
| 20% | augmented | mean | 6.451 +/- 0.611 | 4.656 +/- 0.535 | 4.710 +/- 0.543 |
| 20% | augmented | p95 | 16.851 +/- 1.802 | 12.119 +/- 1.217 | 12.544 +/- 1.463 |
| 20% | augmented | max | 39.646 +/- 4.072 | 38.495 +/- 8.668 | 27.128 +/- 2.803 |
| 20% | augmented | any_gt10 | 31.393 +/- 6.634 | — | — |
| 20% | augmented | any_gt50 | 0.022 +/- 0.038 | — | — |

## Original 10% context (different streams, not same-realization paired)

Historical MPS evaluation used root 20261018; new CPU evaluation uses root
20261020. These are descriptive cross-amplitude comparisons, not paired
realization differences. Original mean realization/seed medians R/L/C (%):
- clean: 6.913 / 3.162 / 4.421
- augmented: 2.745 / 1.971 / 2.012

## Interpretation and limits

Augmentation reduces errors relative to matched clean controls at both new
amplitudes, but errors increase beyond the original 10% context; this is not
amplitude-invariant robustness. Individual condition metrics, including maxima
and strict any-component >10%/>50% rates, are retained in results.json.
The displayed max is an average of condition maxima, not a global maximum.
The three realizations repeat the same 1,500 circuits, not 4,500 independent
circuits. Three-seed SD is descriptive, not a confidence interval. No new
parameter distribution, noise type, fitting, training, model selection or
latency study was used. Two unseen Gaussian amplitudes do not establish
unseen-type, hardware, model-mismatch or general OOD robustness.

Prediction binaries are only in ignored runs/noise_extrapolation; their SHA256
hashes are recorded in results.json. Provenance/stream hashes are in manifest.json.
