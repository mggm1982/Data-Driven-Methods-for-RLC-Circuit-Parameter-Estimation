# RLC measurement-noise experiment results

## Main finding

Noise augmentation substantially improves robustness at larger noise levels, with a modest clean-data accuracy trade-off. This is not yet a conventional-fitting comparison.

## Median absolute relative error (%)

Values below are the arithmetic mean of per-condition circuit medians across three training seeds and, at nonzero noise, three noise realizations. They are not pooled medians or confidence intervals. Original checkpoint values average only over noise realizations.

| Noise level | Original R / L / C | Matched clean R / L / C | Noise-trained R / L / C |
|---|---:|---:|---:|
| 0% | 0.95 / 0.66 / 0.71 | 0.79 / 0.68 / 0.68 | 0.96 / 0.68 / 0.82 |
| 1% | 1.25 / 0.75 / 0.90 | 1.19 / 0.77 / 0.92 | 1.02 / 0.72 / 0.84 |
| 2% | 1.83 / 0.95 / 1.28 | 1.79 / 0.98 / 1.30 | 1.14 / 0.81 / 0.92 |
| 5% | 3.69 / 1.75 / 2.47 | 3.73 / 1.74 / 2.46 | 1.66 / 1.21 / 1.28 |
| 10% | 6.85 / 3.19 / 4.39 | 6.91 / 3.16 / 4.42 | 2.75 / 1.97 / 2.01 |

## Tail errors: mean of per-condition 95th percentiles (%)

| Noise level | Matched clean R / L / C | Noise-trained R / L / C |
|---|---:|---:|
| 0% | 3.03 / 2.35 / 2.41 | 3.16 / 2.39 / 2.65 |
| 1% | 3.94 / 2.61 / 3.17 | 3.23 / 2.51 / 2.67 |
| 2% | 6.15 / 3.18 / 4.90 | 3.53 / 2.66 / 2.91 |
| 5% | 13.95 / 6.10 / 10.88 | 5.10 / 3.82 / 3.97 |
| 10% | 27.72 / 11.53 / 21.15 | 8.88 / 6.33 / 6.63 |

## Latency

Range across all seven checkpoints; each checkpoint has five warmups and 50 measurements. Inputs are preloaded. Includes preprocessing, transfer, inference, decoding and synchronization; excludes recording, generation, loading and training. CPU uses one Torch thread.

| Device | Single-circuit median (ms) | Single-circuit p95 (ms) | Batch-of-128 median (ms) |
|---|---:|---:|---:|
| CPU | 0.047–0.048 | 0.049–0.051 | 0.772–0.976 |
| MPS | 0.466–0.576 | 0.622–1.409 | 1.131–1.670 |

CPU is faster for this small MLP in both measured request sizes; Metal acceleration is not automatically beneficial. These compute times are in addition to the unchanged 50 ms observation window. Benchmarks use clean inputs and are not noise-level-specific.

## Protocol and limits

- 10% means Gaussian sigma = 0.1 × fixed clean-training channel RMS, not 10% of each waveform. Channel RMS: 0.00124291074 A and 1.00255024 V. At 10%, sigma is approximately 124.3 µA and 0.1003 V.
- Three matched clean/noise-trained pairs; paired initialization/shuffle, fixed five-level validation selection for both controls. Original historical model used clean-only validation selection.
- All seven models receive the same held-out noisy inputs. Noise hashes and prediction-file hashes verified. Source data and original model hashes are unchanged.
- 1,500 independent test circuits, not 4,500 independent circuits when repeated with noise. Test set has only six overdamped and one near-critical circuit.
- White independent Gaussian noise only; no hardware measurements, bandwidth effects, correlated noise or circuit-model mismatch tested.
- No accuracy claim against conventional fitting yet. Shorter windows and fewer points are not part of this run.

## Files

- `runs/noise/results.json`: full per-realization/regime errors, training histories and benchmarks.
- `runs/noise/manifest.json`: protocol, environment, seeds, RMS and provenance.
- `runs/noise/*_predictions.npz`: paired physical-unit predictions, truth, IDs, conditions and noise hashes.
- `docs/noise_experiments.md`: reproduction and interpretation.
