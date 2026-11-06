# Conventional fitting versus neural estimation

## Finding

With the correct ideal circuit model and full observations, bounded nonlinear least squares is substantially more accurate than the current MLP, including under the tested measurement noise. The neural network is much faster computationally. Neither observation should be generalized to arbitrary hardware or model mismatch.

## Median absolute relative parameter error (%)

R / L / C in each cell. Fitting averages per-realization medians (one clean condition, three noisy realizations). Neural values average per-condition medians across three augmented-training seeds and the same realizations. These are descriptive averages, not pooled medians or confidence intervals.

| Noise | Noise-trained MLP | Conventional fitting |
|---|---:|---:|
| 0% | 0.958 / 0.677 / 0.815 | 1.15e-07 / 2.29e-07 / 2.1e-07 |
| 1% | 1.02 / 0.718 / 0.845 | 0.0157 / 0.0184 / 0.016 |
| 2% | 1.14 / 0.805 / 0.921 | 0.0316 / 0.0364 / 0.0311 |
| 5% | 1.66 / 1.21 / 1.28 | 0.0811 / 0.09 / 0.0802 |
| 10% | 2.75 / 1.97 / 2.01 | 0.162 / 0.178 / 0.162 |

## Tail accuracy at 10% noise

| Metric | Noise-trained MLP R / L / C (%) | Fitting R / L / C (%) |
|---|---:|---:|
| Mean per-condition p95 | 8.88 / 6.33 / 6.63 | 0.615 / 0.742 / 0.629 |

Across **all 19,500 fits**, maximum observed R/L/C errors were 1.53 / 2.05 / 2.3%. All fits converged; all predictions were finite. Convergence alone is not proof of a global optimum, but actual parameter errors were checked. There were 39 fits with at least one active bound.

## Processing time

- Fitting: median **6.926 ms**, p95 **11.209 ms**, mean **7.410 ms** per circuit; 144.50 seconds total solve time.
- Neural CPU: per-checkpoint single-circuit medians **0.047–0.048 ms**, including preprocessing and output decoding, using one Torch thread.
- Approximate median compute-only ratio: **145×** in favor of the MLP. This is a comparison of full-run fitting latencies to warmed neural microbenchmarks, not an identical-workload benchmark.
- Both use a **50 ms observation window**: illustrative acquisition-plus-median-processing times are about **56.93 ms fitting** versus **50.05 ms neural**, excluding acquisition overhead.
- These are serial fitting measurements, not parallel fitting or optimized analytical parameter-extraction timings.

## Why fitting does well here

The response family is known exactly, only three unknowns are estimated, and all 10,002 scalar measurements are used. Independent white noise is averaged through the physical model. By contrast, the MLP approximates an inverse mapping and retains approximation/training error even on clean data. That is a plausible explanation, not a separately proven error decomposition.

## Fairness and verification

- Identical test circuits, measurement grids and physical noise perturbations. Noise hashes, IDs, truths, levels and realization layouts matched the original neural prediction file.
- Dataset, original neural checkpoint and neural-results hashes unchanged.
- Fixed geometric-center start, known parameter bounds, log-coordinate least squares, training-RMS residual weighting; no true labels supplied to fitting.
- All converged and failed estimates included by design; this run had zero failures.
- Known ideal series-RLC physics and bounds are supplied to fitting. The neural approach uses simulated physics during training. Neither has been evaluated on real circuitry.
- Test regime coverage is sparse: six overdamped and one near-critical circuit. No claim about general superiority under other noise models or shorter observations.

## Reproduce / inspect

Protocol: `docs/conventional_fitting.md`. Command: `.venv/bin/python -m scripts.fit_baseline` (requires fresh output directory). Raw outputs: `runs/fitting/manifest.json`, `results.json`, `predictions.npz`. Neural outputs remain under `runs/noise/`.

## Interpretation for the project

The current evidence supports an **accuracy–computation trade-off**, not “neural networks beat conventional estimation.” The already-planned shorter-window and fewer-point studies can test how that trade-off changes when less measurement information is available.
