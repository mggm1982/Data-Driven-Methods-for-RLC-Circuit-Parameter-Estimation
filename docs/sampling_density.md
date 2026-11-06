# Measurement-point density at fixed observation duration

## Research question

How do neural and conventional parameter estimators behave when fewer instantaneous
measurements are available during the **same 50 ms** transient? This final core
experiment separates sampling density from the earlier observation-window study.

| Points/channel, including both endpoints | Original-grid stride | Spacing | Sampling rate |
| --- | ---: | ---: | ---: |
| 5001 (verified historical reuse) | 1 | 10 microseconds | 100 kHz |
| 1001 | 5 | 50 microseconds | 20 kHz |
| 501 | 10 | 100 microseconds | 10 kHz |
| 101 | 50 | 500 microseconds | 2 kHz |

Both current and capacitor-voltage channels are retained. The selected grids are
uniform, nested subsets of the original recorded grid, all including t=0 and
50 ms. This does not change circuit splits, labels, physical parameter ranges,
step input or recording duration.

## Sampling and aliasing assumptions

These are **instantaneous selected point measurements**, not averages or a
bandwidth-limited acquisition process. No anti-alias filter is applied before
selection. Aliasing can occur, especially at 101 points (2 kHz sampling rate,
1 kHz Nyquist frequency), because some permitted circuits oscillate above 1 kHz.
The study intentionally measures this stated acquisition scenario; it does not
claim to simulate every physical sensor/ADC chain.

The conventional model is evaluated at the exact selected time points, not at a
newly interpolated dense grid. An optimizer can converge to incorrect parameters
when measurements are ambiguous, or can fail to meet its termination criteria.
Record convergence and parameter error separately. Neural estimators can also
fail on sparse/ambiguous measurements; no superiority is assumed.

## Protocol

For each of 1001, 501 and 101 points, train three matched clean/noise-augmented
MLP pairs (seeds 20261014–20261016). Keep hidden widths 128/64, ReLU activations,
log-target transform, Adam, learning rate 0.001, batch 128, maximum 200 epochs
and patience 25. Only the input dimension and corresponding first-layer parameter
count change. Input preprocessing is fitted on clean training circuits at the
selected points only; target preprocessing uses the unchanged training labels.

Noise levels remain 0%, 1%, 2%, 5%, 10% of the **original full-grid clean-training
channel RMS**, not a recomputed sparse-grid RMS or each waveform's amplitude.
Noise-trained models sample one level uniformly per circuit per epoch, with
fresh independent Gaussian draws. Augmentation uses a dedicated RNG; paired
initialization and shuffled circuit order are preserved within a grid. Online
training noise is not exactly paired across different input shapes.

For validation/test, generate original full-sized float32 noise batches in the
original batch128/circuit/channel/time layout, then retain the selected indices.
Retain full-bank and selected-noise hashes. Add physical signals and noise in
float32 once; both fitting and every network receive the exact same measurements.
Coarser conditions are exact subsets of the denser noisy recorded conditions.

Select both neural controls using mean standardized log-target validation MSE
across five fixed validation-noise banks, one realization per level. Finish
**all** new checkpoint selection before any new test evaluation. Test on the same
1500 circuits, one clean condition and three realizations at every nonzero level
(13 conditions, 19500 prediction rows/model/grid). Repetitions are not new
independent circuits.

Conventional fitting retains the original fixed geometric-center start, physical
bounds, log coordinates, original full-training RMS residual weights, SciPy TRF
solver with two-point finite-difference Jacobian and linear loss, 1e-10 tolerances,
and at most 200 function evaluations. No label-informed starts, multistart,
neural starts, warm starts or test-based tuning. Failures are retained in error
metrics and saved diagnostics.

The completed six matched 5001-point neural models and fitting results are
verified against artifact/provenance hashes and copied unchanged. No training,
fitting, test predictions or benchmarks are repeated for that grid. Historical
original clean-model results are not included as an additional training seed.

## Reproduction and outputs

Requires the original local dataset plus completed `runs/noise/` and
`runs/fitting/` artifacts, including checkpoint and prediction binaries:

```sh
.venv/bin/python -m scripts.sampling_experiment
.venv/bin/python -m unittest discover -s tests -v
```

Output defaults to a fresh, Git-ignored `runs/sampling/` directory. Existing or
protected input directories cannot be overwritten. `--output-dir` selects a new
destination. Count/seed/budget options permit smoke tests; the core study uses
all defaults. Older source files remain frozen so their recorded provenance
hashes remain valid.

`results.json` records selected indices/time arrays, versions, settings, source,
dataset and historical-result hashes, model dimensions, original RMS, noise
contracts, selection/augmentation assumptions, per-condition/regime errors,
training histories, fit convergence/timing and neural CPU/MPS benchmarks.
Subdirectories such as `101points/` hold checkpoints and per-method prediction
NPZ files. Each new archive includes full and selected noise hashes plus selected
indices/time points; fitting archives also retain per-circuit diagnostics.
Historical full-grid archives are copied without altering their old schemas.

After verification, archive measured JSON and a readable report under
`results/sampling/` and commit implementation, tests, protocol and results as
one completed experiment, following `AGENTS.md`. Large binaries remain local.

## Timing and interpretation

Use repeated warmed neural single-circuit CPU/MPS latency (including
preprocessing/transfer/inference/decoding) and actual serial fitting latency
(including float64 conversion/scaling/solve/decoding). Thread budget is one;
noise generation and disk/reporting are excluded. These are differently structured
benchmarks, not identical workloads. Historical 5001-point timings are explicitly
not contemporaneous remeasurements.

Every estimate still waits for a **50 ms recording**. Fewer points can reduce data
volume and processing cost without reducing observation duration. Report this
separately from shorter windows. Accuracy should include median/tail/worst errors,
solver status, and difficult cases—not only averages or convergence counts.

Changing input size also changes total network parameter count and learning
difficulty; this is separately trained matching-input estimation, not robustness
of one unchanged network to dropped inputs. Sparse overdamped/near-critical test
groups, ideal matched physics and independent white noise remain limitations.
No optimizer changes or extra tuning should be triggered merely to obtain a
preferred winner from the held-out test results.

## First completed study

All 18 new neural models and 58,500 new fitting problems completed. Every fit
reported convergence. Historical 5001-point results were verified and copied;
28 prediction archives passed hashes, finite-output, paired-layout and retained
noise-hash checks. Data/reference/source and original-checkpoint hashes remained
unchanged. All 36 tests passed before the full run.

At 10% original-training-RMS noise, mean per-condition median R/L/C errors were:

| Points/channel | Noise-trained MLP (%) | Conventional fitting (%) |
| --- | ---: | ---: |
| 5001 (reused) | 2.745 / 1.971 / 2.012 | 0.162 / 0.178 / 0.162 |
| 1001 | 2.856 / 2.280 / 2.222 | 0.360 / 0.398 / 0.366 |
| 501 | 2.611 / 2.349 / 2.213 | 0.493 / 0.552 / 0.502 |
| 101 | 3.709 / 4.221 / 3.510 | 1.327 / 1.850 / 1.569 |

Neural values average medians across three training seeds and three test-noise
realizations; fitting averages over the same realizations. These are not pooled
medians or confidence intervals. Fitting retains better median accuracy.

The 101-point grid reveals rare but severe fixed-start fitting errors: at 10%
noise, mean per-condition p95 R/L/C errors are **47.73 / 55.40 / 33.02%** for
fitting versus **19.98 / 27.07 / 22.95%** for augmented networks. However, estimates
with any parameter over 10% error are **more frequent for the network** (27.97%
versus 17.44%); errors exceeding 50% are less frequent (1.81% versus 6.73%).
These thresholds are descriptive post-run reporting, not selection criteria.

Even on clean 101-point inputs, 52/1500 fits exceed 10% error on some parameter
despite reporting convergence. Their residual costs are far above those of the
accurate fits, demonstrating poor optimizer solutions in this setting—not proving
that the information is fundamentally insufficient. The single-start solver is
not representative of every conventional method; multistart/better initialization
could alter the accuracy/runtime trade-off. Aliasing is possible but not isolated
as the cause of every bad estimate.

At 101 points, augmented-network CPU median latency across seeds was
0.023–0.024 ms, versus 1.623 ms median fitting latency. The required acquisition
still lasts 50 ms. Full errors, outlier rates and timing definitions are archived
in [`results/sampling/`](../results/sampling/); large binaries remain local.

This completes the agreed core study: noise, observation duration and measurement
density, each comparing neural and conventional estimators. No follow-up tuning
or new experiment was performed to force a preferred outcome.
