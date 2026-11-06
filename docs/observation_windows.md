# Observation-window experiment

## Question and scope

How does parameter-estimation accuracy change when the transient is observed for
less time? Compare matching clean-trained and noise-trained MLPs against the
same fixed-start nonlinear fitting baseline on exactly the same measurements.
This is not yet the fewer-measurement-points experiment: sampling interval stays
fixed at **10 microseconds**, and both channels are retained.

| Window (ms) | Points per channel, including t=0 | Neural input features |
| --- | ---: | ---: |
| 1 | 101 | 202 |
| 2 | 201 | 402 |
| 5 | 501 | 1002 |
| 10 | 1001 | 2002 |
| 20 | 2001 | 4002 |
| 50 | 5001 | 10002 |

## Reproduction

Requires completed local neural-noise and conventional fitting runs, including
checkpoints and prediction binaries, not just archived JSON summaries:

```sh
.venv/bin/python -m scripts.window_experiment
.venv/bin/python -m unittest discover -s tests -v
```

Output defaults to a fresh `runs/windows/` directory (Git-ignored). Existing output
or protected input directories cannot be overwritten. The script verifies source
and historical artifact hashes before accepting reuse. `--output-dir` selects a
new directory; selected-window/seed and training-budget flags are available for
smoke tests, but the full study uses defaults.

## Data and noise pairing

Use the original circuit splits, labels, full-grid source waveforms, noise levels
0%, 1%, 2%, 5%, 10%, and three test realizations at nonzero levels (clean once).
Noise amplitudes and fitting residual weights use the **original full-duration
clean-training channel RMS**, not a recalculated short-window noise scale.

For each validation/test condition, draw float32 noise in the original full
`(batch, 2, 5001)` layout, using original batch size 128, seeds and row order,
**then crop** to the chosen time prefix. Drawing a shorter array with the same
seed would not reproduce the same channel/circuit noise offsets. Verify original
full-condition noise hashes, and record prefix hashes separately. Add clean and
noise prefixes in float32 once; fitting and every network receive the same
rounded physical measurement array. Shorter windows are exact prefixes of the
longer recorded conditions.

Validation is one fixed realization per level, independently seeded from test
and training. Repeated noisy observations do not create new independent circuits.

## Neural model and selection

For each of 1, 2, 5, 10 and 20 ms, train three matched clean/augmented pairs using
the existing seeds 20261014–20261016. Reuse the completed six paired models at
50 ms. Do not treat the historical original clean model as an extra window seed.

Only input dimension changes: hidden widths remain 128 and 64, ReLU activations,
three standardized log-parameter outputs. Each window has clean-training-prefix
input mean/std; target preprocessing uses unchanged clean training labels. There
is no per-waveform peak normalization. Smaller inputs also reduce first-layer
parameter counts, so this is a matching-input estimator study, not a strictly
constant-parameter architecture comparison.

Pairs share initialization and shuffled training-circuit order within a window.
Noise-trained models choose one of the five noise levels uniformly per circuit
per epoch and add fresh independent noise. The dedicated augmentation RNG is
independent of shuffle/initialization. Its draw consumption depends on prefix
size: **online training noise is not exactly paired across windows**. Exact
cross-window noise pairing applies to validation/test measurements.

Defaults remain Adam, learning rate 0.001, batch 128, at most 200 epochs and
25 epochs of patience. Both controls are selected by the mean standardized
log-target validation MSE across five fixed original-bank prefixes. Clean
validation MSE is recorded separately. All new checkpoints for all requested
windows are selected before any new test evaluation; no settings are chosen
from test errors.

This evaluates the accuracy achievable after retraining for a known input
window. It does not evaluate missing inputs fed unchanged into the original
fixed-size network or one universal model supporting variable windows.

## Conventional estimation

Use the unchanged fitting method: log-coordinate bounded least squares,
geometric-center fixed start, original full-training RMS residual weights,
`trf` with two-point Jacobian, linear loss, tolerances 1e-10, maximum 200 function
evaluations. Pass the exact short time-grid prefix and both short measurements.
No multistart, truth-based initialization, neural initialization or warm starts.
Retain failures and all diagnostic arrays; convergence is not parameter accuracy.

## Timing and 50 ms reuse

New neural checkpoints have warmed, repeated CPU/MPS single-circuit and batch128
benchmarks including preprocessing/transfer/forward/decode. Fitting records actual
per-circuit conversion/scaling/solve/decode times and diagnostics; noise generation
and disk/reporting are excluded. Numerical thread budget is one in a fresh CLI
process. Short fits may expose solver-overhead floors.

Report observation duration **separately from processing time**, and add them
when estimating earliest answer availability. Compare single-circuit latency,
not neural batch throughput, to serial fitting latency.

The completed 50 ms models, fitting predictions, metrics and timings are copied
only after verifying hashes and paired layouts. They are not retrained, refitted,
reevaluated or rebenchmarked. Label their timings as historical; they were not
measured contemporaneously with shorter-window runs.

## Outputs and interpretation

`runs/windows/results.json` records settings, source/data/reference hashes,
versions, noise contracts, per-window model dimensions, training histories,
per-condition/regime errors, fit diagnostics/timing summaries and benchmarks.
Subdirectories such as `1ms/` contain models and per-method prediction NPZ files.
Each completed method/window has 19,500 condition-major prediction rows and
original test IDs/truths; new prefixes record both full and prefix noise hashes.

Once verified, archive results and a readable comparison report under
`results/windows/` and commit code/tests/protocol/results together, following
`AGENTS.md`. Large checkpoints and prediction binaries remain local.

Interpret accuracy using median/tail errors and failure cases, not just average
error or solver success. Hold the noise definition fixed in physical units; avoid
claiming that a short recording is equivalent to a longer recording at a different
noise level. Sparse overdamped/near-critical test groups and ideal simulated
physics remain limitations. Fewer measurement points at fixed 50 ms duration is
the next separate experiment, after reviewing and committing this one.

## First completed run

All 30 new neural models and 97,500 new fitting problems completed; every new
fit reported convergence. The 50 ms artifacts were verified and reused. All
42 prediction archives passed hash, finite-output, row-layout and pairing checks;
source/data/reference hashes remained unchanged. All 30 tests passed before the
full run.

At 10% original-training-RMS noise, mean per-condition median R/L/C errors were:

| Window | Noise-trained MLP (%) | Conventional fitting (%) |
| --- | ---: | ---: |
| 1 ms | 2.933 / 0.816 / 0.928 | 0.830 / 0.370 / 0.382 |
| 2 ms | 2.310 / 0.983 / 0.869 | 0.305 / 0.242 / 0.207 |
| 5 ms | 1.614 / 0.967 / 1.017 | 0.192 / 0.190 / 0.171 |
| 10 ms | 1.825 / 1.314 / 1.303 | 0.168 / 0.180 / 0.165 |
| 20 ms | 2.077 / 1.417 / 1.407 | 0.163 / 0.178 / 0.162 |
| 50 ms (reused) | 2.745 / 1.971 / 2.012 | 0.162 / 0.178 / 0.162 |

Neural summaries average medians across three model seeds and three noise
realizations; fitting averages across the same realizations. These are not pooled
medians or confidence intervals. See the archived report for clean-control errors,
tails, all noise levels, and single-circuit latency plus observation duration.

Conventional fitting stays more accurate and largely plateaus by 10–20 ms.
The current networks often benefit from shorter input records, but input dimension,
parameter count and learning difficulty also change. This does not show that extra
measurements inherently reduce information or establish a universally optimal
window. No test-based follow-up tuning was performed.

Measured results and readable report are committed under
[`results/windows/`](../results/windows/). Models and prediction binaries remain
local under `runs/windows/`, with hashes recorded in the results.
