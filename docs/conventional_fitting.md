# Conventional parameter fitting: paired noise baseline

## Reproduction

Run after the completed neural-noise experiment:

```sh
.venv/bin/python -m scripts.fit_baseline
.venv/bin/python -m unittest discover -s tests -v
```

Inputs are the unchanged `data/` dataset and `runs/noise/manifest.json` /
`results.json`. Output is a new, Git-ignored `runs/fitting/` directory; existing
output directories are refused. Alternative paths can be supplied with
`--data-dir`, `--noise-dir` and `--output-dir`.

## Estimator

Each circuit is estimated independently using SciPy `least_squares`, fitting
both full waveforms to the same ideal analytical series-RLC step response used
by the simulator. This gives fitting an exactly specified physical model; it
does not test hardware-model mismatch.

- Unknowns are natural-log R, L and C, decoded to physical units.
- Bounds match the known dataset ranges: R 10–100 ohms, L 10–50 mH,
  C 1–10 microfarads.
- Every fit starts from the **same geometric range center**:
  R 31.6228 ohms, L 22.3607 mH, C 3.16228 microfarads.
- Residuals are `(predicted-observed)/clean_training_channel_RMS`, stacked across
  both channels and all time points, in float64. This is not per-waveform
  normalization. At positive noise levels it is equivalent to inverse-noise-sigma
  weighting up to one common factor; fixed scaling also works at zero noise and
  avoids changing absolute termination scales between levels.
- Trust-region reflective solver (`trf`), two-point finite-difference Jacobian,
  ordinary squared loss, `ftol=xtol=gtol=1e-10`, maximum 200 function evaluations
  (`--max-nfev` can override this).

There is **no truth-based initialization**, neural initialization, multistart,
condition-to-condition warm start or tuning based on test errors. This is one
specified conventional baseline, not a claim to represent every conventional
estimator or the best possible optimizer configuration.

## Pairing and diagnostics

All 1,500 test circuits are evaluated under the same 13 conditions as the neural
study: one clean condition and three noise realizations at each of 1%, 2%, 5%,
10% training-RMS noise. Float32 perturbation generation, batch layout, addition
rounding, row order and per-condition SHA-256 must agree with the neural study.
Dataset hashes, training-only RMS, noise contract and NumPy version are checked.

All fits, including failures, enter parameter-error metrics. Record solver status,
success flag, message, cost, optimality, active bounds, numerical function/Jacobian
evaluation counts and actual residual-call counts. Numerical Jacobian calls are
not fully represented by SciPy's `nfev`. On solver exceptions the best finite
bounded iterate is retained and marked failed; exception counts are -1 and
optimality is NaN in the diagnostic NPZ.

**Solver convergence does not establish correct parameters or a global optimum.**
Use parameter errors and reconstructed-waveform residuals to interpret results;
never omit high-error converged fits.

## Outputs and timing

- `manifest.json`: solver settings, fixed start, bounds, weights, noise contract,
  reference/data/source hashes and environment.
- `results.json`: per-condition/regime relative errors, convergence counts, bound
  hits and per-circuit elapsed-time summaries. Updated after every condition.
- `predictions.npz`: condition-major physical predictions/truth, circuit IDs,
  levels, realization indices, noise hashes and per-circuit solver diagnostics.

CPU solves are serial with numerical thread environment set to one before imports
in a fresh CLI process, and Torch CPU threads set to one. Per-circuit timing
includes float64 conversion, scaling, optimization and physical output decoding.
It excludes disk reads, synthetic-noise generation and reporting. Report median,
p95, mean and total elapsed time; fitting throughput here is serial throughput,
not a parallel or batched optimizer benchmark.

Neural CPU timings use one thread and include input preprocessing, forward pass,
output decoding and associated transfers. They are repeated warm microbenchmarks;
fitting timings span the actual 19,500 estimation problems. These are useful
measured processing costs, not identical benchmark workloads. Compare neural
**single-circuit** latency to fitting per-circuit latency, not neural batch
throughput. Both estimators still require the same **50 ms observation window**.

## First measured comparison

All 19,500 fits completed and reported convergence; predictions were finite.
Saved IDs, truths, condition/noise hashes and realization order matched the neural
study. Source data, original neural checkpoint and neural-results hashes remained
unchanged. All 23 unit/integration tests passed before the full run.

At 10% training-RMS noise, mean per-realization median R/L/C relative errors were
**0.162 / 0.178 / 0.162%** for fitting. Noise-trained networks averaged
**2.745 / 1.971 / 2.012%** across three training seeds and the same three noise
realizations. Fitting was also far more accurate on clean data. These summaries
are averages of medians, not pooled medians or confidence intervals.

Across all conditions, maximum observed R/L/C fitting errors were
**1.532 / 2.049 / 2.297%**; 39 fits had at least one active bound. Zero solver
failures occurred, but success should still not be treated as proof of a global
optimum in future experiments.

Median per-circuit fitting latency was **6.926 ms** (p95 **11.209 ms**), versus
**0.047–0.048 ms** median for CPU neural inference across checkpoints. The network
was about 145 times faster computationally in these measurements, but both use
50 ms recordings: acquisition-plus-processing is approximately 56.93 versus
50.05 ms, ignoring other acquisition overhead. These are differently structured
benchmarks as described above.

The result supports an **accuracy–computation trade-off**, not neural superiority.
Fitting has the exactly correct ideal physical model and many independent noisy
measurements; real model mismatch and reduced observations remain untested.
Full comparison: `runs/fitting/report.md`; raw metrics and diagnostics are in
`results.json` and `predictions.npz` in that directory. Exact copies of the
manifest, measured results and report are committed under
[`results/fitting/`](../results/fitting/); the binary prediction file stays local
with its hash recorded.
