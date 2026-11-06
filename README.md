# Clean series-RLC dataset

This project generates noiseless series-RLC training data and trains a first waveform-to-parameter
MLP. Measurement-noise experiments and a conventional least-squares fitting baseline
are available; parasitics and hardware simulation are not implemented.
`IDEA.md` is the original project idea.

## Reproduce

From the repository root (tested versions are pinned in `requirements.txt`):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/generate_dataset.py metadata
.venv/bin/python scripts/generate_dataset.py pilot
.venv/bin/python scripts/generate_dataset.py waveforms
```

All commands accept `--data-dir PATH` (default `data`). `metadata` overwrites the CSV
and resets the configuration; regenerate waveforms afterward to keep them aligned.
`waveforms` reads the saved CSV without resampling, reruns the pilot before the full
simulation, validates the persisted NPZ, and records measured results in the config.
Allow roughly 1 GB of RAM and 401 MB of disk space for waveform generation.
The uncompressed NPZ stays local and is ignored by Git.

## Physics and sampling

Known **series RLC**, ideal 1 V step at t=0, zero initial current and capacitor voltage:

- `di/dt = (1 - R*i - v_C)/L`
- `dv_C/dt = i/C`
- Steady state: `i -> 0 A`, `v_C -> 1 V`.

10,000 independent log-uniform draws: R = 10–100 ohms, L = 0.010–0.050 H,
C = 1–10 microfarads. Parameter seed: **20261010**; shuffled split seed: **20261011**
(NumPy PCG64). Exactly 7,000 train / 1,500 validation / 1,500 test circuits;
each unique circuit belongs to one split. Damping is not stratified.

## Files

- `data/circuits.csv`: `sample_id`, `resistance_ohm`, `inductance_h`,
  `capacitance_f`, `damping_ratio`, `split`. R/L/C are target labels;
  damping ratio is derived analysis metadata, not model input.
- `data/dataset_config.json`: physics, units, ranges, seeds, counts, versions,
  and actual validation measurements. This is documentation, not model input.
- `data/waveforms.npz`: CSV-order `sample_id` `(10000,)`, shared float64 `time_s`
  `(5001,)`, float32 `current_a` and `capacitor_voltage_v` `(10000, 5001)` each.
  Time spans 0–50 ms inclusive at 10 microseconds. Signals are in physical units,
  not peak-normalized. Both channels retain information needed to identify R/L/C;
  voltage alone identifies RC and LC, not all three parameters individually.

## Numerical validation

Float64 analytical responses cover underdamped, critically damped, and overdamped
circuits. Sinc handles the underdamped critical limit; stable roots and `expm1`
avoid cancellation/overflow on the overdamped side. Signals are cast only for storage.
Independent `solve_ivp` DOP853 checks use `rtol=1e-11`, `atol=1e-14`.
Tests cover exact/near-critical cases, all range corners, deterministic metadata,
invalid metadata, and persisted CSV round trips.

Before full generation, the pilot checks eight range corners, three critical/near-critical
cases, and twelve seeded CSV samples. It compares float32 responses to the independent
ODE solver and checks linear interpolation against a 1-microsecond analytical grid.
The resolution gate is 0.5% of peak current and 0.005 V; reference gates are 1e-8 A
and 1e-6 V, allowing float32 storage error. These are pilot checks, not a proof of
global interpolation bounds. Saved waveforms are checked for shape, dtype, finite
values, zero initial states, and ID/time alignment; fifteen persisted rows receive
independent reference checks. All circuits also receive a 200 ms steady-state check.

Do not require exact settling at 50 ms: the slowest permitted underdamped envelope
is approximately `exp(-5)`. End-of-window residuals are reported explicitly.
The uniform grid describes sampled signals, not every possible between-sample extremum.
Future noisy copies must remain in their original circuit split to avoid leakage.

## Measured generation results

Generated using Python 3.14.8, NumPy 2.5.3, and SciPy 1.18.1. All seven tests passed.

- Metadata: 10,000 unique IDs and parameter combinations; split counts 7,000/1,500/1,500.
  Damping ratios span 0.0255123–1.348526: 9,933 underdamped, 67 overdamped,
  no exactly critical draws, and 14 within `abs(zeta - 1) < 0.01`.
- 23-case pilot: maximum float32-vs-ODE errors **9.12e-10 A / 5.96e-8 V**.
  Maximum interpolation errors on the 10x finer grid: **0.2285% of peak current**
  and **0.001242 V**. Pilot residuals at 50 ms: **9.44e-5 A / 0.005992 V**.
- Saved dataset: both signals have shape `(10000, 5001)` and float32 dtype.
  All rows passed finite-value, zero-initial-state, and CSV-ID-order checks;
  the shared time grid matches exactly.
- Fifteen persisted rows (damping extrema, nearest critical, and seeded random rows)
  agree with the independent ODE reference within **4.66e-10 A / 5.97e-8 V**.
- Largest saved end-of-window residuals: **6.52e-5 A / 0.004429 V**.
  Checking every circuit analytically at 200 ms gives residuals at most
  **6.11e-12 A / 8.75e-10 V**, confirming the expected long-time steady state.
- Local `data/waveforms.npz`: **400,201,044 bytes** (about 400 MB decimal).
  Exact measurements and numerical settings are in `data/dataset_config.json`.

## Neural-network baseline (PyTorch / Apple Metal)

Run from the repository root; use module invocation:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m scripts.train_model
# Optional CPU run, kept separate from the MPS checkpoint:
.venv/bin/python -m scripts.train_model --device cpu --output-dir runs/mlp-cpu
```

`--device auto` uses Apple's Metal Performance Shaders (`mps`) when available,
otherwise CPU. Explicit `--device mps` requires a working MPS installation.
Verified here with an Apple M3 Pro, macOS 27.0, Python 3.14.8 and PyTorch 2.14.1;
a real MPS forward/backward/Adam step and full training run passed.

The model flattens both complete channels (current first, then voltage) into
10,002 inputs: `Linear(10002,128) -> ReLU -> Linear(128,64) -> ReLU -> Linear(64,3)`.
One mean and standard deviation per input channel are
fitted across training circuits and time points only, then shared across all
circuits. This is **not** individual waveform normalization: amplitude information
is retained. Targets are natural-log R/L/C, standardized using training labels
only. Outputs are inverse-transformed to physical units, without clipping to
parameter bounds. Damping ratios and circuit IDs are never model inputs.

Defaults: seed 20261014, Adam with learning rate 0.001, batch size 128, at most
200 epochs, and early stopping after 25 epochs without improved validation MSE
in standardized log-target space. The best validation checkpoint is reloaded
before the test split is evaluated once. Test errors do not select checkpoints.
Seeds are recorded, but bit-identical MPS training across runs/platforms is not
promised. Future model choices should use validation, not these test results.

Local, Git-ignored outputs (overwritten when reusing the output directory):

- `runs/mlp/model.pt`: best weights, architecture, training-only preprocessing,
  target names, time grid, seed, PyTorch version, and CSV/NPZ SHA-256 hashes.
- `runs/mlp/metrics.json`: settings, train/validation loss history, timings, and
  test mean/median/95th-percentile/maximum absolute relative error per parameter,
  including damping-regime counts. Relative errors are fractions, not percentages.

For inference, `scripts.train_model.load_checkpoint(path, device)` returns the
model and preprocessing statistics. Feed `(N, 2, 5001)` float32 signals in A/V
on the recorded time grid to `transform_inputs`, convert to a torch tensor,
call `predict` in batches, then `decode` to obtain R (ohms), L (H), C (F).
The loader does not automatically validate new signals or compare dataset hashes;
callers must enforce channel order, units, grid, and provenance.

### First measured run

The first MPS run stopped at epoch 114 and selected epoch 89. Training took
20.42 s; batched prediction for 1,500 held-out circuits took 17.68 ms including
input/output device transfers, excluding data loading and preprocessing.
These are single-run timings, not a CPU speedup or conventional-fitting comparison.

| Parameter | Median relative error | 95th percentile | Maximum |
| --- | ---: | ---: | ---: |
| R | 0.947% | 3.051% | 14.426% |
| L | 0.659% | 2.093% | 8.196% |
| C | 0.705% | 2.468% | 11.809% |

The test set contains 1,494 underdamped and only 6 overdamped circuits, no exactly
critical circuits, and 1 near-critical circuit (`abs(zeta - 1) < 0.01`; overlapping
with the other groups). Small regime groups do not support robust generalization
claims. This is a noiseless neural baseline, not evidence that a network is more
accurate or faster than conventional fitting.

## Measurement-noise study

Run `.venv/bin/python -m scripts.noise_experiment` to compare the original model
with three matched clean/noise-augmented training pairs at 0%, 1%, 2%, 5%, and 10%
Gaussian noise relative to fixed clean-training channel RMS. This leaves the
source dataset and original model unchanged. Outputs go to `runs/noise/`;
nonempty output directories are not overwritten.

See [the experiment protocol](docs/noise_experiments.md) for noise units,
validation selection, paired test realizations, prediction files and CPU/MPS
latency definitions. Reduced observations remain a future stage. The test suite
includes noise/RNG/provenance and synthetic end-to-end experiment checks.

Measured results, manifests and readable reports are archived in
[`results/noise/`](results/noise/). Model checkpoints and prediction binaries stay
local in `runs/noise/`; their hashes are recorded in the archived results.

## Conventional fitting comparison

Run `.venv/bin/python -m scripts.fit_baseline` after the neural-noise study.
This fits both channels with bounded nonlinear least squares from a fixed initial
guess on the identical noisy test inputs, retaining failed solves in the results.
See [the fitting protocol](docs/conventional_fitting.md) for weighting, solver
settings, diagnostics, timing boundaries and limitations. Outputs go to a fresh
`runs/fitting/` directory; existing models and measurements are unchanged.

Measured results, manifest and comparison report are archived in
[`results/fitting/`](results/fitting/). Per-circuit binary predictions remain local
in `runs/fitting/`, with their hash recorded in the archived results.

## Shorter observation windows

Run `.venv/bin/python -m scripts.window_experiment` after the two preceding
studies. Compare matched-input clean/noise-trained networks and conventional
fitting at 1, 2, 5, 10, 20 and 50 ms while retaining the original 10-microsecond
sampling interval and physical noise amplitudes. The 50 ms results are verified
and reused, not rerun. See [the window protocol](docs/observation_windows.md) for
exact prefix pairing, retraining, provenance and timing. Outputs use a fresh
`runs/windows/` directory. Reducing measurement points at a fixed duration is a
separate subsequent experiment. Measured results and the comparison report are
archived in [`results/windows/`](results/windows/); checkpoints and per-circuit
prediction binaries remain local, with hashes recorded.

## Fewer measurement points at fixed duration

Run `.venv/bin/python -m scripts.sampling_experiment` to compare neural and
conventional estimators at 5001, 1001, 501 and 101 points per channel over the
same 50 ms record. The 5001-point results are verified and reused; reduced-input
networks are retrained. Physical noise amplitudes stay fixed and both methods
receive identical selected noisy samples. No anti-alias filter is applied.
See [the sampling protocol](docs/sampling_density.md) for assumptions, pairing,
validation selection, diagnostics and timing. Outputs use a fresh
`runs/sampling/` directory. Measured results, descriptive outlier rates and the
comparison report are archived in [`results/sampling/`](results/sampling/).
The core noise/window/sampling study is complete; see the report for the
median-versus-tail accuracy trade-off, not an unqualified neural-superiority claim.
