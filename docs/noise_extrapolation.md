# Final bounded unseen-amplitude stress test (fixed before evaluation)

Run `.venv/bin/python -m scripts.noise_extrapolation` once. This is the final
bounded experiment; no further experiment or selection is authorized here.

## Locked scope

Use the original fullgrid5001, 0–50 ms record and original 1,500 test circuits,
with the six existing matched clean/augmented validation-selected checkpoints
for seeds 20261014, 20261015 and 20261016 from `runs/noise/`. No retraining,
checkpoint/architecture/threshold selection, fitting, new parameter draws,
other noise types, clipping or filtering. Preserve source data and all originals.
Evaluate on CPU, one Torch thread, batches of 128; no timing study.

Only amplitudes 15% and 20% are added. Physical Gaussian sigma is amplitude
times the original clean-training channel RMS over every time point, including
voltage DC, in current-A / capacitor-voltage-V order. Recompute with the frozen
helper and require exact agreement with the archived RMS and preprocessing.

Use three new independent streams: NumPy PCG64 SeedSequence([20261020, 2, r]),
r=0,1,2. Reset the same stream for each amplitude and model. Thus base float32
standard normals are shared across amplitudes and all six estimators; different
realizations are separate streams. Generate in CSV test order, fixed batches of
128, C order (circuit, channel, time). Exactly follow frozen `add_noise` order:
float32 normals * float32 amplitude * float32 training RMS; add to physical
float32 clean waveform, then frozen preprocessing/inference/decode. No clipping.
Hash concatenated base normals, physical perturbations and noisy input batches.
Require identical condition hashes across all six models and base-normal hashes
across amplitudes. NumPy version and batching are part of regeneration contract.

## Provenance gates and outputs

Before evaluation verify dataset file SHA256 against `results/noise/manifest.json`,
original checkpoint SHA256, time grid and target names; all six checkpoint hashes
against `results/noise/results.json`, seeds, augmentation identities, original
five-level validation selection, selected epochs/scores and exact clean-training
preprocessing. Verify archived prediction-file hashes, split IDs/order and truth
labels against original data; record split-ID/label hashes for all three splits.

Save new per-circuit predictions only under ignored `runs/noise_extrapolation/`;
refuse existing nonempty outputs. Archive manifest, results and readable report
under `results/noise_extrapolation/`, including artifact and protocol/code hashes.
Record each seed/amplitude/realization's R/L/C median, mean, p95, maximum absolute
relative errors and any-component >10% and >50% rates (strict >). For every
statistic, average three realization statistics within each seed, then report
mean +/- sample SD across three seeds (ddof=1). Do not pool medians or interpret
averaged maxima as a global worst case; retain individual condition maxima.
Generate standalone `paper/tables/noise_extrapolation.tex`: 15/20% x CMLP/NMLP,
R/L/C median errors as mean +/- seed SD, in percent. Support read-only table check.

Compare descriptively with archived original 10% values, explicitly different
noise streams (root 20261018), not same-realization paired with the new levels;
historical evaluation used MPS, whereas this evaluation uses CPU. Three realizations
repeat the same circuits, not 4,500 independent circuits; seeds are descriptive,
not confidence intervals. This tests only two unseen amplitudes of the same
Gaussian family within the original parameter distribution, not unseen noise
types, hardware, model mismatch, or general OOD robustness. Update manuscript and
review response row 20 accordingly. Run full unit tests, both generated-table
checks and `git diff --check`; do not commit or begin another experiment.
