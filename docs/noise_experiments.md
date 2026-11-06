# Neural measurement-noise experiments

## Scope and reproduction

This stage compares the original clean-trained MLP with matched clean and
noise-augmented controls. Conventional fitting is a separate comparison described
in [the fitting protocol](conventional_fitting.md); shorter windows and fewer points
remain subsequent stages, not part of this experiment. Series-RLC physics, 50 ms
observation duration, 5,001 points per channel, and original circuit splits stay
unchanged. The source dataset and `runs/mlp/model.pt` are never overwritten.

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m scripts.noise_experiment
```

Default output is `runs/noise/` (ignored by Git). The command refuses a nonempty
output directory; use a different `--output-dir` for a new experiment. Training
and evaluation run on MPS when available, otherwise CPU. All settings, versions,
noise definitions, seeds and data/checkpoint hashes are saved in `manifest.json`
and `results.json`. No test results are used for checkpoint selection.

## Noise definition

Each measured channel is its ideal waveform plus independent zero-mean Gaussian
noise at every circuit/time/channel. Physical standard deviation is

`sigma[channel] = level * RMS(clean training waveforms[channel])`.

The fixed RMS includes **all training time points and the voltage DC component**;
it is not the AC-only RMS, per-circuit RMS, or peak. Levels are 0%, 1%, 2%, 5%,
and 10%. Consequently, these are not equal per-circuit SNR levels. Weak-current
circuits can be more difficult than strong-current circuits at the same level.
Noise is not clipped, filtered or forced to zero at t=0. This is a simple white
measurement-noise model, not a model of every real sensor or circuit imperfection.

Input/target preprocessing is fitted on clean training circuits only. During
augmentation, adding `level * RMS / channel_std` noise to standardized inputs is
algebraically equivalent to adding physical noise before preprocessing (apart
from float32 rounding). There is no per-waveform normalization.

## Paired training and model selection

Train three matched pairs, seeds 20261014, 20261015, 20261016:

- **Clean controls:** clean inputs only.
- **Augmented models:** uniformly choose one of the five levels per circuit on
  each epoch, then add fresh noise. The two channels use the same level but
  independent noise draws.

Each pair shares initial weights, shuffled circuit order, clean training-only
preprocessing, architecture and optimizer settings. Augmentation has its own RNG,
so consuming random noise does not alter circuit order or initialization.
Defaults: Adam, LR 0.001, batch 128, at most 200 epochs, patience 25.

**Both controls are selected using mean standardized log-target validation MSE
across the same fixed five-level validation bank**, one realization per level.
Clean validation MSE is recorded separately. The original baseline was selected
on clean validation only; it is a historical reference, not an identically
selected paired control. Do not attribute every difference from that original
model solely to augmentation.

Validation bank root seed: 20261017; test bank root: 20261018. RNG domains are
separate from training. Noise banks are regenerated in batches instead of saving
large waveform copies. Exact regeneration depends on the recorded batch size,
NumPy version, split row order, and the `noise_contract` in the manifest.

## Accuracy outputs

Only after all checkpoints are selected, evaluate each on the same 1,500 held-out
circuits. Use three test-noise realizations at every nonzero level and one clean
condition (13 conditions total). Noise hashes must match across all seven models:
the original checkpoint, three clean controls and three augmented models.

`results.json` records per-realization mean/median/p95/max absolute relative
parameter errors and damping-group counts. Errors are fractions, not percentages.
Three noise realizations are repeated measurements of the same circuits, **not**
4,500 independent test circuits. Likewise, variation across three training seeds
is descriptive, not a population confidence interval.

Each `<model_key>_predictions.npz` contains physical-unit `prediction` and `truth`
columns (R, L, C), `sample_id`, `level`, `realization`, `model_key`, target names and
condition noise hashes. These support paired failure analysis and future fitting
comparisons. No exactly critical circuits are in the random test set; overdamped
and near-critical groups remain very small.

## Latency

Every checkpoint is benchmarked on CPU (one Torch thread by default) and MPS if
available. There are five warmups and 50 timed repetitions for a single circuit
and a batch of 128. Median and p95 elapsed times, actual batch size and median
throughput are recorded. Device synchronization brackets measurements.

Timing includes input preprocessing, CPU-to-device transfer, forward pass,
device-to-CPU transfer and physical-unit output decoding. It excludes file loading,
noise generation, checkpoint loading, training and warmup. Inputs are preloaded
clean signals: these are compute benchmarks, not noise-level-specific latency
experiments. No CPU/MPS speedup is assumed in advance; small requests can favor CPU.

Observation duration is still **50 ms**, in addition to processing time. Batched
throughput is not single-request latency, and these repeated microbenchmarks are
not complete hardware-acquisition measurements.

## First completed run

All six paired models trained successfully on MPS; the original plus six new
checkpoints passed evaluation (13 conditions each). Source-data/original-model
hashes and identical test-noise hashes across models were verified. All saved
predictions were finite. All 18 unit/integration tests passed.

At 10% noise, matched clean controls versus augmented models had R/L/C errors:

- Mean of per-condition medians: **6.91 / 3.16 / 4.42%** versus
  **2.75 / 1.97 / 2.01%**.
- On clean inputs, the corresponding mean medians were **0.79 / 0.68 / 0.68%**
  versus **0.96 / 0.68 / 0.82%**: improved noise robustness has a modest clean-data
  trade-off, especially for R and C.

These summaries average circuit-level medians across three model seeds and,
for nonzero noise, three realizations; they are not pooled medians or confidence
intervals. Detailed tails and all levels are recorded in `runs/noise/results.json`
and the generated `runs/noise/report.md`. Exact copies of the measured JSON files
and report are committed under [`results/noise/`](../results/noise/); large binary
outputs remain local with their hashes recorded.

Across all seven checkpoints, measured median single-circuit latency was
**0.047–0.048 ms on CPU** versus **0.466–0.576 ms on MPS**. Batch-of-128 median
latency was **0.772–0.976 ms on CPU** versus **1.131–1.670 ms on MPS**. CPU won
for this small network and these request sizes; these timings do not establish
performance against conventional fitting. The observation duration remains 50 ms.
