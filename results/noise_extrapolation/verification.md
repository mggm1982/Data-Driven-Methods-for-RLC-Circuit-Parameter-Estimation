# Execution and verification

The protocol was written before the evaluator and before test evaluation.
The bounded experiment was run once, successfully, with a 1,800-second tool timeout:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m scripts.noise_extrapolation
```

All six checkpoints completed, each with six conditions (15%/20% x three
realizations), CPU one Torch thread, batches 128. No timing measurements, training,
fitting, checkpoint changes or further experiments were performed.

Checks actually run:

- `.venv/bin/python -m unittest discover -s tests -p 'test_noise_extrapolation.py' -v`: 3 passed before evaluation (an initial fixture target-key typo was corrected).
- `.venv/bin/python -m unittest discover -s tests -v`: **57 tests passed**.
- `python3 paper/generate_tables.py --check`: all 8 original generated tables/reports current.
- `.venv/bin/python -m scripts.noise_extrapolation --check-table`: standalone table current, regenerated from per-condition statistics without inference.
- `git diff --check`: passed.
- `git check-ignore runs/noise_extrapolation/clean_20261014_predictions.npz`: confirms prediction output is ignored.
- A read-only `.venv/bin/python` heredoc called `scripts.noise_extrapolation.verify()` again after evaluation: dataset, original and six matched checkpoints, original prediction hashes/IDs/labels, clean-training preprocessing, RMS and validation-selected identities still match archived records. It also verified every recorded source/protocol hash, all six new prediction-file hashes, `(6,1500,3)` predictions with 1,500 distinct IDs, exact recomputed condition metrics and summary statistics, cross-model condition-hash equality and three distinct realization base-normal hashes. No inference was rerun.
- `git diff --name-only -- scripts/noise_experiment.py scripts/train_model.py scripts/generate_dataset.py`: empty; frozen helpers unchanged.
- `command -v pdflatex`: unavailable. PDF layout, table width/float placement, page count and bibliography rendering were not checked.

Manifest/source hashes are in `manifest.json`; measured condition statistics,
seed summaries and prediction paths/SHA256 are in `results.json`; interpretation
and historical 10% context are in `report.md`. Only ignored
`runs/noise_extrapolation/` contains new per-circuit predictions. Originals are
preserved. No commit was created; awaiting parent verification/commit.
