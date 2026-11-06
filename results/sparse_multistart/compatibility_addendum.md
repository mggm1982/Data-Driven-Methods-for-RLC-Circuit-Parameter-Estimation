# Post-measurement compatibility addendum

The measured run used the exact sources now preserved in `measured_source/`, with hashes matching the unchanged `manifest.json`. It did **not** use the subsequently corrected reproducer. See `amendment.json` for the original-to-snapshot mapping and corrected source hashes.

The measured explicit-start extension to `scripts/fit_baseline.py` broke the frozen-source hash check used by the historical window/sampling code. That file is now restored exactly to HEAD and the archived fitting/window/sampling source hash. The explicit-start implementation is instead duplicated locally in `scripts/sparse_multistart.py`; the frozen fitter and its provenance checks remain untouched.

Compatibility validation, not a new experiment: fixed test positions 0, 749 and 1499 in the saved clean condition and all three saved 10%-noise realizations, all five starts (60 solves). Predictions, costs, selected starts and every non-timing solver diagnostic matched the saved measurements exactly. Snapshot/input/dataset/protocol/archive and regenerated noise hashes passed. The original `window.references()` also passed, including frozen-source checks and six historical checkpoint/prediction archives. Details: `compatibility_validation.json`; reproduce with `.venv/bin/python -m results.sparse_multistart.validate_compatibility`.

All 47 unit tests passed; `paper/generate_tables.py --check` found all eight generated tables/reports current; `git diff --check` passed. No full experiment rerun, sweep, tuning, training, or new performance claim. Fixed-case equivalence does not prove equivalence on every possible input; timings were intentionally excluded.

The measured manifest, results, report, prediction arrays and `docs/sparse_multistart.md` protocol remain unchanged. This separate addendum does not revise the measured protocol or imply that corrected source was used during measurement. No commit was created.
