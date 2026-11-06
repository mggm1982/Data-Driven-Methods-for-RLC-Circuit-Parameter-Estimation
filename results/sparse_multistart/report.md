# Sparse five-start diagnostic

101 uniform samples over 50 ms; original 1,500 test circuits. All five starts run; selection uses residual only.
Errors below are percentages. Historical accuracy and timing are archived single-start; new center and full-five timings are contemporaneous.

| Condition | Method | Median R/L/C (%) | Mean R/L/C (%) | p95 R/L/C (%) | Max R/L/C (%) | Any >10% | Any >50% | Median / mean / p95 latency (ms) |
|---|---|---|---|---|---|---|---|---|
| clean | historical_single | 0.000 / 0.000 / 0.000 | 18.807 / 9.769 / 0.991 | 0.000 / 0.000 / 0.000 | 879.400 / 396.437 / 96.094 | 3.47% (52/1500) | 3.40% (51/1500) | 1.614 / 1.885 / 3.451 |
| clean | new_center | 0.000 / 0.000 / 0.000 | 18.807 / 9.769 / 0.991 | 0.000 / 0.000 / 0.000 | 879.400 / 396.437 / 96.094 | 3.47% (52/1500) | 3.40% (51/1500) | 1.616 / 1.867 / 3.398 |
| clean | full5 | 0.000 / 0.000 / 0.000 | 0.005 / 0.005 / 0.005 | 0.000 / 0.000 / 0.000 | 4.582 / 4.577 / 4.806 | 0.00% (0/1500) | 0.00% (0/1500) | 8.604 / 9.852 / 16.126 |
| 10% realization 0 | historical_single | 1.323 / 1.848 / 1.522 | 23.369 / 15.647 / 6.209 | 47.421 / 54.804 / 31.804 | 879.400 / 396.437 / 138.938 | 17.93% (269/1500) | 6.47% (97/1500) | 1.634 / 1.873 / 3.388 |
| 10% realization 0 | new_center | 1.323 / 1.848 / 1.522 | 23.369 / 15.647 / 6.209 | 47.421 / 54.804 / 31.804 | 879.400 / 396.437 / 138.938 | 17.93% (269/1500) | 6.47% (97/1500) | 1.640 / 1.883 / 3.386 |
| 10% realization 0 | full5 | 1.288 / 1.770 / 1.475 | 4.544 / 6.225 / 5.435 | 22.272 / 30.254 / 26.423 | 131.796 / 188.176 / 138.937 | 15.13% (227/1500) | 3.07% (46/1500) | 8.952 / 9.948 / 16.228 |
| 10% realization 1 | historical_single | 1.327 / 1.876 / 1.604 | 22.295 / 15.123 / 6.052 | 46.576 / 54.522 / 35.198 | 879.400 / 396.437 / 115.501 | 17.07% (256/1500) | 6.60% (99/1500) | 1.627 / 1.862 / 3.388 |
| 10% realization 1 | new_center | 1.327 / 1.876 / 1.604 | 22.295 / 15.123 / 6.052 | 46.576 / 54.522 / 35.198 | 879.400 / 396.437 / 115.501 | 17.07% (256/1500) | 6.60% (99/1500) | 1.656 / 1.912 / 3.491 |
| 10% realization 1 | full5 | 1.274 / 1.768 / 1.565 | 4.639 / 6.265 / 5.308 | 24.071 / 31.631 / 28.651 | 167.195 / 156.719 / 115.500 | 14.20% (213/1500) | 3.40% (51/1500) | 9.048 / 10.157 / 16.642 |
| 10% realization 2 | historical_single | 1.332 / 1.825 / 1.582 | 23.008 / 15.467 / 6.124 | 49.205 / 56.870 / 32.045 | 879.400 / 396.437 / 135.324 | 17.33% (260/1500) | 7.13% (107/1500) | 1.629 / 1.865 / 3.312 |
| 10% realization 2 | new_center | 1.332 / 1.825 / 1.582 | 23.008 / 15.467 / 6.124 | 49.205 / 56.870 / 32.045 | 879.400 / 396.437 / 135.324 | 17.33% (260/1500) | 7.13% (107/1500) | 1.662 / 1.913 / 3.376 |
| 10% realization 2 | full5 | 1.268 / 1.736 / 1.551 | 4.495 / 6.131 / 5.321 | 23.068 / 31.165 / 26.237 | 136.714 / 151.132 / 135.324 | 14.47% (217/1500) | 3.73% (56/1500) | 9.059 / 10.120 / 16.282 |
| 10% pooled (repeated circuits) | historical_single | 1.327 / 1.845 / 1.576 | 22.891 / 15.412 / 6.128 | 48.367 / 55.785 / 33.711 | 879.400 / 396.437 / 138.938 | 17.44% (785/4500) | 6.73% (303/4500) | 1.630 / 1.867 / 3.368 |
| 10% pooled (repeated circuits) | new_center | 1.327 / 1.845 / 1.576 | 22.891 / 15.412 / 6.128 | 48.367 / 55.785 / 33.711 | 879.400 / 396.437 / 138.938 | 17.44% (785/4500) | 6.73% (303/4500) | 1.652 / 1.903 / 3.418 |
| 10% pooled (repeated circuits) | full5 | 1.279 / 1.757 / 1.529 | 4.559 / 6.207 / 5.355 | 23.902 / 31.152 / 26.866 | 167.195 / 188.176 / 138.937 | 14.60% (657/4500) | 3.40% (153/4500) | 9.012 / 10.075 / 16.329 |

## Finding and verification

Initialization accounts for a substantial part of the original sparse tail: five starts remove all 52 clean >10% cases and halve the noisy >50% incidence (6.73% to 3.40%). The noisy extreme tail remains, and is still above the historical augmented-network 1.81%; that network's >10% incidence is higher (27.97% versus 14.60%). Historical neural rates average the original seeds/realizations; the new fitting rates pool the equally sized three original realizations.

Completed exactly four conditions, 6,000 circuit-condition evaluations and 30,000 starts, in 60.496 seconds wall time including bank generation and output. Every start reported success; no condition had an all-nonfinite-cost circuit. New center predictions equal the historical archive exactly in all four conditions. Full and selected noise hashes matched their originals. No data, splits, checkpoints or original measured result archives were replaced.

Verification performed:
- `.venv/bin/python -m unittest discover -s tests -p 'test_sparse_multistart.py' -v`: 4 passed before execution.
- `.venv/bin/python -m scripts.sparse_multistart`: completed the authorized diagnostic once.
- `.venv/bin/python -m unittest discover -s tests -v`: all 46 passed after execution.
- `python3 paper/generate_tables.py --check`: all 8 generated tables/reports up to date.
- `git diff --check`: passed.
- Independent read-only Python assertions recomputed every recorded input/source/protocol/array SHA256, checked all four `(1500,5,3)` prediction and `(1500,5)` cost shapes, residual-minimum selection, selected prediction indexing and zero historical-center differences: passed.

Full diagnostics, status counts for all starts and selected solutions, input/array hashes, seeds, settings, runtime totals/maxima and software versions are in `results.json` and `manifest.json`. No LaTeX compilation/layout verification was performed.

Clean inaccurate count is the any-component >10% count above, not solver failure. Convergence does not imply parameter accuracy. Five starts do not certify a global optimum; transient under-resolution/aliasing may remain. No test-based tuning was performed. No historical tables or artifacts were replaced.
