# Descriptive transient diagnostics (protocol fixed before generation)

Concerns 18/19 only. Read existing local prediction archives; no training, fitting,
new predictions, checkpoint/model selection, or changes to frozen source scripts.
Use the original 1,500 CSV-order test circuits. Verify dataset and prediction SHA256
against measured manifests, exact IDs/truth/conditions, model keys, full noise
hashes and available prefix/selected noise hashes and grids. Record input,
implementation, test and protocol hashes. Large inputs remain local/ignored.

From truth in physical units compute tau=2L/R seconds and
fd=sqrt(1/(LC)-(R/(2L))^2)/(2*pi) Hz only where the radicand is positive
(underdamped). Exclude critical/overdamped circuits from both binned plots and
associations; separately account for all 1,500 circuits and report their errors.
Fixed tau edges [0.2,0.5,1,2,5,10] ms and fd edges [0,0.5,1,2,5] kHz.
Bins are left-closed/right-open except the final upper endpoint is included.
If truth frequencies exceed 5 kHz, replace only the last edge with infinity.
No bins or plotted curves are selected from errors; abort on uncovered underdamped
truths. Empty bins have count zero and null statistics, not zero error.

Dense windows: clean-trained and augmented MLPs, all three seeds
20261014/15/16 at all 1/2/5/10/20/50 ms, discovering existing saved prediction
paths. Sparse101: historical single-start fitter, archived five-start selected
fitter, and all three augmented MLP seeds. Clean once; 10% original-training-RMS
noise in realizations 0/1/2. No historical original MLP as an extra seed.

For each model/seed/condition and each bin report count, component R/L/C median
absolute relative error, median of circuit-wise maximum component error, and
counts/rates any component >10% and >50%. Also report all/underdamped/critical/
overdamped groups. Errors/rates are percentages. For every statistic average
realizations within seed, then seeds; descriptive sample seed SD (n-1), not a CI.
Fitters have one seedless series: realization mean, no seed SD. Preserve individual
condition counts in summary JSON; aggregated counts are means, not independent
circuits. Optional Spearman correlations use tau/fd versus max-component error,
per condition on underdamped circuits; no p-values or causal interpretation.

Pre-stated standalone HTML inline-SVG plots: sparse single vs five starts, clean
and 10%, any-component >10% rates against tau and frequency (two panels); dense
1/5/50 ms clean-trained and augmented max-component median against tau, separate
clean and 10% panels. All other windows/seeds retained in summaries/report. CSS
uses --fg/--muted/--border/--accent, no hardcoded colors/fonts. No dependencies,
publication graphics, figure inclusion or giant new per-circuit files required.

Report measured associations and limitations in manuscript and response rows
18/19: descriptive reused test circuits, correlated physical parameters, changing
network dimensions/training effects, no isolated decay/aliasing cause or
prospectively optimal window. Run mathematical/bin/aggregation/pairing tests,
full tests, existing table check, generation and git diff --check. Do not commit
or begin any new experiments until parent verification/commit.
