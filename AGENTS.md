# Project workflow

- User preference: after each authorized experiment is completed and verified,
  commit its implementation, tests, protocol, and measured results together before
  starting the next experiment, unless the user asks otherwise.
- Keep separate experiments in chronological commits. Stage explicit paths; do not
  include unrelated user changes. Run relevant tests and `git diff --check` before
  committing. Report the commit hash and any limitations.
- Archive measured JSON results/manifests and readable reports under `results/`.
  Keep large datasets, model checkpoints and per-circuit prediction binaries local
  under the existing ignored paths; record their hashes for provenance.
- Preserve original circuit splits, source noiseless data, and earlier checkpoints.
  Never tune model choices on held-out test results. Do not expand experiment scope
  merely to obtain more favorable results.
