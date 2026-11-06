"""Frozen five-start sparse diagnostic; python -m scripts.sparse_multistart."""
# Thread limits must precede numerical imports.
from scripts import fit_baseline as fit

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from scripts import sampling_experiment as sample
from scripts import window_experiment as window
from scripts import generate_dataset as gen
from scripts import noise_experiment as noise
from scripts import train_model as base

SEED = 20261019
CONDITIONS = [(0, 0), (4, 0), (4, 1), (4, 2)]


def predetermined_starts():
    rng = np.random.Generator(np.random.PCG64(SEED))
    return np.vstack([fit.START, rng.uniform(*fit.LOG_BOUNDS, size=(4, 3))])


def select_start(costs):
    """First lowest finite residual; no labels or status filtering."""
    costs = np.asarray(costs)
    finite = np.isfinite(costs)
    return int(np.argmin(np.where(finite, costs, np.inf))) if finite.any() else 0


def fit_five(observed, rms, time_s, starts):
    begin = time.perf_counter()
    predictions, diagnostics = [], []
    for start in starts:
        prediction, info = fit.fit_waveform(observed, rms, time_s=time_s,
                                           max_nfev=200, log_start=start)
        predictions.append(prediction)
        diagnostics.append(info)
    costs = [row['cost'] for row in diagnostics]
    selected = select_start(costs)
    arrays = {'start_prediction': np.array(predictions), 'selected_start': selected,
              'prediction': predictions[selected],
              **{key: np.array([row[key] for row in diagnostics]) for key in diagnostics[0]}}
    arrays['full5_elapsed_seconds'] = time.perf_counter() - begin
    return arrays


def accuracy(truth, prediction):
    error = np.abs(prediction - truth) / truth
    return {'count': len(truth), 'component_relative_error': {
        name: {stat: float(value) for stat, value in zip(
            ('median', 'mean', 'p95', 'max'),
            (np.median(error[:, j]), np.mean(error[:, j]),
             np.quantile(error[:, j], .95), np.max(error[:, j])))}
        for j, name in enumerate(gen.RANGES)},
        **{f'any_gt_{threshold}_percent': {'count': int(np.any(error > threshold / 100, axis=1).sum()),
            'rate': float(np.any(error > threshold / 100, axis=1).mean())}
           for threshold in (10, 50)}}


def timing(durations):
    return {f'{name}_seconds': float(value) for name, value in zip(
        ('median', 'mean', 'p95', 'max', 'total'),
        (np.median(durations), np.mean(durations), np.quantile(durations, .95),
         np.max(durations), np.sum(durations)))}


def statuses(status, success):
    return {'status_counts': {str(s): int((status == s).sum()) for s in np.unique(status)},
            'success': int(success.sum()), 'failure': int((~success).sum())}


def summarize(truth, old, arrays):
    n = len(truth)
    selected = arrays['selected_start']
    rows = np.arange(n)
    return {'historical_single': {'accuracy': accuracy(truth, old['prediction']),
                'historical_timing': timing(old['elapsed_seconds']),
                'solver': statuses(old['status'], old['success'])},
            'new_center': {'accuracy': accuracy(truth, arrays['start_prediction'][:, 0]),
                'new_timing': timing(arrays['elapsed_seconds'][:, 0]),
                'solver': statuses(arrays['status'][:, 0], arrays['success'][:, 0])},
            'full5': {'accuracy': accuracy(truth, arrays['prediction']),
                'new_timing': timing(arrays['full5_elapsed_seconds']),
                'selected_solver': statuses(arrays['status'][rows, selected], arrays['success'][rows, selected]),
                'all_starts_solver': statuses(arrays['status'], arrays['success']),
                'per_start_solver': [statuses(arrays['status'][:, j], arrays['success'][:, j]) for j in range(5)],
                'selected_start_counts': np.bincount(selected, minlength=5).tolist(),
                'no_finite_cost_count': int((~np.isfinite(arrays['cost']).any(axis=1)).sum())},
            'center_prediction_max_absolute_difference_from_archive': float(np.max(
                np.abs(arrays['start_prediction'][:, 0] - old['prediction'])))}


def verify_pairing(original, record, path, ids, truth, hashes, rms):
    expected_solver = {**fit.SOLVER, 'max_nfev': 200, 'bounds_physical': fit.BOUNDS.tolist(),
                       'fixed_start_physical': np.exp(fit.START).tolist(), 'coordinates': 'log(R,L,C)'}
    if (original['dataset_sha256'] != hashes or original['solver'] != expected_solver or
            original['channel_rms_physical'] != rms.tolist() or
            original['noise_contract'] != noise.NOISE_CONTRACT or
            original['versions']['numpy'] != np.__version__ or
            original['batch_size'] != 128 or original['realizations'] != 3):
        raise ValueError('Original sparse provenance mismatch')
    reference = [r['noise_sha256'] for r in record['test']]
    window.verify_archive(path, record, ids, truth, 'fitting', reference)
    indices = sample.selected_indices(101)
    with np.load(path, allow_pickle=False) as a:
        if (not np.array_equal(a['selected_indices'], indices) or
                not np.array_equal(a['time_s'], gen.TIME[indices]) or
                not np.array_equal(a['condition_selected_noise_sha256'],
                                   [r['selected_noise_sha256'] for r in record['test']])):
            raise ValueError('Original sparse grid/selected noise mismatch')


def output_paths(run_dir, result_dir):
    protected = [Path(p) for p in ('data', 'runs/mlp', 'runs/noise', 'runs/fitting',
                 'runs/windows', 'runs/sampling', 'results/noise', 'results/fitting',
                 'results/windows', 'results/sampling', 'results/paper_review', 'docs', 'paper', 'scripts', 'tests')]
    run = fit.fresh_output(run_dir, [*protected, result_dir])
    result = fit.fresh_output(result_dir, [*protected, run_dir])
    return run, result


def run():
    run_dir, result_dir = output_paths(Path('runs/sparse_multistart'), Path('results/sparse_multistart'))
    torch.set_num_threads(1)
    original_path = Path('results/sampling/results.json')
    original = window.read_json(original_path)
    record = original['sampling']['101']['models']['fitting']
    old_path = Path('runs/sampling/101points/fitting_predictions.npz')
    hashes = {n: base.digest(Path('data') / n) for n in ('circuits.csv', 'waveforms.npz')}
    x, y, splits, _ = base.load_data(Path('data'))
    test = splits['test']
    if x.dtype != np.float32 or x.shape[1:] != (2, 5001) or len(test) != 1500:
        raise ValueError('Require original full float32 grid and 1500 test circuits')
    rms = noise.channel_rms(x, splits['train'], 128)
    truth = y[test]
    ids = np.array([r['sample_id'] for r in gen.read_metadata(Path('data/circuits.csv'))])[test]
    verify_pairing(original, record, old_path, ids, truth, hashes, rms)
    # Check original full noise manifest and all recorded historical input hashes.
    nm = Path('runs/noise/manifest.json')
    nr = Path('runs/noise/results.json')
    reference = fit.validate_reference(window.read_json(nm), window.read_json(nr), hashes, rms)
    if reference != [r['noise_sha256'] for r in record['test']]:
        raise ValueError('Original sparse/full noise hashes disagree')
    for path, digest in original['reference_sha256'].items():
        if base.digest(Path(path)) != digest:
            raise ValueError(f'Original reference hash mismatch: {path}')
    test_x = x[test]
    del x
    starts = predetermined_starts()
    indices = sample.selected_indices(101)
    for directory in (run_dir, result_dir):
        directory.mkdir(parents=True, exist_ok=False)
    manifest = {'protocol_sha256': base.digest(Path('docs/sparse_multistart.md')),
        'dataset_sha256': hashes,
        'input_sha256': {str(p): base.digest(p) for p in
                        (original_path, old_path, nm, nr, *map(Path, original['reference_sha256']))},
        'source_sha256': {str(Path(m.__file__)): base.digest(Path(m.__file__)) for m in
                         (fit, sample, window, gen, noise, base)},
        'implementation_sha256': base.digest(Path(__file__)),
        'versions': {'python': fit.platform.python_version(), 'platform': fit.platform.platform(),
                     'numpy': np.__version__, 'scipy': fit.scipy.__version__, 'torch': str(torch.__version__)},
        'thread_environment': fit.THREAD_ENV, 'cpu_threads': torch.get_num_threads(),
        'solver': original['solver'], 'start_seed': SEED, 'rng': 'NumPy PCG64',
        'starts_log': starts.tolist(), 'starts_physical': np.exp(starts).tolist(),
        'selection': 'All five always; lowest finite cost, ties first; no finite cost retains center and flags.',
        'conditions': CONDITIONS, 'test_circuits': len(test), 'selected_indices': indices.tolist(),
        'time_s': gen.TIME[indices].tolist(), 'channel_rms_physical': rms.tolist(),
        'noise_contract': noise.NOISE_CONTRACT, 'batch_size': 128,
        'residual': original['residual'], 'bank_contract': original['bank_contract'],
        'timing': 'Historical single separate; new center and full5 same pass, serial single thread. Full5 includes selection and diagnostic assembly; excludes bank/disk/reporting.',
        'clean_inaccurate_definition': 'any component relative error >10%; also report >50%',
        'limitations': 'Fixed finite restart budget, not proof of global optimality or identifiability. Three noisy realizations repeat the same 1500 circuits. No test tuning.'}
    gen.write_json(result_dir / 'manifest.json', manifest)
    results = {'conditions': [], 'completed_conditions': 0}
    saved, historical = [], []
    wall = time.perf_counter()
    with np.load(old_path, allow_pickle=False) as archive:
        for i, r in CONDITIONS:
            condition = noise.conditions(3).index((i, r))
            print(f'Condition level={noise.LEVELS[i]}, realization={r}', flush=True)
            observed, digests = sample.selected_bank(test_x, rms, 101, 'test', i, r)
            ref = record['test'][condition]
            if (digests['full_sha256'] != ref['noise_sha256'] or
                    digests['selected_sha256'] != ref['selected_noise_sha256']):
                raise ValueError('Regenerated full/selected noise hash mismatch')
            rows = []
            for j, waveform in enumerate(observed):
                rows.append(fit_five(waveform, rms, gen.TIME[indices], starts))
                if (j + 1) % 250 == 0:
                    print(f'  {j + 1}/1500 circuits (five starts each)', flush=True)
            arrays = {k: np.array([row[k] for row in rows]) for k in rows[0]}
            old = {k: archive[k][condition * len(test):(condition + 1) * len(test)]
                   for k in ('prediction', 'elapsed_seconds', 'status', 'success')}
            report = {'level': noise.LEVELS[i], 'realization': r, **digests,
                      **summarize(truth, old, arrays)}
            arrays.update(truth=truth, sample_id=ids, level=np.full(len(test), noise.LEVELS[i]),
                          realization=np.full(len(test), r))
            path = run_dir / f'condition_{i}_{r}.npz'
            np.savez_compressed(path, **arrays, starts_log=starts, selected_indices=indices,
                                time_s=gen.TIME[indices])
            report['arrays_path'] = str(path.relative_to(Path.cwd()))
            report['arrays_sha256'] = base.digest(path)
            results['conditions'].append(report)
            results['completed_conditions'] += 1
            saved.append(arrays)
            historical.append(old)
            gen.write_json(result_dir / 'results.json', results)
    pooled = {k: np.concatenate([a[k] for a in saved[1:]]) for k in saved[0]}
    pooled_old = {k: np.concatenate([a[k] for a in historical[1:]]) for k in historical[0]}
    results['pooled_10_percent'] = summarize(pooled['truth'], pooled_old, pooled)
    results['wall_seconds'] = time.perf_counter() - wall
    gen.write_json(result_dir / 'results.json', results)
    write_report(result_dir, results)


def write_report(directory, results):
    lines = ['# Sparse five-start diagnostic', '',
             '101 uniform samples over 50 ms; original 1,500 test circuits. All five starts run; selection uses residual only.',
             'Errors below are percentages. Historical accuracy and timing are archived single-start; new center and full-five timings are contemporaneous.', '',
             '| Condition | Method | Median R/L/C (%) | Mean R/L/C (%) | p95 R/L/C (%) | Max R/L/C (%) | Any >10% | Any >50% | Median / mean / p95 latency (ms) |',
             '|---|---|---|---|---|---|---|---|---|']
    for label, report in [('clean', results['conditions'][0]),
                          *[(f'10% realization {r}', results['conditions'][r + 1]) for r in range(3)],
                          ('10% pooled (repeated circuits)', results['pooled_10_percent'])]:
        for method in ('historical_single', 'new_center', 'full5'):
            row = report[method]
            a = row['accuracy']
            stats = [' / '.join(f'{100 * a["component_relative_error"][name][stat]:.3f}'
                               for name in gen.RANGES) for stat in ('median', 'mean', 'p95', 'max')]
            rates = [f'{100 * a[f"any_gt_{t}_percent"]["rate"]:.2f}% ({a[f"any_gt_{t}_percent"]["count"]}/{a["count"]})' for t in (10, 50)]
            t = row.get('historical_timing', row.get('new_timing'))
            latency = ' / '.join(f'{1000 * t[k + "_seconds"]:.3f}' for k in ('median', 'mean', 'p95'))
            lines.append('| ' + ' | '.join([label, method, *stats, *rates, latency]) + ' |')
    lines += ['', 'Full diagnostics, status counts for all starts and selected solutions, input/array hashes, seeds, settings, runtime totals/maxima and software versions are in `results.json` and `manifest.json`.', '',
              'Clean inaccurate count is the any-component >10% count above, not solver failure. Convergence does not imply parameter accuracy. Five starts do not certify a global optimum; transient under-resolution/aliasing may remain. No test-based tuning was performed. No historical tables or artifacts were replaced.']
    (directory / 'report.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
