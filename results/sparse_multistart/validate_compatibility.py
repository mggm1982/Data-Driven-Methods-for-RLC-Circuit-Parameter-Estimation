"""Bounded compatibility check only; never runs the measured experiment.

Run from repository root: .venv/bin/python -m results.sparse_multistart.validate_compatibility
"""
from scripts import sparse_multistart as multi

from pathlib import Path

import numpy as np
import torch


def main():
    directory = Path('results/sparse_multistart')
    manifest = multi.window.read_json(directory / 'manifest.json')
    results = multi.window.read_json(directory / 'results.json')
    snapshots = directory / 'measured_source'
    measured_fit_hash = next(value for path, value in manifest['source_sha256'].items()
                             if Path(path).name == 'fit_baseline.py')
    assert multi.base.digest(snapshots / 'fit_baseline.py') == measured_fit_hash
    assert multi.base.digest(snapshots / 'sparse_multistart.py') == manifest['implementation_sha256']
    assert multi.base.digest(Path('docs/sparse_multistart.md')) == manifest['protocol_sha256']
    for path, digest in manifest['input_sha256'].items():
        assert multi.base.digest(Path(path)) == digest, path
    for name, digest in manifest['dataset_sha256'].items():
        assert multi.base.digest(Path('data') / name) == digest, name

    torch.set_num_threads(1)
    x, y, splits, _ = multi.base.load_data(Path('data'))
    rms = multi.noise.channel_rms(x, splits['train'], 128)
    np.testing.assert_array_equal(rms, manifest['channel_rms_physical'])
    test = splits['test']
    ids = np.array([row['sample_id'] for row in multi.gen.read_metadata(Path('data/circuits.csv'))])[test]
    stats = multi.base.fit_preprocessing(x, y, splits['train'])
    # Real original references path: includes its unmodified frozen-source check,
    # all six historical checkpoints and their prediction archives. No fitting.
    reference, models, _, _ = multi.window.references(
        multi.window.parser().parse_args([]), manifest['dataset_sha256'], rms, ids, y[test], stats)
    assert len(reference) == 13 and len(models) == 6
    test_x = x[test]
    del x
    starts = multi.predetermined_starts()
    np.testing.assert_array_equal(starts, manifest['starts_log'])
    indices = multi.sample.selected_indices(101)
    fixed_rows = [0, 749, 1499]  # Fixed by position, not selected for fit quality.
    checks = []
    for (i, r), record in zip(multi.CONDITIONS, results['conditions']):
        path = Path(record['arrays_path'])
        assert multi.base.digest(path) == record['arrays_sha256'], str(path)
        observed, digests = multi.sample.selected_bank(test_x, rms, 101, 'test', i, r)
        assert digests['full_sha256'] == record['full_sha256']
        assert digests['selected_sha256'] == record['selected_sha256']
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['sample_id'], ids)
            np.testing.assert_array_equal(saved['truth'], y[test])
            np.testing.assert_array_equal(saved['starts_log'], starts)
            np.testing.assert_array_equal(saved['time_s'], multi.gen.TIME[indices])
            for row in fixed_rows:
                actual = multi.fit_five(observed[row], rms, multi.gen.TIME[indices], starts)
                # Exact comparisons, no relaxed tolerance. Latencies are not
                # comparable across sessions and intentionally are not validated.
                for key in ('start_prediction', 'cost', 'prediction', 'selected_start',
                            'status', 'success', 'message', 'nfev', 'njev', 'optimality',
                            'bound_hits', 'residual_calls'):
                    np.testing.assert_array_equal(actual[key], saved[key][row], err_msg=key)
                checks.append({'level_index': i, 'realization': r, 'row': row,
                               'sample_id': str(ids[row]), 'starts_checked': 5,
                               'predictions_and_costs_exact': True,
                               'selected_start': int(actual['selected_start'])})
    multi.gen.write_json(directory / 'compatibility_validation.json', {
        'command': '.venv/bin/python -m results.sparse_multistart.validate_compatibility',
        'scope': 'Fixed rows 0,749,1499 in clean and all three saved 10% realizations; all five starts. No sweep, tuning, training, or replacement of measurements.',
        'reproducer_sha256': multi.base.digest(Path(multi.__file__)),
        'validation_source_sha256': multi.base.digest(Path(__file__)),
        'restored_fitter_sha256': multi.base.digest(Path(multi.fit.__file__)),
        'versions': {'numpy': np.__version__, 'scipy': multi.fit.scipy.__version__,
                     'torch': str(torch.__version__), 'python': multi.fit.platform.python_version()},
        'thread_environment': multi.fit.THREAD_ENV,
        'cpu_threads': torch.get_num_threads(),
        'original_window_references': 'Passed unmodified references(), including original frozen-source check and six checkpoints/prediction archives.',
        'measured_snapshot_and_input_hash_checks': 'passed',
        'comparison': 'Exact array equality for predictions, costs, selection and all non-timing solver diagnostics; timing deliberately excluded.',
        'solves': len(checks) * 5, 'checks': checks,
        'limitation': 'Finite fixed-case compatibility validation, not a new experiment or proof for every possible input; no new performance measurements reported.'})
    print(f'Passed {len(checks) * 5} fixed-case solves and original window references; saved compatibility_validation.json')


if __name__ == '__main__':
    main()
