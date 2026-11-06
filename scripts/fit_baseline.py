"""Fixed-start physical least-squares baseline. Run: python -m scripts.fit_baseline.

Solver success is NOT evidence of correct parameters or a global optimum.
No selection, truth-dependent initialization, warm starts, or parallel solves.
"""
import os

# Set before numerical imports in a fresh CLI process; no extra dependencies.
THREAD_ENV = {key: '1' for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                                  'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS',
                                  'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
os.environ.update(THREAD_ENV)

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.optimize import least_squares
import torch

from scripts import generate_dataset as gen
from scripts import train_model as base
from scripts import noise_experiment as noise

BOUNDS = np.array(list(gen.RANGES.values()), dtype=np.float64).T
LOG_BOUNDS = np.log(BOUNDS)
START = LOG_BOUNDS.mean(axis=0)
SOLVER = dict(method='trf', jac='2-point', loss='linear', ftol=1e-10,
              xtol=1e-10, gtol=1e-10)


def fit_waveform(observed, rms, time_s=gen.TIME, max_nfev=200):
    """Time conversion, fixed scaling, solve and physical prediction decoding."""
    start = time.perf_counter()
    observed = np.asarray(observed, dtype=np.float64)
    scale = np.asarray(rms, dtype=np.float64)[:, None]
    calls, best_cost, best = 0, float('inf'), START.copy()

    def residual(log_parameters):
        nonlocal calls, best_cost, best
        calls += 1
        parameters = np.clip(np.exp(log_parameters), *BOUNDS)
        value = ((np.asarray(gen.response(*parameters, time_s)) - observed) / scale).ravel()
        cost = .5 * np.dot(value, value)
        if np.isfinite(cost) and cost < best_cost:
            best_cost, best = float(cost), log_parameters.copy()
        return value

    try:
        result = least_squares(residual, START.copy(), bounds=LOG_BOUNDS,
                               max_nfev=max_nfev, **SOLVER)
        parameters = np.clip(np.exp(result.x), *BOUNDS)
        info = {key: getattr(result, key) for key in
                ('status', 'success', 'message', 'nfev', 'njev', 'cost', 'optimality')}
        info['bound_hits'] = result.active_mask
    except Exception as error:
        # Retain the best finite bounded iterate, never omit failed circuits.
        parameters = np.clip(np.exp(best), *BOUNDS)
        info = dict(status=-1, success=False, message=f'{type(error).__name__}: {error}',
                    nfev=-1, njev=-1, cost=best_cost, optimality=np.nan,
                    bound_hits=np.where(best <= LOG_BOUNDS[0], -1,
                                        np.where(best >= LOG_BOUNDS[1], 1, 0)))
    info.update(residual_calls=calls, elapsed_seconds=time.perf_counter() - start)
    return parameters, info


def bank_inputs(x, rms, level_index, realization, batch_size=128):
    """Original bank layout: float32 perturbation AND physical add before cast."""
    rng = noise.noise_rng('test', level_index=level_index, realization=realization)
    batches, digest = [], hashlib.sha256()
    for start in range(0, len(x), batch_size):
        batch = x[start:start + batch_size]
        perturbation = noise.add_noise(np.zeros_like(batch), noise.LEVELS[level_index], rms, rng)
        digest.update(perturbation.tobytes())
        batches.append(batch + perturbation)
    return np.concatenate(batches), digest.hexdigest()


def fit_condition(observed, rms, max_nfev=200, progress=False):
    predictions, diagnostics = [], []
    for index, waveform in enumerate(observed):
        predicted, info = fit_waveform(waveform, rms, max_nfev=max_nfev)
        predictions.append(predicted)
        diagnostics.append(info)
        if progress and (index + 1) % 250 == 0:
            print(f'  {index + 1}/{len(observed)} solves', flush=True)
    return {'prediction': np.array(predictions),
            **{key: np.array([row[key] for row in diagnostics]) for key in diagnostics[0]}}


def summarize(arrays):
    durations = arrays['elapsed_seconds']
    return {'convergence': {'success': int(arrays['success'].sum()),
                            'failure': int((~arrays['success']).sum()),
                            'status_counts': {str(s): int((arrays['status'] == s).sum())
                                              for s in np.unique(arrays['status'])},
                            'circuits_at_bounds': int(np.any(arrays['bound_hits'] != 0, axis=1).sum())},
            'timing': {'median_seconds': float(np.median(durations)),
                       'p95_seconds': float(np.quantile(durations, .95)),
                       'mean_seconds': float(durations.mean()),
                       'total_seconds': float(durations.sum()),
                       'circuits_per_second': float(len(durations) / durations.sum())}}


def validate_reference(manifest, results, hashes, rms):
    if manifest['dataset_sha256'] != hashes:
        raise ValueError('Dataset hashes do not match noise experiment')
    if (manifest['noise_contract'] != noise.NOISE_CONTRACT or
            manifest['versions']['numpy'] != np.__version__ or
            manifest['levels'] != list(noise.LEVELS) or
            manifest['settings']['batch_size'] != 128 or
            manifest['settings']['realizations'] != 3):
        raise ValueError('Incompatible noise contract/version/settings')
    if not np.array_equal(np.asarray(manifest['channel_rms_physical']), rms.astype(np.float64)):
        raise ValueError('Training RMS recomputation disagrees with noise manifest')
    expected = [(noise.LEVELS[i], r) for i, r in noise.conditions(3)]
    reference = None
    for model in results.values():
        reports = model['test']
        if [(row['level'], row['realization']) for row in reports] != expected:
            raise ValueError('Reference condition layout mismatch')
        hashes = [row['noise_sha256'] for row in reports]
        if reference is not None and hashes != reference:
            raise ValueError('Reference model noise hashes disagree')
        reference = hashes
    if reference is None:
        raise ValueError('Missing reference condition hashes')
    return reference


def fresh_output(output, protected):
    output = output.resolve()
    for path in protected:
        path = path.resolve()
        if output == path or path in output.parents or output in path.parents:
            raise ValueError('Output overlaps a protected directory')
    if output.exists():
        raise ValueError('Refusing to overwrite existing output directory')
    return output


def run(args):
    output = fresh_output(args.output_dir, (args.data_dir, args.noise_dir, Path('runs/mlp'),
                                           Path('data'), Path('runs/noise')))
    torch.set_num_threads(1)
    manifest_path, results_path = args.noise_dir / 'manifest.json', args.noise_dir / 'results.json'
    original = json.loads(manifest_path.read_text())
    reference_results = json.loads(results_path.read_text())
    hashes = {name: base.digest(args.data_dir / name) for name in ('circuits.csv', 'waveforms.npz')}
    if hashes != original['dataset_sha256']:
        raise ValueError('Dataset hashes do not match noise experiment')
    x, y, splits, damping = base.load_data(args.data_dir)
    if x.dtype != np.float32 or len(splits['test']) != 1500:
        raise ValueError('Require original float32 data and all 1500 test circuits')
    rms = noise.channel_rms(x, splits['train'], 128)
    reference = validate_reference(original, reference_results, hashes, rms)
    test = splits['test']
    test_x = x[test]
    del x
    ids = np.array([row['sample_id'] for row in gen.read_metadata(args.data_dir / 'circuits.csv')])[test]
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'dataset_sha256': hashes, 'noise_manifest_sha256': base.digest(manifest_path),
                'noise_results_sha256': base.digest(results_path),
                'source_sha256': {str(p): base.digest(p) for p in
                                  (Path(__file__), Path(gen.__file__), Path(base.__file__), Path(noise.__file__))},
                'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                             'scipy': scipy.__version__, 'torch': str(torch.__version__),
                             'platform': platform.platform()},
                'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'solver': {**SOLVER, 'max_nfev': args.max_nfev, 'bounds_physical': BOUNDS.tolist(),
                           'fixed_start_physical': np.exp(START).tolist(), 'coordinates': 'log(R,L,C)'},
                'residual': 'C-order stacked (prediction-observed)/fixed clean training channel RMS, float64; sigma weighting up to global level factor; defined at zero.',
                'channel_rms_physical': rms.tolist(), 'channel_order_units': original['channel_order_units'],
                'target_names': list(gen.RANGES), 'levels': list(noise.LEVELS),
                'noise_contract': noise.NOISE_CONTRACT, 'batch_size': 128, 'realizations': 3,
                'selection': 'None; fixed geometric center for every solve; all failures retained in metrics.',
                'cpu_threads': torch.get_num_threads(), 'thread_environment': THREAD_ENV,
                'thread_note': 'Run in fresh CLI process so environment precedes numerical imports; serial solves.',
                'timing_note': 'Per circuit includes float64 conversion, scaling, solve, physical decode; excludes bank generation, disk and reporting. Throughput is serial solve throughput.',
                'caveat': 'Solver success != correct parameters or global optimum. Realizations repeat the same circuits. nfev excludes numerical-Jacobian calls; residual_calls counts them. Exception nfev/njev=-1, optimality=NaN in NPZ.'}
    gen.write_json(output / 'manifest.json', manifest)
    result, saved = {'test': [], 'completed_conditions': 0}, []
    for condition, (i, r) in enumerate(noise.conditions(3)):
        print(f'Condition {condition + 1}/13: level={noise.LEVELS[i]}, realization={r}', flush=True)
        observed, digest = bank_inputs(test_x, rms, i, r, original['settings']['batch_size'])
        if digest != reference[condition]:
            raise ValueError('Regenerated noise hash differs from reference')
        arrays = fit_condition(observed, rms, args.max_nfev, progress=True)
        del observed
        result['test'].append({'level': noise.LEVELS[i], 'realization': r,
                               'noise_sha256': digest,
                               'regimes': base.metrics(y[test], arrays['prediction'], damping[test]),
                               **summarize(arrays)})
        arrays.update(truth=y[test], sample_id=ids, level=np.full(len(test), noise.LEVELS[i]),
                      realization=np.full(len(test), r), model_key=np.full(len(test), 'fitting'))
        saved.append(arrays)
        result['completed_conditions'] += 1
        gen.write_json(output / 'results.json', result)
        print(f"  {result['test'][-1]['timing']}", flush=True)
    combined = {key: np.concatenate([a[key] for a in saved]) for key in saved[0]}
    combined.update(condition_noise_sha256=np.array(reference), target_names=np.array(list(gen.RANGES)))
    np.savez_compressed(output / 'predictions.npz', **combined)
    result.update(predictions_sha256=base.digest(output / 'predictions.npz'), **summarize(combined))
    gen.write_json(output / 'results.json', result)
    print(f'Baseline saved to {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--noise-dir', type=Path, default=Path('runs/noise'))
    parser.add_argument('--output-dir', type=Path, default=Path('runs/fitting'))
    parser.add_argument('--max-nfev', type=int, default=200)
    args = parser.parse_args()
    if args.max_nfev < 1:
        parser.error('--max-nfev must be positive')
    run(args)


if __name__ == '__main__':
    main()
