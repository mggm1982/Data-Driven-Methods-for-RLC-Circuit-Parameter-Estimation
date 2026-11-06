"""Fixed-50ms instantaneous point sampling; python -m scripts.sampling_experiment.

No anti-alias filter: aliasing is possible, especially at 101 points (dt=0.5ms).
No superiority is assumed. Historical full-grid results are verified and reused.
"""
# Must precede numerical imports: frozen fitting module sets BLAS thread limits.
from scripts import fit_baseline as fit

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from scripts import window_experiment as window
from scripts import generate_dataset as gen
from scripts import noise_experiment as noise
from scripts import train_model as base

COUNTS = [5001, 1001, 501, 101]
SEEDS = window.SEEDS
BATCH = window.BATCH
TRAINING = ('history', 'best_epoch', 'best_validation_mse', 'clean_validation_mse', 'training_seconds')


def selected_indices(count):
    if count not in COUNTS:
        raise ValueError('Require an approved sampling count')
    return np.arange(0, 5001, 5000 // (count - 1))


def selected_bank(x, rms, count, split, level_index, realization=0):
    """Draw original batch128 full banks, THEN select; never redraw shorter shapes."""
    indices = selected_indices(count)
    if x.dtype != np.float32 or x.shape[1:] != (2, 5001):
        raise ValueError('Require original float32 full-grid waveforms')
    rng = noise.noise_rng(split, level_index=level_index, realization=realization)
    full, selected = hashlib.sha256(), hashlib.sha256()
    observed = np.empty((len(x), 2, count), np.float32)
    for start in range(0, len(x), BATCH):
        batch = x[start:start + BATCH]
        perturbation = noise.add_noise(np.zeros_like(batch), noise.LEVELS[level_index], rms, rng)
        full.update(perturbation.tobytes())
        retained = perturbation[:, :, indices]
        selected.update(retained.tobytes())
        observed[start:start + len(batch)] = batch[:, :, indices] + retained
    return observed, dict(full_sha256=full.hexdigest(), selected_sha256=selected.hexdigest())


def evaluate(directory, x, y, damping, ids, rms, count, reference, args, keys, persist):
    indices = selected_indices(count)
    time_s = gen.TIME[indices]
    reports = {k: {'test': []} for k in [*keys, 'fitting']}
    saved = {k: [] for k in reports}
    for condition, (i, r) in enumerate(noise.conditions(3)):
        observed, digests = selected_bank(x, rms, count, 'test', i, r)
        if digests['full_sha256'] != reference[condition]:
            raise ValueError('Original full test noise hash mismatch')
        # Identical physical float32 sum for fitting and every neural model.
        for key in reports:
            if key == 'fitting':
                arrays = window.fit_prefix(observed, rms, time_s, args.max_nfev)
            else:
                model, stats = base.load_checkpoint(directory / f'{key}.pt', args.device)
                inputs = torch.from_numpy(base.transform_inputs(observed, stats))
                arrays = {'prediction': base.decode(base.predict(model, inputs, torch.device(args.device), BATCH), stats)}
                del model, inputs
            report = dict(level=noise.LEVELS[i], realization=r, noise_sha256=digests['full_sha256'],
                          selected_noise_sha256=digests['selected_sha256'],
                          regimes=base.metrics(y, arrays['prediction'], damping))
            if key == 'fitting':
                report.update(fit.summarize(arrays))
            reports[key]['test'].append(report)
            arrays.update(truth=y, sample_id=ids, level=np.full(len(y), noise.LEVELS[i]),
                          realization=np.full(len(y), r), model_key=np.full(len(y), key))
            saved[key].append(arrays)
        persist(reports)
    devices = [torch.device('cpu')]
    if torch.backends.mps.is_available():
        devices.append(torch.device('mps'))
    for key, chunks in saved.items():
        arrays = {k: np.concatenate([a[k] for a in chunks]) for k in chunks[0]}
        arrays.update(condition_noise_sha256=np.array(reference), target_names=np.array(list(gen.RANGES)),
                      selected_indices=indices, time_s=time_s,
                      condition_selected_noise_sha256=np.array([r['selected_noise_sha256'] for r in reports[key]['test']]))
        path = directory / f'{key}_predictions.npz'
        np.savez_compressed(path, **arrays)
        reports[key]['predictions_sha256'] = base.digest(path)
        if key == 'fitting':
            reports[key].update(fit.summarize(arrays))
        else:
            path = directory / f'{key}.pt'
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            model, stats = base.load_checkpoint(path)
            reports[key].update(checkpoint_sha256=base.digest(path),
                training={k: checkpoint[k] for k in TRAINING},
                benchmark=[noise.benchmark(model, stats, x[:, :, indices], d, args.repeats, args.warmup)
                           for d in devices])
        persist(reports)
    return reports


def run(args):
    output = fit.fresh_output(args.output_dir, (args.data_dir, args.noise_dir, args.fitting_dir,
        Path('data'), Path('runs/mlp'), Path('runs/noise'), Path('runs/fitting'), Path('runs/windows')))
    torch.set_num_threads(1)
    if args.device == 'auto':
        args.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    hashes = {n: base.digest(args.data_dir / n) for n in ('circuits.csv', 'waveforms.npz')}
    x, y, splits, damping = base.load_data(args.data_dir)
    if x.dtype != np.float32 or x.shape[1:] != (2, 5001) or len(splits['test']) != 1500:
        raise ValueError('Require original float32 full-grid data and 1500 test circuits')
    rms = noise.channel_rms(x, splits['train'])
    ids = np.array([r['sample_id'] for r in gen.read_metadata(args.data_dir / 'circuits.csv')])[splits['test']]
    reference, models, fitted, ref_hashes = window.references(args, hashes, rms, ids, y[splits['test']],
                                                           base.fit_preprocessing(x, y, splits['train']))
    output.mkdir(parents=True)
    result = dict(settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        dataset_sha256=hashes, reference_sha256=ref_hashes,
        source_sha256={str(Path(m.__file__)): base.digest(Path(m.__file__)) for m in (fit, gen, base, noise, window)},
        implementation_sha256=base.digest(Path(__file__)), thread_environment=fit.THREAD_ENV,
        versions={'python': fit.platform.python_version(), 'platform': fit.platform.platform(),
                  'numpy': np.__version__, 'scipy': fit.scipy.__version__, 'torch': str(torch.__version__)},
        solver={**fit.SOLVER, 'max_nfev': args.max_nfev, 'bounds_physical': fit.BOUNDS.tolist(),
                'fixed_start_physical': np.exp(fit.START).tolist(), 'coordinates': 'log(R,L,C)'},
        sampling_contract='Uniform nested original-grid instantaneous point samples over inclusive fixed 50ms; no anti-alias filtering. Aliasing possible, especially 101 points, dt=0.5ms. No superiority assumed.',
        residual='C-order float64 (physical prediction-observed)/ORIGINAL full train channel RMS; serial fixed-start solves, all failures retained; nfev excludes numerical Jacobian calls.',
        error_interpretation='Convergence is solver status, NOT parameter accuracy. Relative errors reported separately by base.metrics for all circuits, including failures; no accuracy threshold tuned.',
        timing='Neural: noise.benchmark preprocessing/transfer/forward/decode, synchronized CPU and available MPS, single and batch128. Fitting: fit.summarize per-circuit solve latency, excludes bank/disk. Historical timings not remeasured.',
        channel_rms_physical=rms.tolist(), channel_order_units=['current_a', 'capacitor_voltage_v'],
        noise_contract=noise.NOISE_CONTRACT, levels=list(noise.LEVELS), realizations=3, batch_size=BATCH,
        bank_contract='Draw original full (n,2,5001) noise in batch128 including partial final batch, then select indices. Hash concatenated C-order physical float32 perturbation batches, full and selected. Shared physical float32 sum for all methods; validation/test exactly paired across grids.',
        selection='All new checkpoints selected before any new test evaluation; mean standardized log MSE across five fixed validation banks, one realization each; no test-based choices.',
        preprocessing='Clean TRAIN selected points only via base.fit_preprocessing; unchanged TRAIN log targets; original full TRAIN RMS for noise and residuals.',
        augmentation_pairing='Dedicated NumPy RNG per selected shape; same seed initialization/shuffle within clean/augmented pair. Normalized scale full RMS/selected std. No exact train augmentation pairing across counts.',
        archive_layout='Condition-major 13 conditions (zero once, four nonzero levels x three realizations), original CSV test order; 19500 rows/model/grid. prediction/truth (N,3), sample_id/level/realization/model_key (N,), target_names (3,), condition_noise_sha256 (13,). New archives add condition_selected_noise_sha256 (13,), selected_indices and time_s. Fitting retains every diagnostic array including failures. Historical full-grid archives copied unchanged; selected hashes/time recorded here.',
        historical_original='Only matched clean/augmented seed pairs and fitting reused; historical validation selection and timings, not new selection.',
        sampling={})
    def save():
        gen.write_json(output / 'results.json', result)
    keys = list(models)
    # Phase one: select ALL new checkpoints, without passing any test data.
    for count in args.samples:
        indices = selected_indices(count)
        directory = output / f'{count}points'
        directory.mkdir()
        entry = result['sampling'][str(count)] = dict(samples=count, stride=int(indices[1]),
            selected_indices=indices.tolist(), time_s=gen.TIME[indices].tolist(), input_size=2 * count,
            architecture=[2 * count, 128, 64, 3], parameters=sum(p.numel() for p in base.build_model(2 * count).parameters()),
            status='training', models={})
        if count == 5001:
            entry.update(models=window.reuse_50(directory, args, models, fitted), status='reused',
                         provenance='Historical full-count reuse; never retrained, refit or benchmarked.',
                         noise_hashes=[dict(full_sha256=h, selected_sha256=h) for h in reference])
            save()
            continue
        selected_x = x[:, :, indices]
        stats = base.fit_preprocessing(selected_x, y, splits['train'])
        banks, bank_hashes = [], []
        for i in range(len(noise.LEVELS)):
            observed, digests = selected_bank(x[splits['validation']], rms, count, 'validation', i)
            banks.append(torch.from_numpy(base.transform_inputs(observed, stats)))
            bank_hashes.append(digests)
        del observed
        entry['validation_noise_hashes'] = bank_hashes
        for seed in args.seeds:
            for augmented in (False, True):
                key = f'{"augmented" if augmented else "clean"}_{seed}'
                print(f'Training {count}points {key}', flush=True)
                ck = window.train_one(selected_x[splits['train']], y[splits['train']], banks,
                    y[splits['validation']], stats, rms, args, seed, augmented, hashes, gen.TIME[indices])
                torch.save(ck, directory / f'{key}.pt')
                entry['models'][key] = dict(checkpoint_sha256=base.digest(directory / f'{key}.pt'),
                                            training={k: ck[k] for k in TRAINING})
                save()
        entry['status'] = 'selected'
        del banks, selected_x
        save()
    # Phase two: fixed evaluation; never feeds checkpoint selection.
    test = splits['test']
    test_x = x[test]
    del x
    for count in args.samples:
        if count == 5001:
            continue
        entry = result['sampling'][str(count)]
        def persist(reports):
            entry['models'] = {k: {**entry['models'].get(k, {}), **v} for k, v in reports.items()}
            save()
        entry['status'] = 'evaluating'
        evaluate(output / f'{count}points', test_x, y[test], damping[test], ids, rms,
                 count, reference, args, keys, persist)
        entry['status'] = 'complete'
        save()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for flag, default in [('data-dir', 'data'), ('noise-dir', 'runs/noise'),
                          ('fitting-dir', 'runs/fitting'), ('output-dir', 'runs/sampling')]:
        p.add_argument('--' + flag, type=Path, default=Path(default))
    p.add_argument('--samples', type=int, nargs='+', default=COUNTS, choices=COUNTS)
    p.add_argument('--seeds', type=int, nargs='+', default=SEEDS, choices=SEEDS)
    p.add_argument('--device', choices=['auto', 'cpu', 'mps'], default='auto')
    for flag, default in [('epochs', 200), ('patience', 25), ('repeats', 50), ('warmup', 5), ('max-nfev', 200)]:
        p.add_argument('--' + flag, type=int, default=default)
    p.add_argument('--learning-rate', type=float, default=.001)
    return p


def main():
    p = parser()
    args = p.parse_args()
    if (min(args.epochs, args.patience, args.repeats, args.warmup, args.max_nfev) < 1 or
            args.max_nfev != 200 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or
            len(set(args.samples)) != len(args.samples) or len(set(args.seeds)) != len(args.seeds)):
        p.error('Require positive settings, fixed max-nfev=200, and unique samples/seeds')
    run(args)


if __name__ == '__main__':
    main()
