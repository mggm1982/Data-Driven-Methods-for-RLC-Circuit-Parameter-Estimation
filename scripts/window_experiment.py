"""Inclusive observation-prefix experiment; run in a fresh process with -m scripts.window_experiment.

Only input size changes in the fixed 128/64 MLP. Full-record training RMS is
used for every noise amplitude and fitting residual. Augmentation has a dedicated
NumPy stream: paired clean/augmented initialization and shuffle match, but draws
across windows are NOT exactly paired (prefix sizes consume different streams).
All new checkpoints are selected before any new test evaluation; no test tuning.
"""
# This import MUST precede NumPy/Torch: it sets the single-thread BLAS environment.
from scripts import fit_baseline as fit

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from scripts import generate_dataset as gen
from scripts import noise_experiment as noise
from scripts import train_model as base

WINDOWS = [1, 2, 5, 10, 20, 50]
SEEDS = [20261014, 20261015, 20261016]
BATCH = 128


def window_count(ms, time_s=gen.TIME):
    count = int(round(ms * .001 / (time_s[1] - time_s[0]))) + 1
    if count > len(time_s) or not np.isclose(time_s[count - 1], ms * .001):
        raise ValueError('Window endpoint is not on original time grid')
    return count


def prefix_bank(x, rms, count, split, level_index, realization=0):
    """Draw in ORIGINAL batch/channel/time layout, then crop; include partial batch.

    Only prefix physical sums are retained. Hashes cover perturbations, not sums.
    The float32 physical sum here is shared by fitting and neural evaluation.
    """
    rng = noise.noise_rng(split, level_index=level_index, realization=realization)
    full, prefix = hashlib.sha256(), hashlib.sha256()
    observed = np.empty((len(x), 2, count), dtype=np.float32)
    for start in range(0, len(x), BATCH):
        batch = x[start:start + BATCH]
        perturbation = noise.add_noise(np.zeros_like(batch), noise.LEVELS[level_index], rms, rng)
        full.update(perturbation.tobytes())
        cropped = perturbation[:, :, :count]
        prefix.update(cropped.tobytes())
        observed[start:start + len(batch)] = batch[:, :, :count] + cropped
    return observed, {'full_sha256': full.hexdigest(), 'prefix_sha256': prefix.hexdigest()}


def train_one(x, y, banks, val_y, stats, rms, args, seed, augmented, hashes, time_s):
    device = torch.device(args.device)
    torch.manual_seed(seed)
    model = base.build_model(x.shape[1] * x.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    shuffle = torch.Generator().manual_seed(seed)
    rng = noise.noise_rng('train', seed)
    targets = ((np.log(y) - stats['target_mean']) / stats['target_std']).astype(np.float32)
    val_target = (np.log(val_y) - stats['target_mean']) / stats['target_std']
    history, best, best_epoch, state = [], float('inf'), 0, None
    base.synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.
        order = torch.randperm(len(x), generator=shuffle).numpy()
        for offset in range(0, len(x), BATCH):
            ids = order[offset:offset + BATCH]
            inputs = base.transform_inputs(x[ids], stats).reshape(len(ids), 2, -1)
            if augmented:
                levels = rng.choice(np.array(noise.LEVELS, dtype=np.float32), size=len(ids))
                inputs = noise.add_noise(inputs, levels, rms / stats['channel_std'], rng)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(
                model(torch.from_numpy(inputs.reshape(len(ids), -1)).to(device)),
                torch.from_numpy(targets[ids]).to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(ids)
        losses = [float(np.mean((base.predict(model, bank, device, BATCH) - val_target) ** 2))
                  for bank in banks]
        score = float(np.mean(losses))
        history.append(dict(epoch=epoch, train_mse=total / len(x), validation_mse=score,
                            clean_validation_mse=losses[0]))
        if score < best:
            best, best_epoch = score, epoch
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= args.patience:
            break
    base.synchronize(device)
    return dict(state_dict=state, input_size=2 * len(time_s),
                architecture=[2 * len(time_s), 128, 64, 3],
                preprocessing={k: torch.from_numpy(v.copy()) for k, v in stats.items()},
                time_s=torch.from_numpy(time_s.copy()), target_names=list(gen.RANGES),
                dataset_sha256=hashes, seed=seed, augmented=augmented,
                noise={'levels': list(noise.LEVELS), 'channel_rms_physical': rms.tolist(),
                       'contract': noise.NOISE_CONTRACT},
                settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                best_epoch=best_epoch, best_validation_mse=best, history=history,
                clean_validation_mse=history[best_epoch - 1]['clean_validation_mse'],
                training_seconds=time.perf_counter() - start)


def read_json(path):
    return json.loads(path.read_text())


def verify_archive(path, record, ids, y, key, reference):
    if base.digest(path) != record['predictions_sha256']:
        raise ValueError(f'Prediction hash mismatch: {path}')
    with np.load(path, allow_pickle=False) as a:
        count = len(reference)
        expected = noise.conditions(3)
        checks = [np.array_equal(a['sample_id'], np.tile(ids, count)),
                  np.array_equal(a['truth'], np.tile(y, (count, 1))),
                  np.array_equal(a['level'], np.repeat([noise.LEVELS[i] for i, _ in expected], len(ids))),
                  np.array_equal(a['realization'], np.repeat([r for _, r in expected], len(ids))),
                  np.array_equal(a['condition_noise_sha256'], reference),
                  np.array_equal(a['target_names'], list(gen.RANGES)),
                  np.all(a['model_key'] == key), a['prediction'].shape == (len(ids) * count, 3)]
        if not all(checks):
            raise ValueError(f'Prediction paired layout mismatch: {path}')
        if key == 'fitting' and not all(k in a for k in
                ('success', 'status', 'message', 'nfev', 'njev', 'cost', 'optimality',
                 'bound_hits', 'residual_calls', 'elapsed_seconds')):
            raise ValueError('Missing fitting diagnostics')


def references(args, hashes, rms, ids, y, stats):
    """Verify completed historical runs, including recorded sources and checkpoints."""
    nm, nr = args.noise_dir / 'manifest.json', args.noise_dir / 'results.json'
    fm, fr = args.fitting_dir / 'manifest.json', args.fitting_dir / 'results.json'
    manifest, models, fitting, fitted = map(read_json, (nm, nr, fm, fr))
    reference = fit.validate_reference(manifest, models, hashes, rms)
    if (fitting['dataset_sha256'] != hashes or fitting['noise_manifest_sha256'] != base.digest(nm)
            or fitting['noise_results_sha256'] != base.digest(nr)
            or fitting['solver'] != {**fit.SOLVER, 'max_nfev': 200,
                'bounds_physical': fit.BOUNDS.tolist(), 'fixed_start_physical': np.exp(fit.START).tolist(),
                'coordinates': 'log(R,L,C)'}
            or fitting['channel_rms_physical'] != manifest['channel_rms_physical']
            or fitting['noise_contract'] != noise.NOISE_CONTRACT
            or fitting['versions']['numpy'] != np.__version__
            or fitted['completed_conditions'] != 13
            or [(r['level'], r['realization']) for r in fitted['test']] !=
               [(noise.LEVELS[i], r) for i, r in noise.conditions(3)]
            or [r['noise_sha256'] for r in fitted['test']] != reference):
        raise ValueError('Fitting provenance mismatch or incomplete run')
    if {Path(p).name for p in fitting['source_sha256']} != {
            'fit_baseline.py', 'generate_dataset.py', 'train_model.py', 'noise_experiment.py'}:
        raise ValueError('Incomplete frozen source provenance')
    for source, digest in fitting['source_sha256'].items():
        if base.digest(Path(__file__).parent / Path(source).name) != digest:
            raise ValueError(f'Frozen source hash mismatch: {source}')
    if manifest['settings']['seeds'] != SEEDS:
        raise ValueError('Historical paired seeds mismatch')
    selected = {}
    for seed in args.seeds:
        for augmented in (False, True):
            key = f'{"augmented" if augmented else "clean"}_{seed}'
            record = models[key]
            path = args.noise_dir / f'{key}.pt'
            if base.digest(path) != record['checkpoint_sha256']:
                raise ValueError('Checkpoint hash mismatch')
            ck = torch.load(path, map_location='cpu', weights_only=True)
            noise.verify_provenance(ck, hashes, gen.TIME)
            if (ck['seed'] != seed or ck['noise']['augmented'] != augmented or
                    ck['architecture'] != [10002, 128, 64, 3] or
                    ck['noise']['contract'] != noise.NOISE_CONTRACT or
                    ck['noise']['channel_rms_physical'] != rms.tolist() or
                    any(ck[k] != v for k, v in record['training'].items()) or
                    any(not np.array_equal(ck['preprocessing'][k].numpy(), v) for k, v in stats.items())):
                raise ValueError('Historical paired model/preprocessing mismatch')
            verify_archive(args.noise_dir / f'{key}_predictions.npz', record, ids, y, key, reference)
            selected[key] = record
    verify_archive(args.fitting_dir / 'predictions.npz', fitted, ids, y, 'fitting', reference)
    return reference, selected, fitted, {str(p): base.digest(p) for p in (nm, nr, fm, fr)}


def reuse_50(directory, args, models, fitted):
    """Copy verified artifacts only: never train, fit, predict or benchmark 50 ms."""
    reports = {}
    for key, record in {**models, 'fitting': fitted}.items():
        source = args.fitting_dir / 'predictions.npz' if key == 'fitting' else args.noise_dir / f'{key}_predictions.npz'
        shutil.copyfile(source, directory / f'{key}_predictions.npz')
        if key != 'fitting':
            shutil.copyfile(args.noise_dir / f'{key}.pt', directory / f'{key}.pt')
        reports[key] = {**record, 'reused': True, 'source_predictions': str(source),
                        'timing_provenance': 'Historical timings, not remeasured; historical validation selection.'}
    return reports


def fit_prefix(observed, rms, time_s, max_nfev):
    rows = [fit.fit_waveform(waveform, rms, time_s=time_s, max_nfev=max_nfev)
            for waveform in observed]
    return {'prediction': np.array([p for p, _ in rows]),
            **{k: np.array([info[k] for _, info in rows]) for k in rows[0][1]}}


def evaluate_window(directory, x, y, damping, ids, rms, count, reference, args, keys, persist):
    reports = {k: {'test': []} for k in [*keys, 'fitting']}
    saved = {k: [] for k in reports}
    for condition, (i, r) in enumerate(noise.conditions(3)):
        observed, digests = prefix_bank(x, rms, count, 'test', i, r)
        if digests['full_sha256'] != reference[condition]:
            raise ValueError('Original full test noise hash mismatch')
        # One physical float32 sum, consumed by all methods, including fitting.
        for key in reports:
            if key == 'fitting':
                arrays = fit_prefix(observed, rms, gen.TIME[:count], args.max_nfev)
            else:
                model, stats = base.load_checkpoint(directory / f'{key}.pt', args.device)
                inputs = torch.from_numpy(base.transform_inputs(observed, stats))
                arrays = {'prediction': base.decode(base.predict(model, inputs, torch.device(args.device), BATCH), stats)}
                del model, inputs
            report = dict(level=noise.LEVELS[i], realization=r, noise_sha256=digests['full_sha256'],
                          prefix_noise_sha256=digests['prefix_sha256'],
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
                      condition_prefix_noise_sha256=np.array([r['prefix_noise_sha256'] for r in reports[key]['test']]))
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
                training={k: checkpoint[k] for k in ('best_epoch', 'best_validation_mse',
                    'clean_validation_mse', 'training_seconds', 'history')},
                benchmark=[noise.benchmark(model, stats, x[:, :, :count], d, args.repeats, args.warmup)
                           for d in devices])
        persist(reports)
    return reports


def run(args):
    output = fit.fresh_output(args.output_dir, (args.data_dir, args.noise_dir, args.fitting_dir,
                                               Path('data'), Path('runs/mlp'), Path('runs/noise'), Path('runs/fitting')))
    torch.set_num_threads(1)
    if args.device == 'auto':
        args.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    hashes = {n: base.digest(args.data_dir / n) for n in ('circuits.csv', 'waveforms.npz')}
    x, y, splits, damping = base.load_data(args.data_dir)
    if x.dtype != np.float32 or x.shape[1:] != (2, 5001) or len(splits['test']) != 1500:
        raise ValueError('Require original float32 full-grid data and 1500 test circuits')
    rms = noise.channel_rms(x, splits['train'])
    ids = np.array([r['sample_id'] for r in gen.read_metadata(args.data_dir / 'circuits.csv')])[splits['test']]
    reference, models, fitted, ref_hashes = references(args, hashes, rms, ids, y[splits['test']],
                                                     base.fit_preprocessing(x, y, splits['train']))
    output.mkdir(parents=True)
    result = dict(settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        dataset_sha256=hashes, reference_sha256=ref_hashes,
        source_sha256={str(Path(m.__file__)): base.digest(Path(m.__file__)) for m in (fit, gen, base, noise)},
        implementation_sha256=base.digest(Path(__file__)), thread_environment=fit.THREAD_ENV,
        versions={'python': fit.platform.python_version(), 'platform': fit.platform.platform(),
                  'numpy': np.__version__, 'scipy': fit.scipy.__version__, 'torch': str(torch.__version__)},
        solver={**fit.SOLVER, 'max_nfev': args.max_nfev, 'bounds_physical': fit.BOUNDS.tolist(),
                'fixed_start_physical': np.exp(fit.START).tolist(), 'coordinates': 'log(R,L,C)'},
        residual='C-order (physical prediction-observed)/ORIGINAL full train channel RMS, float64; serial fixed-start solves, all failures retained; nfev excludes numerical Jacobian calls.',
        timing='Neural: noise.benchmark preprocessing/transfer/forward/decode, synchronized CPU and available MPS, single and batch128; fitting: fit.summarize per-circuit solve latency, excludes bank/disk. Historical 50ms timings not remeasured.',
        channel_rms_physical=rms.tolist(), channel_order_units=['current_a', 'capacitor_voltage_v'],
        noise_contract=noise.NOISE_CONTRACT, levels=list(noise.LEVELS), batch_size=BATCH,
        selection='All new checkpoints selected before test evaluation; mean standardized log MSE, five fixed original validation-bank prefixes, one realization each; no test-based choices.',
        preprocessing='Clean prefix train only; unchanged train log targets; noise and fitting residual scales use ORIGINAL full train RMS.',
        augmentation_pairing='Same seed init/shuffle within window. Dedicated NumPy augmentation RNG, normalized scale RMS/prefix std. Prefix size changes RNG consumption; no exact augmentation pairing across windows.',
        archive_layout='Condition-major (zero once, four nonzero levels x three repeats), original CSV test order within condition; 19500 rows/model/window. prediction/truth (N,3), sample_id/level/realization/model_key (N,), target_names (3,), condition_noise_sha256 (13,); new prefixes add condition_prefix_noise_sha256 (13,). Fitting retains all solver diagnostic arrays, including failures.',
        historical_original='Excluded from window comparison; only matched clean/augmented seed pairs and fitting reused.', windows={})
    def save():
        gen.write_json(output / 'results.json', result)
    keys = list(models)
    # Phase one: ALL checkpoint selection. Test waveforms are never passed to training.
    for ms in args.windows_ms:
        count = window_count(ms)
        directory = output / f'{ms}ms'
        directory.mkdir()
        entry = result['windows'][str(ms)] = dict(samples=count, input_size=2 * count,
            architecture=[2 * count, 128, 64, 3], parameters=sum(p.numel() for p in base.build_model(2 * count).parameters()),
            status='training', models={})
        if ms == 50:
            entry.update(models=reuse_50(directory, args, models, fitted), status='reused',
                         noise_hashes=[dict(full_sha256=h, prefix_sha256=h) for h in reference])
            save()
            continue
        stats = base.fit_preprocessing(x[:, :, :count], y, splits['train'])
        banks, bank_hashes = [], []
        for i in range(len(noise.LEVELS)):
            observed, digests = prefix_bank(x[splits['validation']], rms, count, 'validation', i)
            banks.append(torch.from_numpy(base.transform_inputs(observed, stats)))
            bank_hashes.append(digests)
        del observed
        entry['validation_noise_hashes'] = bank_hashes
        for seed in args.seeds:
            for augmented in (False, True):
                key = f'{"augmented" if augmented else "clean"}_{seed}'
                print(f'Training {ms}ms {key}', flush=True)
                ck = train_one(x[splits['train'], :, :count], y[splits['train']], banks,
                    y[splits['validation']], stats, rms, args, seed, augmented, hashes, gen.TIME[:count])
                torch.save(ck, directory / f'{key}.pt')
                entry['models'][key] = {'checkpoint_sha256': base.digest(directory / f'{key}.pt'),
                                       'training': {k: ck[k] for k in ('history', 'best_epoch', 'best_validation_mse', 'training_seconds')}}
                save()
        entry['status'] = 'selected'
        del banks
        save()
    # Phase two: fixed protocol evaluation, never used for any checkpoint choice.
    test = splits['test']
    test_x = x[test]
    del x
    for ms in args.windows_ms:
        if ms == 50:
            continue
        entry = result['windows'][str(ms)]
        def persist(reports):
            entry['models'] = {k: {**entry['models'].get(k, {}), **v} for k, v in reports.items()}
            save()
        entry['status'] = 'evaluating'
        evaluate_window(output / f'{ms}ms', test_x, y[test], damping[test], ids, rms,
                        window_count(ms), reference, args, keys, persist)
        entry['status'] = 'complete'
        save()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for flag, default in [('data-dir', 'data'), ('noise-dir', 'runs/noise'),
                          ('fitting-dir', 'runs/fitting'), ('output-dir', 'runs/windows')]:
        p.add_argument('--' + flag, type=Path, default=Path(default))
    p.add_argument('--windows-ms', type=int, nargs='+', default=WINDOWS, choices=WINDOWS)
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
            len(set(args.windows_ms)) != len(args.windows_ms) or len(set(args.seeds)) != len(args.seeds)):
        p.error('Require positive settings, fixed max-nfev=200, and unique windows/seeds')
    run(args)


if __name__ == '__main__':
    main()
