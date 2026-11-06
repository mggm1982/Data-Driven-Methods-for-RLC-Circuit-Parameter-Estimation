"""Paired clean/white-noise MLP experiment; never modifies the baseline.

Physical sigma = level * CLEAN TRAIN channel RMS (all times, voltage DC included).
Adding level * RMS/std to standardized inputs is algebraically identical to
physical noise before (x-mean)/std, up to float32 rounding; no per-waveform scaling.
Validation/test banks are regenerated one batch at a time, never held in memory.
Run: python -m scripts.noise_experiment (use --epochs 1 --seeds 42 for a smoke run).
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
import torch
from torch import nn

from scripts import train_model as base
from scripts import generate_dataset as gen

LEVELS = (0., .01, .02, .05, .10)
VALIDATION_SEED, TEST_SEED = 20261017, 20261018
NOISE_CONTRACT = {
    'definition': 'Independent zero-mean Gaussian per circuit/channel/time; sigma=level*clean-train RMS, including DC.',
    'rng': 'NumPy PCG64(SeedSequence([root, domain, level_index, realization])); float32 standard_normal in C order (circuit,channel,time).',
    'domains': {'train': 0, 'validation': 1, 'test': 2},
    'roots': {'train': 'paired model seed', 'validation': VALIDATION_SEED, 'test': TEST_SEED},
    'order': 'CSV split order, recorded batch size; reset bank stream for every model/validation epoch. Zero level consumes no noise.',
    'augmentation': 'Separate from torch shuffle RNG; choose one level uniformly per circuit each epoch, then fresh float32 normals for entire batch.',
    'hash': 'SHA256 of concatenated C-order float32 physical perturbation batches per condition; batch size and NumPy version are part of contract.',
}


def noise_rng(split, seed=0, level_index=0, realization=0):
    domain = NOISE_CONTRACT['domains'][split]
    root = seed if split == 'train' else VALIDATION_SEED if split == 'validation' else TEST_SEED
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [root, domain, level_index, realization])))


def channel_rms(x, train, batch_size=128):
    squares = np.zeros(2, dtype=np.float64)
    for start in range(0, len(train), batch_size):
        batch = x[train[start:start + batch_size]].astype(np.float64)
        squares += np.square(batch).sum(axis=(0, 2))
    return np.sqrt(squares / (len(train) * x.shape[2])).astype(np.float32)


def add_noise(x, levels, scale, rng):
    """Copy float32 inputs; scale is physical RMS or RMS/std for normalized x."""
    result = np.array(x, dtype=np.float32, copy=True)
    levels = np.asarray(levels, dtype=np.float32)
    if np.any(levels):
        amplitude = levels if levels.ndim == 0 else levels[:, None, None]
        result += (rng.standard_normal(result.shape, dtype=np.float32) *
                   amplitude * np.asarray(scale, dtype=np.float32)[None, :, None])
    return result


def conditions(realizations):
    return [(i, r) for i in range(len(LEVELS))
            for r in range(1 if i == 0 else realizations)]


def bank_predict(model, stats, x, rms, split, level_index, realization, device, batch_size):
    """Identical physical perturbations across models, regardless of preprocessing."""
    rng = noise_rng(split, level_index=level_index, realization=realization)
    predictions, digest = [], hashlib.sha256()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = x[start:start + batch_size]
            perturbation = add_noise(np.zeros_like(batch), LEVELS[level_index], rms, rng)
            digest.update(perturbation.tobytes())
            inputs = base.transform_inputs(batch + perturbation, stats)
            predictions.append(model(torch.from_numpy(inputs).to(device)).cpu().numpy())
    return np.concatenate(predictions), digest.hexdigest()


def validation_scores(model, stats, x, y, rms, device, batch_size):
    target = (np.log(y) - stats['target_mean']) / stats['target_std']
    losses = []
    for i in range(len(LEVELS)):
        z, _ = bank_predict(model, stats, x, rms, 'validation', i, 0, device, batch_size)
        losses.append(float(np.mean((z - target) ** 2)))
    return float(np.mean(losses)), losses[0]


def verify_provenance(checkpoint, hashes, time_s):
    if checkpoint.get('dataset_sha256') != hashes:
        raise ValueError('Checkpoint dataset hashes do not match')
    if not np.array_equal(checkpoint.get('time_s'), time_s):
        raise ValueError('Checkpoint time grid does not match')
    if checkpoint.get('target_names') != list(gen.RANGES):
        raise ValueError('Checkpoint target names/order do not match')


def train_one(x, y, val_x, val_y, stats, rms, args, seed, augmented, hashes, time_s):
    """Only clean train/validation arrays enter this helper; no test access."""
    device = torch.device(args.device)
    torch.manual_seed(seed)
    model = base.build_model(x.shape[1] * x.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    shuffle = torch.Generator().manual_seed(seed)
    rng = noise_rng('train', seed)
    targets = ((np.log(y) - stats['target_mean']) / stats['target_std']).astype(np.float32)
    history, best, best_epoch, state = [], float('inf'), 0, None
    base.synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.
        order = torch.randperm(len(x), generator=shuffle).numpy()
        for offset in range(0, len(x), args.batch_size):
            ids = order[offset:offset + args.batch_size]
            inputs = base.transform_inputs(x[ids], stats).reshape(len(ids), 2, -1)
            if augmented:
                levels = rng.choice(np.array(LEVELS, dtype=np.float32), size=len(ids))
                inputs = add_noise(inputs, levels, rms / stats['channel_std'], rng)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(
                model(torch.from_numpy(inputs.reshape(len(ids), -1)).to(device)),
                torch.from_numpy(targets[ids]).to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(ids)
        score, clean_score = validation_scores(model, stats, val_x, val_y, rms,
                                                device, args.batch_size)
        history.append({'epoch': epoch, 'train_mse': total / len(x),
                        'validation_mse': score, 'clean_validation_mse': clean_score})
        if score < best:
            best, best_epoch = score, epoch
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= args.patience:
            break
    base.synchronize(device)
    return {'state_dict': state, 'input_size': x.shape[1] * x.shape[2],
            'architecture': [x.shape[1] * x.shape[2], 128, 64, 3],
            'preprocessing': {k: torch.from_numpy(v.copy()) for k, v in stats.items()},
            'time_s': torch.from_numpy(time_s.copy()), 'target_names': list(gen.RANGES),
            'dataset_sha256': hashes, 'seed': seed, 'torch_version': str(torch.__version__),
            'best_epoch': best_epoch, 'best_validation_mse': best,
            'clean_validation_mse': history[best_epoch - 1]['clean_validation_mse'],
            'history': history, 'training_seconds': time.perf_counter() - start,
            'noise': {'augmented': augmented, 'levels': list(LEVELS),
                      'channel_rms_physical': rms.tolist(), 'contract': NOISE_CONTRACT,
                      'selection': 'equal mean standardized log-target MSE over fixed validation bank, one realization per level'},
            'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}


def evaluate(model, stats, x, y, damping, ids, rms, device, batch_size, realizations):
    reports, predictions, levels, repeats, hashes = [], [], [], [], []
    for i, r in conditions(realizations):
        z, digest = bank_predict(model, stats, x, rms, 'test', i, r, device, batch_size)
        predicted = base.decode(z, stats)
        reports.append({'level': LEVELS[i], 'realization': r,
                        'regimes': base.metrics(y, predicted, damping), 'noise_sha256': digest})
        predictions.append(predicted)
        levels.extend([LEVELS[i]] * len(x))
        repeats.extend([r] * len(x))
        hashes.append(digest)
    count = len(predictions)
    return reports, {'prediction': np.concatenate(predictions), 'truth': np.tile(y, (count, 1)),
                     'sample_id': np.tile(ids, count), 'level': np.array(levels),
                     'realization': np.array(repeats), 'condition_noise_sha256': np.array(hashes),
                     'target_names': np.array(list(gen.RANGES))}


def benchmark(model, stats, x, device, repeats, warmup):
    """Includes preprocessing, transfer, forward, output transfer and decoding."""
    model.to(device).eval()

    def measure(batch):
        def once():
            with torch.no_grad():
                inputs = torch.from_numpy(base.transform_inputs(batch, stats)).to(device)
                return base.decode(model(inputs).cpu().numpy(), stats)
        for _ in range(warmup):
            once()
        base.synchronize(device)
        durations = []
        for _ in range(repeats):
            base.synchronize(device)
            start = time.perf_counter()
            once()
            base.synchronize(device)
            durations.append(time.perf_counter() - start)
        median, p95 = np.quantile(durations, [.5, .95])
        return {'n': len(batch), 'repeats': repeats, 'median_ms': float(median * 1000),
                'p95_ms': float(p95 * 1000), 'median_circuits_per_second': float(len(batch) / median)}

    return {'device': str(device), 'cpu_threads': torch.get_num_threads(),
            'single': measure(x[:1]), 'batch128': measure(x[:128]),
            'note': 'Preloaded clean inputs; generation/disk excluded; synchronized; warmup excluded.'}


def run(args):
    output = args.output_dir.resolve()
    for protected in (args.data_dir.resolve(), args.original_checkpoint.parent.resolve()):
        if output == protected or protected in output.parents:
            raise ValueError('Output must be separate from dataset and original model directories')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refusing to overwrite nonempty output directory')
    torch.set_num_threads(args.cpu_threads)
    args.device = ('mps' if torch.backends.mps.is_available() else 'cpu') if args.device == 'auto' else args.device
    device = torch.device(args.device)
    hashes = {name: base.digest(args.data_dir / name) for name in ('circuits.csv', 'waveforms.npz')}
    original = torch.load(args.original_checkpoint, map_location='cpu', weights_only=True)
    verify_provenance(original, hashes, gen.TIME)
    x, y, splits, damping = base.load_data(args.data_dir)
    stats = base.fit_preprocessing(x, y, splits['train'])
    rms = channel_rms(x, splits['train'])
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'dataset_sha256': hashes, 'original_checkpoint_sha256': base.digest(args.original_checkpoint),
                'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                             'scipy': scipy.__version__, 'torch': str(torch.__version__),
                             'platform': platform.platform()},
                'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'channel_order_units': ['current_a', 'capacitor_voltage_v'],
                'channel_rms_physical': rms.tolist(), 'levels': list(LEVELS),
                'noise_contract': NOISE_CONTRACT, 'accuracy_device': str(device),
                'cpu_threads': torch.get_num_threads(), 'validation_realizations': 1,
                'selection': 'Both paired controls selected on equal-average fixed-bank MSE across all levels.',
                'caveat': 'Realizations are repeated measurements of the same circuits, not independent circuits. MPS bitwise training reproducibility is not guaranteed.'}
    gen.write_json(output / 'manifest.json', manifest)
    paths = {'original': args.original_checkpoint}
    for seed in args.seeds:
        for augmented in (False, True):
            key = f'{"augmented" if augmented else "clean"}_{seed}'
            print(f'Training {key}', flush=True)
            checkpoint = train_one(x[splits['train']], y[splits['train']],
                                   x[splits['validation']], y[splits['validation']],
                                   stats, rms, args, seed, augmented, hashes, gen.TIME)
            paths[key] = output / f'{key}.pt'
            torch.save(checkpoint, paths[key])
    # The test split is first used only after every checkpoint has been selected.
    test = splits['test']
    test_x = x[test]
    ids = np.array([row['sample_id'] for row in gen.read_metadata(args.data_dir / 'circuits.csv')])[test]
    result, reference_hashes = {}, None
    devices = [torch.device('cpu')]
    if torch.backends.mps.is_available():
        devices.append(torch.device('mps'))
    for key, path in paths.items():
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        verify_provenance(checkpoint, hashes, gen.TIME)
        model, model_stats = base.load_checkpoint(path, device)
        reports, arrays = evaluate(model, model_stats, test_x, y[test], damping[test], ids,
                                   rms, device, args.batch_size, args.realizations)
        noise_hashes = arrays['condition_noise_sha256'].tolist()
        if reference_hashes is not None and reference_hashes != noise_hashes:
            raise ValueError('Test noise pairing failed')
        reference_hashes = noise_hashes
        arrays['model_key'] = np.full(len(arrays['level']), key)
        np.savez_compressed(output / f'{key}_predictions.npz', **arrays)
        result[key] = {'checkpoint_sha256': base.digest(path), 'test': reports,
                       'training': {name: checkpoint[name] for name in
                                    ('best_epoch', 'best_validation_mse', 'clean_validation_mse',
                                     'training_seconds', 'history') if name in checkpoint},
                       'predictions_sha256': base.digest(output / f'{key}_predictions.npz'),
                       'benchmark': [benchmark(model, model_stats, test_x, d, args.repeats, args.warmup)
                                     for d in devices]}
        gen.write_json(output / 'results.json', result)
    print(f'Experiment saved to {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--original-checkpoint', type=Path, default=Path('runs/mlp/model.pt'))
    parser.add_argument('--output-dir', type=Path, default=Path('runs/noise'))
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps'], default='auto')
    parser.add_argument('--seeds', type=int, nargs='+', default=[20261014, 20261015, 20261016])
    for flag, default in [('epochs', 200), ('patience', 25), ('batch-size', 128),
                          ('realizations', 3), ('repeats', 50), ('warmup', 5), ('cpu-threads', 1)]:
        parser.add_argument('--' + flag, type=int, default=default)
    parser.add_argument('--learning-rate', type=float, default=.001)
    args = parser.parse_args()
    if (min(args.epochs, args.patience, args.batch_size, args.realizations,
            args.repeats, args.warmup, args.cpu_threads) < 1 or
            not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or
            len(set(args.seeds)) != len(args.seeds) or
            any(s < 0 or s in (VALIDATION_SEED, TEST_SEED) for s in args.seeds)):
        parser.error('Require positive settings, unique nonnegative training seeds distinct from bank roots')
    run(args)


if __name__ == '__main__':
    main()
