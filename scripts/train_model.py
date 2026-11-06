"""Train a two-waveform MLP; preprocessing is fitted on training circuits only."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from scripts import generate_dataset as gen


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_data(directory):
    rows = gen.read_metadata(directory / 'circuits.csv')
    with np.load(directory / 'waveforms.npz', allow_pickle=False) as saved:
        if not np.array_equal(saved['sample_id'], [r['sample_id'] for r in rows]):
            raise ValueError('CSV and waveform IDs must match in order')
        if not np.array_equal(saved['time_s'], gen.TIME):
            raise ValueError('Unexpected time grid')
        x = np.stack([saved['current_a'], saved['capacitor_voltage_v']], axis=1)
    if x.shape != (len(rows), 2, len(gen.TIME)) or not np.isfinite(x).all():
        raise ValueError('Invalid waveform shape or values')
    y = np.array([[r[k] for k in gen.RANGES] for r in rows], dtype=np.float64)
    splits = {s: np.flatnonzero([r['split'] == s for r in rows])
              for s in ('train', 'validation', 'test')}
    return x, y, splits, np.array([r['damping_ratio'] for r in rows])


def fit_preprocessing(x, y, train):
    # One mean/std per channel, shared across all times and circuits.
    channels = x[train].astype(np.float64)
    log_y = np.log(y[train])
    return {'channel_mean': channels.mean(axis=(0, 2)).astype(np.float32),
            'channel_std': channels.std(axis=(0, 2)).astype(np.float32),
            'target_mean': log_y.mean(axis=0).astype(np.float32),
            'target_std': log_y.std(axis=0).astype(np.float32)}


def transform_inputs(x, stats):
    return ((x - stats['channel_mean'][None, :, None]) /
            stats['channel_std'][None, :, None]).reshape(len(x), -1)


def decode(z, stats):
    return np.exp(z * stats['target_std'] + stats['target_mean'])


def build_model(input_size):
    return nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(),
                         nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))


def predict(model, x, device, batch_size):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(b.to(device)).cpu()
                          for b in x.split(batch_size)]).numpy()


def metrics(truth, predicted, damping):
    relative = np.abs(predicted - truth) / truth
    groups = {'all': np.ones(len(truth), dtype=bool), 'underdamped': damping < 1,
              'critical': damping == 1, 'overdamped': damping > 1,
              'near_critical': np.abs(damping - 1) < .01}
    report = {}
    for name, mask in groups.items():
        report[name] = {'count': int(mask.sum())}
        if mask.any():
            report[name]['relative_error'] = {
                key: {'mean': float(relative[mask, j].mean()),
                      'median': float(np.median(relative[mask, j])),
                      'p95': float(np.quantile(relative[mask, j], .95)),
                      'max': float(relative[mask, j].max())}
                for j, key in enumerate(gen.RANGES)}
    return report


def synchronize(device):
    if device.type == 'mps':
        torch.mps.synchronize()


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('mps' if args.device == 'auto' and
                          torch.backends.mps.is_available() else
                          'cpu' if args.device == 'auto' else args.device)
    x, y, splits, damping = load_data(args.data_dir)
    stats = fit_preprocessing(x, y, splits['train'])
    x = torch.from_numpy(transform_inputs(x, stats))
    targets = torch.from_numpy(((np.log(y) - stats['target_mean']) /
                                stats['target_std']).astype(np.float32))
    model = build_model(x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    train_ids = torch.from_numpy(splits['train'])
    val_ids = splits['validation']
    best_loss, best_epoch, best_state = float('inf'), 0, None
    history = []
    synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.
        order = train_ids[torch.randperm(len(train_ids), generator=generator)]
        for ids in order.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(x[ids].to(device)), targets[ids].to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(ids)
        val = predict(model, x[val_ids], device, args.batch_size)
        val_loss = float(np.mean((val - targets[val_ids].numpy()) ** 2))
        history.append({'epoch': epoch, 'train_mse': total / len(train_ids),
                        'validation_mse': val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f'Epoch {epoch}: train={history[-1]["train_mse"]:.6g}, val={val_loss:.6g}', flush=True)
        if epoch - best_epoch >= args.patience:
            break
    synchronize(device)
    training_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {'state_dict': best_state, 'input_size': x.shape[1],
                  'architecture': [x.shape[1], 128, 64, 3],
                  'preprocessing': {k: torch.from_numpy(v) for k, v in stats.items()},
                  'time_s': torch.from_numpy(gen.TIME.copy()),
                  'target_names': list(gen.RANGES), 'best_epoch': best_epoch,
                  'seed': args.seed, 'torch_version': str(torch.__version__),
                  'dataset_sha256': {name: digest(args.data_dir / name)
                                     for name in ('circuits.csv', 'waveforms.npz')}}
    torch.save(checkpoint, args.output_dir / 'model.pt')
    # Evaluate the serialized best checkpoint, not the final epoch's weights.
    restored, restored_stats = load_checkpoint(args.output_dir / 'model.pt', device)
    test_ids = splits['test']
    synchronize(device)
    start = time.perf_counter()
    z = predict(restored, x[test_ids], device, args.batch_size)
    synchronize(device)
    inference_seconds = time.perf_counter() - start
    predicted = decode(z, restored_stats)
    report = {'device': str(device), 'torch_version': str(torch.__version__),
              'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'best_epoch': best_epoch, 'best_validation_mse': best_loss,
              'training_seconds': training_seconds, 'history': history,
              'test_inference_seconds': inference_seconds,
              'test_inference_note': 'Includes CPU-to-device inputs and device-to-CPU predictions; excludes loading and preprocessing.',
              'test': metrics(y[test_ids], predicted, damping[test_ids])}
    (args.output_dir / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('history', 'settings')}, indent=2))


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = build_model(checkpoint['input_size'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()
    stats = {k: v.numpy() for k, v in checkpoint['preprocessing'].items()}
    return model, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=Path('runs/mlp'))
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps'], default='auto')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=20261014)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1 or args.learning_rate <= 0:
        parser.error('Epochs, patience, batch size and learning rate must be positive')
    train(args)


if __name__ == '__main__':
    main()
