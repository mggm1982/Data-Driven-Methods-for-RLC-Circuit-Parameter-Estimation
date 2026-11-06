"""Read-only descriptive diagnostics; python -m scripts.transient_diagnostics."""
import csv
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.stats import rankdata

NAMES = ['resistance_ohm', 'inductance_h', 'capacitance_f']
SEEDS = (20261014, 20261015, 20261016)
CONDITIONS = [(0., 0)] + [(level, r) for level in (.01, .02, .05, .1) for r in range(3)]
TAU_EDGES = np.array([.2, .5, 1, 2, 5, 10])
FD_EDGES = np.array([0, .5, 1, 2, 5])
OUT = Path('results/transient_diagnostics')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def physics(truth):
    r, l, c = np.asarray(truth).T
    radicand = 1 / (l * c) - (r / (2 * l)) ** 2
    fd = np.full(len(r), np.nan)
    under = radicand > 0
    fd[under] = np.sqrt(radicand[under]) / (2 * np.pi)
    return 2 * l / r, fd, radicand


def bin_indices(values, edges, eligible):
    """-1 means excluded; fail rather than silently lose eligible circuits."""
    values, edges = np.asarray(values), np.asarray(edges)
    if np.any(eligible & (~np.isfinite(values) | (values < edges[0]) | (values > edges[-1]))):
        raise ValueError('Uncovered eligible truth')
    result = np.full(len(values), -1, dtype=int)
    result[eligible] = np.minimum(np.searchsorted(edges, values[eligible], side='right') - 1,
                                  len(edges) - 2)
    return result


def metrics(truth, prediction, mask):
    n = int(mask.sum())
    if not n:
        return {'count': 0, **{k: None for k in
                ('R_median_pct', 'L_median_pct', 'C_median_pct', 'max_median_pct',
                 'any_gt10_count', 'any_gt10_pct', 'any_gt50_count', 'any_gt50_pct')}}
    errors = 100 * np.abs(prediction[mask] - truth[mask]) / truth[mask]
    maximum = errors.max(axis=1)
    return {'count': n, **dict(zip(('R_median_pct', 'L_median_pct', 'C_median_pct'),
                                 map(float, np.median(errors, axis=0)))),
            'max_median_pct': float(np.median(maximum)),
            **{f'any_gt{t}_{suffix}': value for t in (10, 50)
               for suffix, value in [('count', int((maximum > t).sum())),
                                     ('pct', float(100 * (maximum > t).mean()))]}}


def spearman(x, y):
    a, b = rankdata(x), rankdata(y)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def aggregate(seed_conditions, seeded=True):
    """Equal realization weight within seed, then equal seed weight; no pooling."""
    keys = seed_conditions[0][0].keys()
    per_seed = [{k: (None if rows[0][k] is None else
                    float(np.mean([r[k] for r in rows]))) for k in keys}
                for rows in seed_conditions]
    return {'per_seed': per_seed,
            'mean': {k: None if per_seed[0][k] is None else
                     float(np.mean([s[k] for s in per_seed])) for k in keys},
            'seed_sd': {k: (float(np.std([s[k] for s in per_seed], ddof=1))
                            if seeded and len(per_seed) > 1 and per_seed[0][k] is not None
                            else None) for k in keys}}


def verify_rows(a, ids, truth, conditions):
    n, count = len(ids), len(conditions)
    checks = [np.array_equal(a['sample_id'], np.tile(ids, count)),
              np.array_equal(a['truth'], np.tile(truth, (count, 1))),
              np.array_equal(a['level'], np.repeat([v for v, _ in conditions], n)),
              np.array_equal(a['realization'], np.repeat([r for _, r in conditions], n)),
              a['prediction'].shape == (n * count, 3), np.isfinite(a['prediction']).all()]
    if not all(checks):
        raise ValueError('Prediction ID/truth/condition/finite pairing mismatch')


def load_archive(path, record, ids, truth, key, full_hashes, selected_hashes=None, window=None):
    if digest(path) != record['predictions_sha256']:
        raise ValueError(f'Prediction SHA256 mismatch: {path}')
    with np.load(path, allow_pickle=False) as saved:
        a = {k: saved[k] for k in saved.files}
    verify_rows(a, ids, truth, CONDITIONS)
    if (not np.array_equal(a['target_names'], NAMES) or not np.all(a['model_key'] == key)
            or [(r['level'], r['realization']) for r in record['test']] != CONDITIONS
            or not np.array_equal(a['condition_noise_sha256'], full_hashes)
            or [r['noise_sha256'] for r in record['test']] != full_hashes):
        raise ValueError(f'Archive model/noise contract mismatch: {path}')
    if selected_hashes is not None:
        if (not np.array_equal(a['selected_indices'], np.arange(0, 5001, 50))
                or not np.array_equal(a['time_s'], (np.arange(5001) * 1e-5)[::50])
                or not np.array_equal(a['condition_selected_noise_sha256'], selected_hashes)
                or [r['selected_noise_sha256'] for r in record['test']] != selected_hashes):
            raise ValueError('Sparse selected-noise/grid mismatch')
    if window is not None and window < 50:
        if not np.array_equal(a['condition_prefix_noise_sha256'],
                              [r['prefix_noise_sha256'] for r in record['test']]):
            raise ValueError('Window prefix-noise mismatch')
    return a


def condition_summary(truth, prediction, tau, fd, radicand, bins):
    under = radicand > 0
    maximum = (np.abs(prediction - truth) / truth).max(axis=1)
    return {'groups': {name: metrics(truth, prediction, mask) for name, mask in
                       [('all', np.ones(len(truth), dtype=bool)), ('underdamped', under),
                        ('critical', radicand == 0), ('overdamped', radicand < 0)]},
            'bins': {axis: [metrics(truth, prediction, indices == i) for i in range(size)]
                     for axis, (indices, size) in bins.items()},
            'spearman_max_error': {'tau': spearman(tau[under], maximum[under]),
                                   'fd': spearman(fd[under], maximum[under])}}


def summarize_series(series):
    summaries = {}
    seeds = list(series)
    for level in (0., .1):
        rows = [[r for r in series[s] if r['level'] == level] for s in seeds]
        def agg(extract):
            return aggregate([[extract(r) for r in seed] for seed in rows], seeds != ['none'])
        summaries[str(level)] = {
            'groups': {g: agg(lambda r: r['groups'][g]) for g in rows[0][0]['groups']},
            'bins': {axis: [agg(lambda r: r['bins'][axis][i]) for i in range(len(values))]
                     for axis, values in rows[0][0]['bins'].items()},
            'spearman_max_error': agg(lambda r: r['spearman_max_error'])}
    return summaries


def panel(title, labels, curves, ylabel):
    """Self-contained categorical SVG; dashes/labels distinguish curves without a palette."""
    width, height = 760, 330
    left, top, bottom = 65, 35, 250
    max_y = max((v for _, values in curves for v in values if v is not None), default=1) or 1
    max_y *= 1.1
    xs = np.linspace(left, width - 40, len(labels))
    svg = [f'<section><h3>{title}</h3><p>{ylabel}</p>',
           f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}">',
           f'<path class="axis" d="M{left},{top}V{bottom}H{width - 30}"/>']
    for v in np.linspace(0, max_y, 5):
        y = bottom - (bottom - top) * v / max_y
        svg.append(f'<text x="5" y="{y:.1f}">{v:.1f}</text>')
    for x, label in zip(xs, labels):
        svg.append(f'<text text-anchor="middle" x="{x:.1f}" y="280">{label}</text>')
    for j, (name, values) in enumerate(curves):
        points = ' '.join(f'{x:.1f},{bottom - (bottom-top)*v/max_y:.1f}'
                          for x, v in zip(xs, values) if v is not None)
        dash = ('none', '8 3', '2 3', '10 3 2 3', '5 5', '1 5')[j % 6]
        svg.append(f'<polyline class="curve" points="{points}" stroke-dasharray="{dash}"><title>{name}</title></polyline>')
    svg.append('</svg><ul>')
    for j, (name, _) in enumerate(curves):
        dash = ('none', '8 3', '2 3', '10 3 2 3', '5 5', '1 5')[j % 6]
        svg.append(f'<li><svg viewBox="0 0 40 10" class="key"><path class="curve" stroke-dasharray="{dash}" d="M0,5H40"/></svg> {name}</li>')
    return '\n'.join(svg) + '</ul></section>'


def render(results):
    edges = results['edges']
    labels = {axis: [f'{float(a):g}–{float(b):g}' for a, b in zip(e, e[1:])] for axis, e in edges.items()}
    models = results['models']
    html = ['<!doctype html><meta charset="utf-8"><title>Transient diagnostics</title>',
            '<style>body{color:var(--fg,inherit);margin:2em}section{border:1px solid var(--border,currentColor);padding:1em;margin-block:1em}svg{max-width:100%}text{fill:var(--muted,currentColor)}.axis{stroke:var(--border,currentColor);fill:none}.curve{stroke:var(--accent,currentColor);fill:none;stroke-width:2}.key{width:3em;height:1em}li,p{color:var(--muted,inherit)}</style>',
            '<h2>Descriptive transient diagnostics</h2><p>Underdamped only; fixed truth bins. Means of condition statistics, not pooled errors. No causal or optimal-window claim. See report.md and summary.json for counts, seed SD and every condition.</p>']
    for axis in ('tau_ms', 'fd_khz'):
        curves = [(f'{method}, {level*100:g}% noise',
                   [b['mean']['any_gt10_pct'] for b in models[f'sparse101/{method}']['summary'][str(level)]['bins'][axis]])
                  for method in ('single', 'five') for level in (0., .1)]
        html.append(panel(f'Sparse101 vs {axis}', labels[axis], curves, 'Any component >10% error (%); bins in ' + axis))
    for level in (0., .1):
        curves = [(f'{w} ms {method}', [b['mean']['max_median_pct'] for b in
                   models[f'window{w}/{method}']['summary'][str(level)]['bins']['tau_ms']])
                  for w in (1, 5, 50) for method in ('clean', 'augmented')]
        html.append(panel(f'Dense windows, {level*100:g}% noise vs tau', labels['tau_ms'], curves,
                          'Median circuit maximum-component error (%); tau bins (ms)'))
    (OUT / 'plots.html').write_text('\n'.join(html) + '\n')
    report = ['# Descriptive decay/frequency diagnostics', '',
              'Read-only original prediction archives; no new experiments or model selection.',
              f"Accounting: {results['accounting']}",
              f'Fixed truth edges: {edges}. Left closed; final upper edge inclusive.',
              'All errors/rates below are percentages. R/L/C and max are medians within each condition/bin.',
              'Average realizations within seed, then seeds; ± is sample seed SD, not CI. Fitter SD is absent.',
              'Counts in summary.json are exact per condition; aggregate counts are means. Noise realizations repeat circuits.',
              'All/underdamped/critical/overdamped errors and per-condition descriptive Spearman tau/fd associations (no p-values) are in summary.json.',
              'plots.html uses only the pre-stated curves. Bins were not derived from errors.', '',
              'No isolated decay/aliasing mechanism: physical covariates correlate, network dimensions and training effects change across windows. No prospective optimum.', '']
    report += ['## Measured associations and boundaries', '',
               'The tables below quantify sparse frequency/decay associations and their change after five starts, plus dense window-dependent associations. Empty bins are not evidence of zero error.',
               'Associations do not establish a causal late-noise mechanism or prospective optimum. Tau and frequency are derived from correlated component labels, not independently intervened variables.',
               'Clean fitter medians rounded to 0.000% are not exact zeros; rank correlations at numerical-scale errors should not be interpreted as meaningful error magnitudes.', '',
               '| Model | Noise | Spearman tau mean ± seed SD | Spearman fd mean ± seed SD |',
               '|---|---|---|---|']
    for model, record in models.items():
        for level, summary in record['summary'].items():
            a = summary['spearman_max_error']
            cells = []
            for axis in ('tau', 'fd'):
                value, sd = a['mean'][axis], a['seed_sd'][axis]
                cells.append('NA' if value is None else f'{value:.3f}' +
                             ('' if sd is None else f' ± {sd:.3f}'))
            report.append(f'| {model} | {float(level)*100:g}% | ' + ' | '.join(cells) + ' |')
    report.append('')
    for model, record in models.items():
        report += [f'## {model}', '', '| Noise | Axis/bin | n | R/L/C median | Max median ± seed SD | >10% / >50% rate |',
                   '|---|---|---:|---|---|---|']
        for level, summary in record['summary'].items():
            for axis, values in summary['bins'].items():
                for label, b in zip(labels[axis], values):
                    m, sd = b['mean'], b['seed_sd']['max_median_pct']
                    def fmt(v):
                        return 'NA' if v is None else f'{v:.3f}'
                    report.append(f"| {float(level)*100:g}% | {axis} {label} | {m['count']:g} | " +
                                  '/'.join(fmt(m[k]) for k in ('R_median_pct', 'L_median_pct', 'C_median_pct')) +
                                  f" | {fmt(m['max_median_pct'])}" + (f' ± {sd:.3f}' if sd is not None else '') +
                                  f" | {fmt(m['any_gt10_pct'])} / {fmt(m['any_gt50_pct'])} |")
        report.append('')
    (OUT / 'report.md').write_text('\n'.join(report) + '\n')


def run():
    inputs = {}
    def read_json(path):
        inputs[path] = digest(path)
        return json.loads(Path(path).read_text())
    windows = read_json('results/windows/results.json')
    sampling = read_json('results/sampling/results.json')
    multi = read_json('results/sparse_multistart/results.json')
    manifest = read_json('results/sparse_multistart/manifest.json')
    hashes = {name: digest(Path('data') / name) for name in ('circuits.csv', 'waveforms.npz')}
    if any(r['dataset_sha256'] != hashes for r in (windows, sampling, manifest)):
        raise ValueError('Dataset SHA256 mismatch')
    inputs.update({f'data/{k}': v for k, v in hashes.items()})
    rows = list(csv.DictReader(Path('data/circuits.csv').open()))
    ids = np.array([int(r['sample_id']) for r in rows if r['split'] == 'test'])
    truth = np.array([[float(r[k]) for k in NAMES] for r in rows if r['split'] == 'test'])
    if len(ids) != 1500 or len(set(ids)) != 1500:
        raise ValueError('Require original 1500 unique test circuits')
    with np.load('data/waveforms.npz', allow_pickle=False) as a:
        if (not np.array_equal(a['sample_id'], [int(r['sample_id']) for r in rows])
                or not np.array_equal(a['time_s'], np.arange(5001) * 1e-5)):
            raise ValueError('Dataset ID/grid mismatch')
    tau, fd, radicand = physics(truth)
    under = radicand > 0
    fd_edges = FD_EDGES.copy()
    if np.max(fd[under]) / 1000 > fd_edges[-1]:
        fd_edges[-1] = np.inf
    bins = {'tau_ms': (bin_indices(tau * 1000, TAU_EDGES, under), len(TAU_EDGES)-1),
            'fd_khz': (bin_indices(fd / 1000, fd_edges, under), len(fd_edges)-1)}
    reference = sampling['sampling']['101']['models']['fitting']
    full_hashes = [r['noise_sha256'] for r in reference['test']]
    selected_hashes = [r['selected_noise_sha256'] for r in reference['test']]
    results = {'edges': {'tau_ms': TAU_EDGES.tolist(), 'fd_khz': fd_edges.tolist()},
               'accounting': {'test': len(ids), 'underdamped': int(under.sum()),
                              'critical': int((radicand == 0).sum()), 'overdamped': int((radicand < 0).sum()),
                              'bin_counts': {axis: np.bincount(idx[idx >= 0], minlength=n).tolist()
                                             for axis, (idx, n) in bins.items()}}, 'models': {}}
    for axis, (idx, _) in bins.items():
        if (idx >= 0).sum() != under.sum():
            raise ValueError(f'Bin coverage mismatch: {axis}')
    def add(name, seed, a, conditions):
        series = results['models'].setdefault(name, {'seeds': {}})['seeds'].setdefault(str(seed), [])
        for j, (level, realization) in enumerate(conditions):
            if level not in (0., .1):
                continue
            prediction = a['prediction'][j*len(ids):(j+1)*len(ids)]
            series.append({'level': level, 'realization': realization,
                           **condition_summary(truth, prediction, tau, fd, radicand, bins)})
    for w in (1, 2, 5, 10, 20, 50):
        for method in ('clean', 'augmented'):
            for seed in SEEDS:
                key = f'{method}_{seed}'
                matches = list(Path(f'runs/windows/{w}ms').glob(f'{key}_predictions.npz'))
                if len(matches) != 1:
                    raise ValueError(f'Missing/ambiguous saved window archive: {w}/{key}')
                path = matches[0]
                record = windows['windows'][str(w)]['models'][key]
                a = load_archive(path, record, ids, truth, key, full_hashes, window=w)
                inputs[str(path)] = record['predictions_sha256']
                add(f'window{w}/{method}', seed, a, CONDITIONS)
    for key in ('fitting', *(f'augmented_{s}' for s in SEEDS)):
        path = Path(f'runs/sampling/101points/{key}_predictions.npz')
        record = sampling['sampling']['101']['models'][key]
        a = load_archive(path, record, ids, truth, key, full_hashes, selected_hashes)
        inputs[str(path)] = record['predictions_sha256']
        add('sparse101/single' if key == 'fitting' else 'sparse101/augmented',
            'none' if key == 'fitting' else key.split('_')[1], a, CONDITIONS)
    expected = [(0., 0), (.1, 0), (.1, 1), (.1, 2)]
    if [(r['level'], r['realization']) for r in multi['conditions']] != expected:
        raise ValueError('Five-start conditions mismatch')
    for row in multi['conditions']:
        path = Path(row['arrays_path'])
        if digest(path) != row['arrays_sha256']:
            raise ValueError('Five-start prediction SHA256 mismatch')
        inputs[str(path)] = row['arrays_sha256']
        condition = (row['level'], row['realization'])
        index = CONDITIONS.index(condition)
        if row['full_sha256'] != full_hashes[index] or row['selected_sha256'] != selected_hashes[index]:
            raise ValueError('Five-start noise pairing mismatch')
        with np.load(path, allow_pickle=False) as a:
            verify_rows(a, ids, truth, [condition])
            if (not np.array_equal(a['selected_indices'], np.arange(0, 5001, 50))
                    or not np.array_equal(a['time_s'], np.arange(5001)[::50] * 1e-5)):
                raise ValueError('Five-start grid mismatch')
            add('sparse101/five', 'none', a, [condition])
    for record in results['models'].values():
        record['summary'] = summarize_series(record['seeds'])
    OUT.mkdir(parents=True, exist_ok=True)
    def save(name, value):
        (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    # JSON has no infinity literal; retain an explicit label if the fixed last edge expands.
    if np.isinf(fd_edges[-1]):
        results['edges']['fd_khz'][-1] = 'inf'
    save('summary.json', results)
    render(results)
    save('manifest.json', {'protocol': 'docs/transient_diagnostics.md',
         'input_sha256': inputs, 'implementation_sha256': {p: digest(p) for p in
             ('scripts/transient_diagnostics.py', 'tests/test_transient_diagnostics.py', 'docs/transient_diagnostics.md')},
         'output_sha256': {str(OUT / p): digest(OUT / p) for p in ('summary.json', 'report.md', 'plots.html')},
         'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
         'verification': 'Exact dataset/archive hashes, CSV split IDs/truth, finite predictions, conditions, model keys, full and available selected/prefix noise hashes and grids verified.',
         'aggregation': 'Per-condition statistics, realizations within seed then seeds; sample seed SD. Percent errors/rates. No fitting seed SD. No pooling independent circuits.',
         'scope': 'Existing archives only; no training, fitting, model selection or causal inference.'})
    print(json.dumps(results['accounting']))
    print(f'Verified {len(inputs)} inputs; wrote {OUT}')


if __name__ == '__main__':
    run()
