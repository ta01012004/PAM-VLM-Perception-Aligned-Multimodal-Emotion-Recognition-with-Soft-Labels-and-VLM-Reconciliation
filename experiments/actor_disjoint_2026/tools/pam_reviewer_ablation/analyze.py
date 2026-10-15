"""Recompute all test metrics, pair by seed; never invent missing results."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd


def analyze(suite):
    sys.path.insert(0, str(suite / 'code/scripts/published16'))
    from aggregate import report_run
    from run_suite import verify_inputs, verify_checkpoint
    import yaml
    verify_inputs(suite)
    spec = json.loads((suite / 'suite.json').read_text())
    rows = []
    initializations = {}
    for run in spec['runs']:
        marker = suite / 'state' / (run['id'] + '.done.json')
        if not marker.exists():
            raise RuntimeError(f'Missing prespecified run: {run["id"]}')
        directory = Path(json.loads(marker.read_text())['run_dir'])
        verify_checkpoint(directory, yaml.safe_load(Path(run['config']).read_text()))
        row, _ = report_run(run, directory, suite / 'reports' / run['id'])
        resource = json.loads((directory / 'resource_usage.json').read_text())
        row.update(arm=run['arm'], seed=run['seed'], training_seconds=resource['training_seconds'],
                   peak_cuda_bytes=resource['peak_cuda_bytes'], **resource['parameter_counts'])
        rows.append(row)
        init = json.loads((directory / 'initialization.json').read_text())
        initializations[(run['seed'], run['arm'])] = init['copied_tensor_hashes']
    for (seed, arm), hashes in initializations.items():
        base = initializations[(seed, 'A0')]
        if any(base.get(key) != value for key, value in hashes.items()):
            raise ValueError(f'Shared initial tensors differ: {seed}/{arm}')
    frame = pd.DataFrame(rows)
    frame.to_csv(suite / 'reports/all_runs.csv', index=False)
    metric_names = ['macro_f1', 'uar', 'accuracy', 'ece', 'nll', 'brier']
    pairs = []
    for seed in spec['seeds']:
        base = frame[(frame.arm == 'A0') & (frame.seed == seed)].iloc[0]
        for _, row in frame[frame.seed == seed].iterrows():
            pairs.append({'arm': row.arm, 'seed': seed,
                          **{key + '_minus_A0': float(row[key] - base[key]) for key in metric_names}})
    pd.DataFrame(pairs).to_csv(suite / 'reports/paired_seed_deltas.csv', index=False)
    contrasts = []
    for seed in spec['seeds']:
        at_seed = frame[frame.seed == seed].set_index('arm')
        if 'H0' in at_seed.index and 'A1' in at_seed.index:
            contrasts.append({'contrast': 'H0_minus_A1_smoothing_only', 'seed': seed,
                              **{key: float(at_seed.loc['H0', key] - at_seed.loc['A1', key])
                                 for key in metric_names}})
        if all(a in at_seed.index for a in ('A0', 'A7', 'A8', 'A9')):
            contrasts.append({'contrast': 'A9_minus_A7_minus_A8_plus_A0_interaction', 'seed': seed,
                              **{key: float(at_seed.loc['A9', key] - at_seed.loc['A7', key]
                                            - at_seed.loc['A8', key] + at_seed.loc['A0', key])
                                 for key in metric_names}})
    if contrasts:
        pd.DataFrame(contrasts).to_csv(suite / 'reports/additional_paired_contrasts.csv', index=False)
    stats = frame.groupby('arm')[metric_names].agg(['mean', 'std'])
    stats.to_csv(suite / 'reports/seed_mean_sd.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, metric, label in zip(axes, ['macro_f1', 'ece'], ['Macro-F1', 'ECE-15 (lower is better)']):
        means = frame.groupby('arm')[metric].mean().reindex(spec['arms'])
        sd = frame.groupby('arm')[metric].std().reindex(spec['arms'])
        axis.bar(means.index, means, color='#466C88', yerr=sd.fillna(0), capsize=3)
        axis.set(ylabel=label, xlabel='Prespecified arm')
        axis.grid(axis='y', alpha=.2)
    fig.suptitle('Fresh T602020 actor holdout: seed mean ± sample SD; not fold uncertainty')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(suite / 'reports' / ('ablation.' + ext), dpi=200)
    plt.close(fig)
    result = {'complete': True, 'completed': len(rows), 'expected': spec['expected'],
              'seeds': spec['seeds'], 'limitations': spec['limitations'],
              'shared_initialization_verified': True, 'historical_results_reused': False}
    (suite / 'reports/summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.suite.resolve()), indent=2))
