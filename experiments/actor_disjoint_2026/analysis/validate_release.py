"""Recompute the complete public PAM evidence on CPU without training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import METRICS, calculate_metrics, validate_actor_partitions

ROOT = Path(__file__).resolve().parents[1]


def safe_file(root: Path, name: str) -> Path:
    path = root / name
    if Path(name).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Artifact path escapes the evidence package')
    return path


def load(root: Path, name: str) -> dict:
    return json.loads(safe_file(root, name).read_text(encoding='utf-8'))


def validate_catalog(frame: pd.DataFrame, root: Path) -> None:
    expected = {
        'primary': {f'h5_fold{i:02}' for i in range(1, 6)}
                   | {f'm10_fold{i:02}' for i in range(1, 11)} | {'t602020_seed0'},
        'ce_only': {f'm10_fold{i:02}' for i in range(1, 11)},
        'reviewer': {f'{arm}_seed{seed}' for arm in
                     [*(f'A{i}' for i in range(10)), 'H0'] for seed in (42, 43, 44)},
    }
    if len(frame) != 59 or frame[['group', 'run']].duplicated().any():
        raise ValueError('Missing or duplicate runs')
    if set(frame.group) != set(expected):
        raise ValueError('Unexpected evidence group')
    for group, ids in expected.items():
        if set(frame.loc[frame.group == group, 'run']) != ids:
            raise ValueError(f'Unexpected run IDs: {group}')
    for column in ('prediction_path', 'config_path', 'test_metrics_path'):
        for name in frame[column]:
            if not safe_file(root, name).is_file():
                raise ValueError(f'Missing artifact: {name}')


def verify_hashes(root: Path) -> None:
    manifest = load(root, 'checksums.json')
    for name, expected in manifest.items():
        actual = hashlib.sha256(safe_file(root, name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Artifact checksum changed: {name}')


def validate_reviewer_aggregates(rows: pd.DataFrame, reports: Path) -> float:
    """Check every published reviewer mean/SD and paired contrast from raw rows."""
    summary = json.loads((reports / 'summary.json').read_text(encoding='utf-8'))
    if (not summary['complete'] or summary['completed'] != 33
            or summary['expected'] != 33 or summary['seeds'] != [42, 43, 44]):
        raise ValueError('Reviewer summary is incomplete')

    maximum_error = 0.0

    def compare(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
        nonlocal maximum_error
        if actual.index.duplicated().any():
            raise ValueError('Duplicate reviewer aggregation identities')
        pd.testing.assert_frame_equal(actual.sort_index().astype(float),
                                      expected.sort_index().astype(float),
                                      check_names=False, atol=1e-7, rtol=0)
        error = np.abs(actual.sort_index().to_numpy(dtype=float)
                       - expected.sort_index().to_numpy(dtype=float)).max()
        maximum_error = max(maximum_error, float(error))

    saved = pd.read_csv(reports / 'all_runs.csv').set_index('run')
    compare(saved[list(METRICS)], rows.set_index('run')[list(METRICS)])
    saved = pd.read_csv(reports / 'seed_mean_sd.csv', header=[0, 1], index_col=0)
    expected = rows.groupby('arm')[list(METRICS)].agg(['mean', 'std'])
    compare(saved, expected)
    paired = rows.set_index(['arm', 'seed'])[list(METRICS)].copy()
    baseline = rows[rows.arm == 'A0'].set_index('seed')
    for arm, seed in paired.index:
        paired.loc[(arm, seed)] -= baseline.loc[seed, list(METRICS)].astype(float)
    paired.columns = [m + '_minus_A0' for m in METRICS]
    saved = pd.read_csv(reports / 'paired_seed_deltas.csv').set_index(['arm', 'seed'])
    compare(saved, paired)
    additional = []
    for seed in (42, 43, 44):
        by_arm = rows[rows.seed == seed].set_index('arm')
        for name, terms in (
                ('H0_minus_A1_smoothing_only', {'H0': 1, 'A1': -1}),
                ('A9_minus_A7_minus_A8_plus_A0_interaction', {'A9': 1, 'A7': -1, 'A8': -1, 'A0': 1})):
            additional.append({'contrast': name, 'seed': seed,
                               **{m: sum(coefficient * by_arm.loc[arm, m]
                                         for arm, coefficient in terms.items()) for m in METRICS}})
    saved = pd.read_csv(reports / 'additional_paired_contrasts.csv').set_index(['contrast', 'seed'])
    compare(saved, pd.DataFrame(additional).set_index(['contrast', 'seed']))
    return maximum_error


def validate_release(root: Path = ROOT) -> dict:
    verify_hashes(root)
    catalog = pd.read_csv(root / 'runs.csv')
    validate_catalog(catalog, root)
    rows, frames, configs = [], {}, {}
    max_error = 0.0

    def check(actual: float, expected: float, description: str) -> None:
        nonlocal max_error
        error = abs(float(actual) - float(expected))
        max_error = max(max_error, error)
        if not np.isfinite(error) or error > 1e-7:
            raise ValueError(f'Metric mismatch: {description}')

    for row in catalog.itertuples(index=False):
        frame = pd.read_csv(root / row.prediction_path)
        config = load(root, row.config_path)
        key = row.group, row.run
        frames[key], configs[key] = frame, config
        protocol = config['published_protocol']
        actors = protocol['actors']
        validate_actor_partitions(actors)
        if len(set().union(*map(set, actors.values()))) != 91:
            raise ValueError(f'Actor assignments do not exhaust CREMA-D: {key}')
        if row.seed != config['seed'] or row.protocol != protocol['protocol']:
            raise ValueError(f'Catalog/config disagreement: {key}')
        identity = frame[['clip_id', 'actor_id', 'true']]
        if frame.clip_id.duplicated().any() or set(frame.actor_id) != set(actors['test']):
            raise ValueError(f'Duplicate clips or incorrect actors: {key}')
        if not np.array_equal(frame.clip_id.str.split('_').str[0].astype(int), frame.actor_id):
            raise ValueError(f'Clip/actor identity mismatch: {key}')
        counts = protocol['counts']['test']
        if len(frame) != counts['clips'] or frame['true'].value_counts().to_dict() != counts['classes']:
            raise ValueError(f'Incorrect test support: {key}')
        result = calculate_metrics(frame)
        saved = load(root, row.test_metrics_path)
        if saved['split'] != 'test' or saved['num_samples'] != len(frame):
            raise ValueError(f'Validation scores substituted for test: {key}')
        if saved['confusion_matrix'] != result['confusion_matrix']:
            raise ValueError(f'Confusion matrix mismatch: {key}')
        for metric in METRICS:
            check(result[metric], getattr(row, metric), f'{key}/{metric}')
        for metric in ('macro_f1', 'uar', 'accuracy'):
            check(result[metric], saved[metric], f'{key}/saved/{metric}')
        check(result['ece'], saved['ece']['ece'], f'{key}/saved/ece')
        rows.append({'group': row.group, 'run': row.run, 'arm': row.arm,
                     'protocol': row.protocol, 'seed': row.seed,
                     **{metric: result[metric] for metric in METRICS}})
        if row.group == 'primary':
            for split in ('train', 'val', 'test'):
                manifest = pd.read_csv(root / f'evidence/primary/splits/{row.run}/manifest_{split}.csv')
                if set(manifest.actor_id) != set(actors[split]):
                    raise ValueError(f'Manifest/config actors mismatch: {key}/{split}')
                if len(manifest) != protocol['counts'][split]['clips'] or manifest.clip_id.duplicated().any():
                    raise ValueError(f'Manifest support mismatch: {key}/{split}')
                if split == 'test' and set(manifest.clip_id) != set(identity.clip_id):
                    raise ValueError(f'Manifest/prediction clips mismatch: {key}')
        if row.group == 'reviewer':
            if config['reviewer_variant']['arm'] != row.arm or row.seed not in (42, 43, 44):
                raise ValueError('Reviewer arm/seed mismatch')
            reference = frames.get(('reviewer', 'A0_seed42'), frame)
            reference_config = configs.get(('reviewer', 'A0_seed42'), config)
            if actors != reference_config['published_protocol']['actors']:
                raise ValueError('Reviewer training/validation/test actor assignment differs')
            if not identity.equals(reference[['clip_id', 'actor_id', 'true']]):
                raise ValueError('Reviewer test identity/order differs')

    metrics = pd.DataFrame(rows)
    max_error = max(max_error, validate_reviewer_aggregates(
        metrics[metrics.group == 'reviewer'], root / 'evidence/reviewer/reports'))
    primary_summary = load(root, 'evidence/primary/reports/summary.json')['protocols']
    pooled_counts = {}
    for protocol, summary in primary_summary.items():
        block = metrics[(metrics.group == 'primary') & (metrics.protocol == protocol)]
        for metric in METRICS:
            check(block[metric].mean(), summary['metrics'][metric]['mean'], f'{protocol}/mean/{metric}')
            if len(block) > 1:
                check(block[metric].std(ddof=1), summary['metrics'][metric]['sd'], f'{protocol}/sd/{metric}')
        if protocol in ('H5', 'M10'):
            pooled = pd.concat([frames[('primary', run)] for run in block.run], ignore_index=True)
            if pooled.clip_id.duplicated().any() or pooled.actor_id.nunique() != 91:
                raise ValueError('Incomplete/overlapping outer test folds')
            result = calculate_metrics(pooled)
            pooled_counts[protocol] = len(pooled)
            for metric in METRICS:
                check(result[metric], summary['pooled_oof'][metric], f'{protocol}/OOF/{metric}')

    paired = load(root, 'evidence/ce_only/reports/summary.json')
    for item in paired['paired_folds']:
        run = f'm10_fold{item["fold"]:02}'
        a, b = frames[('ce_only', run)], frames[('primary', run)]
        if not a[['clip_id', 'actor_id', 'true']].equals(b[['clip_id', 'actor_id', 'true']]):
            raise ValueError('Paired folds differ in identities/order')
        ca, cb = configs[('ce_only', run)], configs[('primary', run)]
        if ca['published_protocol']['actors'] != cb['published_protocol']['actors']:
            raise ValueError('Paired actor partitions differ')
        if ca['loss']['label_smoothing'] != .05 or ca['loss']['soft_labels_enabled']:
            raise ValueError('CE-only control is not the smoothing-preserving control')
        for group, prefix in (('ce_only', 'hard'), ('primary', 'soft')):
            values = metrics[(metrics.group == group) & (metrics.run == run)].iloc[0]
            for metric in METRICS:
                check(values[metric], item[f'{prefix}_{metric}'], f'{run}/{prefix}/{metric}')
    for group, prefix in (('ce_only', 'hard'), ('primary', 'soft')):
        block = metrics[(metrics.group == group) & (metrics.protocol == 'M10')]
        for metric in METRICS:
            for statistic in ('mean', 'sd'):
                value = block[metric].mean() if statistic == 'mean' else block[metric].std(ddof=1)
                check(value, paired['fold_mean_sd'][f'{prefix}_{metric}'][statistic],
                      f'M10/{prefix}/{statistic}/{metric}')
        pooled = pd.read_csv(root / f'evidence/ce_only/reports/{prefix}_M10_oof.csv')
        result = calculate_metrics(pooled)
        combined = pd.concat([frames[(group, run)] for run in block.run], ignore_index=True)
        columns = ['clip_id', 'actor_id', 'true', 'pred', *('p_' + x for x in
                    ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad'))]
        pd.testing.assert_frame_equal(pooled[columns].sort_values('clip_id').reset_index(drop=True),
                                      combined[columns].sort_values('clip_id').reset_index(drop=True),
                                      atol=1e-12, rtol=1e-12)
        for metric in METRICS:
            check(result[metric], paired['pooled_oof'][prefix][metric], f'{prefix}/OOF/{metric}')
    return {'status': 'PASS', 'run_counts': catalog.group.value_counts().to_dict(),
            'max_metric_error': max_error, 'primary_pooled_counts': pooled_counts,
            'reviewer_unique_test_clips': frames[('reviewer', 'A0_seed42')].clip_id.nunique(),
            'scope': 'Exported prediction/configuration verification; not checkpoint reload or training reproduction.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(validate_release(args.root.resolve()), indent=2))
