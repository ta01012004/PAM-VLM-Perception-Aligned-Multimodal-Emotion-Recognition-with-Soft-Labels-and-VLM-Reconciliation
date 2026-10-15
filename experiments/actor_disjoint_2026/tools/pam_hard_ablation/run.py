"""Run original-PAM M10 hard controls inside a GPU SLURM allocation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from importlib.metadata import PackageNotFoundError, version

import pandas as pd
import yaml


def check_packages(expected_versions: dict) -> None:
    for package, expected in expected_versions.items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            actual = 'not-installed'
        if actual != expected:
            raise RuntimeError(f'Environment mismatch: {package}; do not reuse soft results.')


def summarize(suite: Path, specification: dict, report_run, metrics, validate) -> dict:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows, pools = [], {'hard': [], 'soft': []}
    for run in specification['runs']:
        marker = suite / 'state' / (run['id'] + '.done.json')
        if not marker.exists():
            continue
        hard_dir = Path(json.loads(marker.read_text())['run_dir'])
        soft_dir = Path(run['soft_run_dir'])
        hard, hard_pred = report_run(run, hard_dir, suite / 'reports' / run['id'] / 'hard')
        soft, soft_pred = report_run(run, soft_dir, suite / 'reports' / run['id'] / 'soft')
        row = {'fold': run['fold']}
        for key in ('macro_f1', 'uar', 'accuracy', 'ece', 'nll', 'brier'):
            row.update({f'hard_{key}': hard[key], f'soft_{key}': soft[key],
                        f'soft_minus_hard_{key}': soft[key] - hard[key]})
        rows.append(row)
        pools['hard'].append(hard_pred)
        pools['soft'].append(soft_pred)
    summary = {'completed': len(rows), 'expected': 10, 'complete': len(rows) == 10,
               'protocol': 'M10', 'training_seed': 42, 'paired_folds': rows}
    if rows:
        frame = pd.DataFrame(rows).sort_values('fold')
        frame.to_csv(suite / 'reports/paired_folds.csv', index=False)
        summary['fold_mean_sd'] = {
            key: {'mean': float(frame[key].mean()),
                  'sd': float(frame[key].std(ddof=1)) if len(frame) > 1 else None}
            for key in frame if key != 'fold'
        }
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for arm, color in (('hard', '#8A607D'), ('soft', '#117C88')):
            axes[0].plot(frame.fold, frame[arm + '_macro_f1'], 'o-', label=arm, color=color)
            axes[1].plot(frame.fold, frame[arm + '_ece'], 'o-', label=arm, color=color)
        for axis, label in zip(axes, ('Macro-F1', 'ECE (lower is better)')):
            axis.set(xlabel='Actor fold', ylabel=label, xticks=frame.fold)
            axis.legend()
            axis.grid(alpha=.2)
        fig.suptitle(f'M10 original PAM: {len(rows)}/10 paired folds; seed 42')
        fig.tight_layout()
        for extension in ('png', 'pdf'):
            fig.savefig(suite / 'reports' / ('M10_hard_soft.' + extension), dpi=200)
        plt.close(fig)
        if summary['complete']:
            manifests = pd.concat([pd.read_csv(r['test_manifest']) for r in specification['runs']])
            summary['pooled_oof'] = {}
            for arm in ('hard', 'soft'):
                pooled = pd.concat(pools[arm], ignore_index=True)
                validate(pooled, manifests)
                summary['pooled_oof'][arm] = metrics(pooled)
                pooled.to_csv(suite / 'reports' / (arm + '_M10_oof.csv'), index=False)
    (suite / 'reports/summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main(suite: Path) -> None:
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU training is allowed only inside SLURM.')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No allocated CUDA device.')
    code = suite / 'code'
    sys.path.insert(0, str(code / 'scripts/published16'))
    import run_suite as helpers
    from aggregate import report_run, metrics, validate_predictions
    from protocols import write_json

    specification = json.loads((suite / 'suite.json').read_text())
    if len(specification['runs']) != 10:
        raise ValueError('Expected ten M10 hard controls.')
    with helpers.suite_lock(suite):
        try:
            helpers.verify_inputs(suite)
            for run in specification['runs']:
                record = json.loads((Path(run['soft_run_dir']) / 'reproducibility.json').read_text())
                check_packages(record['package_versions'])
                soft_config = yaml.safe_load(Path(run['soft_config']).read_text())
                helpers.verify_checkpoint(Path(run['soft_run_dir']), soft_config)
            write_json(suite / 'state/hardware.json', {
                'job_id': os.environ['SLURM_JOB_ID'], 'hostname': os.uname().nodename,
                'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
                'cuda': torch.version.cuda, 'started_at': helpers.now()})
            helpers.smoke(suite, specification['runs'][0])
            for index, run in enumerate(specification['runs'], 1):
                helpers.verify_inputs(suite)
                soft_config = yaml.safe_load(Path(run['soft_config']).read_text())
                helpers.verify_checkpoint(Path(run['soft_run_dir']), soft_config)
                config = yaml.safe_load(Path(run['config']).read_text())
                marker = suite / 'state' / (run['id'] + '.done.json')
                directory = helpers.run_directory(Path(run['output_root']))
                if marker.exists():
                    if directory != Path(json.loads(marker.read_text())['run_dir']):
                        raise RuntimeError('Completed marker disagrees with hard output directory.')
                else:
                    if directory is not None and not helpers.trained(directory):
                        raise RuntimeError('Partial run exists; inspect before restart, no auto resume.')
                    if directory is None:
                        write_json(suite / 'state/current.json', {
                            'phase': 'training', 'run': run['id'], 'index': index,
                            'expected': 10, 'job_id': os.environ['SLURM_JOB_ID']})
                        helpers.execute(['scripts/train.py', '--config', run['config']],
                                        suite / 'logs' / (run['id'] + '_train.log'))
                        directory = helpers.run_directory(Path(run['output_root']))
                    if directory is None or not helpers.trained(directory):
                        raise RuntimeError('Training did not produce complete artifacts.')
                    helpers.verify_checkpoint(directory, config)
                    if not all((directory / name).is_file() for name in
                               ('test_predictions.csv', 'test_metrics.json')):
                        helpers.execute(['scripts/eval_test.py', '--run-dir', str(directory),
                                         '--batch-size', '4', '--num-workers', '12'],
                                        suite / 'logs' / (run['id'] + '_test.log'))
                    report_run(run, directory, suite / 'reports' / run['id'] / 'hard')
                    helpers.verify_inputs(suite)
                    write_json(marker, {'run_dir': str(directory), 'finished_at': helpers.now()})
                helpers.verify_checkpoint(directory, config)
                summarize(suite, specification, report_run, metrics, validate_predictions)
            write_json(suite / 'state/current.json', {'phase': 'complete', 'completed': 10,
                       'finished_at': helpers.now(), 'job_id': os.environ['SLURM_JOB_ID']})
        except Exception as error:
            write_json(suite / 'state/failure.json', {'error': str(error),
                       'traceback': traceback.format_exc(), 'at': helpers.now()})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    main(parser.parse_args().suite.resolve())
