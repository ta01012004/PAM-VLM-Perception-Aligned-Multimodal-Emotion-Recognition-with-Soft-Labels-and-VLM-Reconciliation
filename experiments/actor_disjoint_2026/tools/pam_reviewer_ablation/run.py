"""SLURM-only smoke or one array task, fail closed on incomplete artifacts."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys
import traceback

import yaml

PACKAGES = ('torch', 'torchvision', 'transformers', 'numpy', 'pandas', 'pyarrow',
            'scikit-learn', 'scipy', 'soundfile', 'PyYAML', 'opencv-python')


def package_versions():
    values = {}
    for package in PACKAGES:
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            values[package] = 'not-installed'
    return values


@contextmanager
def lock(path):
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump({'job': os.environ.get('SLURM_JOB_ID'), 'pid': os.getpid()}, f)
        yield
    finally:
        path.unlink()


def main(suite, smoke, index):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Training/smoke/evaluation allowed only inside SLURM')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No allocated CUDA device')
    if (suite / 'PREPARATION_FAILED.txt').exists():
        raise RuntimeError('Suite preparation failed')
    sys.path.insert(0, str(suite / 'code/scripts/published16'))
    import run_suite as helpers
    from aggregate import report_run
    from protocols import write_json
    helpers.verify_inputs(suite)
    spec = json.loads((suite / 'suite.json').read_text())
    os.environ['TORCH_HOME'] = spec['torch_cache']
    identifier = 'smoke' if smoke else spec['runs'][index]['id']
    with lock(suite / 'state' / (identifier + '.lock')):
        try:
            if smoke:
                completed = suite / 'state/smoke.complete.json'
                if completed.exists():
                    raise RuntimeError('Smoke already completed; refusing duplicate')
                versions = package_versions()
                if any(versions[p] == 'not-installed' for p in ('torch', 'torchvision', 'transformers', 'pyarrow')):
                    raise RuntimeError('Required training packages missing')
                # Verify every architecture AND loss/routing control, no test set.
                for arm in spec['arms']:
                    first = next(r for r in spec['runs'] if r['arm'] == arm)
                    sandbox = suite / 'smokes' / arm
                    sandbox.mkdir(parents=True, exist_ok=True)
                    (sandbox / 'state').mkdir(exist_ok=True)
                    (sandbox / 'logs').mkdir(exist_ok=True)
                    helpers.smoke(sandbox, first)
                    helpers.verify_inputs(suite)
                write_json(completed, {'package_versions': versions, 'arms': list(spec['arms']),
                           'job': os.environ['SLURM_JOB_ID'], 'at': helpers.now(),
                           'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
                           'scope': '8 train/8 validation per arm, no outer test'})
                return
            smoke_record = json.loads((suite / 'state/smoke.complete.json').read_text())
            if package_versions() != smoke_record['package_versions']:
                raise RuntimeError('Package versions changed after common smoke')
            run = spec['runs'][index]
            marker = suite / 'state' / (identifier + '.done.json')
            if marker.exists():
                raise RuntimeError('Completed run exists; refusing duplicate training')
            if helpers.run_directory(Path(run['output_root'])) is not None:
                raise RuntimeError('Existing partial output: manual inspection required; no auto resume')
            config = yaml.safe_load(Path(run['config']).read_text())
            write_json(suite / 'state' / (identifier + '.current.json'), {
                       'phase': 'training', 'job': os.environ['SLURM_JOB_ID'],
                       'gpu': torch.cuda.get_device_name(0), 'at': helpers.now()})
            helpers.execute(['scripts/train.py', '--config', run['config']],
                            suite / 'logs' / (identifier + '_train.log'))
            directory = helpers.run_directory(Path(run['output_root']))
            if directory is None or not helpers.trained(directory):
                raise RuntimeError('Training did not produce complete artifacts')
            helpers.verify_checkpoint(directory, config)
            helpers.verify_inputs(suite)
            helpers.execute(['scripts/eval_test.py', '--run-dir', str(directory),
                             '--batch-size', str(config['training']['batch_size']),
                             '--num-workers', str(config['training']['num_workers'])],
                            suite / 'logs' / (identifier + '_test.log'))
            report_run(run, directory, suite / 'reports' / identifier)
            helpers.verify_checkpoint(directory, config)
            helpers.verify_inputs(suite)
            write_json(marker, {'run_dir': str(directory), 'at': helpers.now(),
                               'job': os.environ['SLURM_JOB_ID'], 'arm': run['arm'], 'seed': run['seed']})
        except Exception as error:
            write_json(suite / 'state' / (identifier + '.failure.json'), {
                       'error': str(error), 'traceback': traceback.format_exc(), 'at': helpers.now()})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if not args.smoke and (args.index is None or args.index < 0):
        parser.error('Require --smoke or nonnegative --index')
    main(args.suite.resolve(), args.smoke, args.index)
