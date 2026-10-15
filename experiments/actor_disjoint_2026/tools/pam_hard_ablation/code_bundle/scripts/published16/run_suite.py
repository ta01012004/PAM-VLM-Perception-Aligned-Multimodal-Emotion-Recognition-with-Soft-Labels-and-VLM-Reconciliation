"""Sequential, fail-fast SLURM runner with independently validated plots."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import pandas as pd
import yaml

from aggregate import aggregate, report_run
from protocols import digest, write_json

ROOT = Path(__file__).resolve().parents[2]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def suite_lock(suite: Path):
    """Fail safely on duplicate execution; stale locks need manual inspection."""
    lock = suite / '.runner.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump({'pid': os.getpid(), 'job_id': os.environ.get('SLURM_JOB_ID'), 'at': now()}, handle)
        yield
    finally:
        lock.unlink()


def verify_config(expected: dict, checkpoint: dict, evaluation: dict) -> None:
    for key in ('model_config', 'data', 'loss', 'training', 'seed', 'baseline', 'optimizer', 'published_protocol'):
        if checkpoint[key] != expected[key] or evaluation[key] != expected[key]:
            raise ValueError(f'Checkpoint/evaluation config differs: {key}')


def verify_inputs(suite: Path) -> None:
    hashes = json.loads((suite / 'input_hashes.json').read_text())
    changed = [path for path, expected in hashes.items() if not Path(path).is_file() or digest(Path(path)) != expected]
    if changed:
        raise RuntimeError(f'Frozen inputs changed; refusing to run: {changed[:5]}')


def execute(command: list[str], log: Path) -> None:
    print(f'{now()} START {command}; log={log}', flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as handle:
        handle.write(f'\n{now()} START {command}\n')
        handle.flush()
        subprocess.run(['srun', '--unbuffered', '--kill-on-bad-exit=1', sys.executable, *command], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    print(f'{now()} FINISHED {command[0]}', flush=True)


def run_directory(output_root: Path) -> Path | None:
    directories = sorted(p for p in output_root.glob('*') if p.is_dir())
    if len(directories) > 1:
        raise RuntimeError(f'Ambiguous multiple attempts under {output_root}')
    return directories[0] if directories else None


def trained(directory: Path) -> bool:
    return all((directory / name).is_file() for name in ('checkpoints/best.pt', 'history.json', 'metrics.json', 'reproducibility.json'))


def verify_checkpoint(directory: Path, config: dict) -> None:
    import torch
    payload = torch.load(directory / 'checkpoints/best.pt', map_location='cpu', weights_only=False)
    if payload['split_checksum'] != Path(config['data']['split_checksum']).read_text().strip():
        raise ValueError('Checkpoint split checksum mismatch')
    if payload['label_mapping'] != json.loads(Path(config['data']['label_mapping']).read_text()):
        raise ValueError('Checkpoint label mapping mismatch')
    verify_config(config, payload['resolved_config'], json.loads((directory / 'resolved_config.json').read_text()))


def smoke(suite: Path, first: dict) -> None:
    marker = suite / 'state/smoke.done.json'
    if marker.exists():
        return
    folder = suite / 'smoke'
    if folder.exists():
        raise RuntimeError('Previous incomplete smoke test exists; inspect rather than overwrite')
    folder.mkdir()
    config = yaml.safe_load(Path(first['config']).read_text())
    config['run_name'] = 'published16_smoke_not_a_scientific_run'
    config['output_root'] = str(folder / 'runs')
    config['training'].update(epochs=1, num_workers=2, gradient_accumulation_steps=1, early_stopping_patience=0)
    for part in ('train', 'val'):
        data = pd.read_csv(config['data'][part + '_manifest']).iloc[:8]
        path = folder / f'manifest_{part}.csv'
        data.to_csv(path, index=False)
        config['data'][part + '_manifest'] = str(path)
    config['data']['decode_error_report'] = str(folder / 'decode_errors.csv')
    checksum = folder / 'checksum.txt'
    checksum.write_text(digest(folder / 'manifest_train.csv') + digest(folder / 'manifest_val.csv'))
    config['data']['split_checksum'] = str(checksum)
    path = folder / 'config.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    execute(['scripts/train.py', '--config', str(path)], suite / 'logs/smoke.log')
    directory = run_directory(folder / 'runs')
    if directory is None or not trained(directory):
        raise RuntimeError('Smoke test did not produce complete artifacts')
    write_json(marker, {'finished_at': now(), 'run_dir': str(directory), 'scope': '8 train + 8 validation clips, no outer-test evaluation'})


def run(suite: Path) -> None:
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Submit through SLURM; never train on login')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA allocation unavailable')
    verify_inputs(suite)
    spec = json.loads((suite / 'suite.json').read_text())
    if len(spec['runs']) != 16:
        raise ValueError('Expected exactly 16 pre-registered runs')
    (suite / 'state').mkdir(exist_ok=True)
    hardware = {'job_id': os.environ['SLURM_JOB_ID'], 'hostname': os.uname().nodename,
                'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'cuda': torch.version.cuda,
                'cpus': os.environ.get('SLURM_CPUS_PER_TASK'), 'started_at': now()}
    write_json(suite / 'state' / ('hardware_' + os.environ['SLURM_JOB_ID'] + '.json'), hardware)
    write_json(suite / 'state/current.json', {'phase': 'smoke', **hardware})
    smoke(suite, spec['runs'][0])
    for number, spec_run in enumerate(spec['runs'], 1):
        marker = suite / 'state' / (spec_run['id'] + '.done.json')
        verify_inputs(suite)
        if marker.exists():
            directory = Path(json.loads(marker.read_text())['run_dir'])
            verify_checkpoint(directory, yaml.safe_load(Path(spec_run['config']).read_text()))
            report_run(spec_run, directory, suite / 'reports' / spec_run['id'])
            print(f'Reusing validated completed run {spec_run["id"]}', flush=True)
            continue
        state = {'run': spec_run['id'], 'index': number, 'expected': 16, 'job_id': os.environ['SLURM_JOB_ID']}
        config = yaml.safe_load(Path(spec_run['config']).read_text())
        output_root = Path(spec_run['output_root'])
        directory = run_directory(output_root)
        if directory is not None and not trained(directory):
            raise RuntimeError(f'Incomplete previous training found at {directory}; automatic checkpoint resume is disabled')
        if directory is None:
            write_json(suite / 'state/current.json', {**state, 'phase': 'training', 'started_at': now()})
            execute(['scripts/train.py', '--config', spec_run['config']], suite / 'logs' / (spec_run['id'] + '_train.log'))
            directory = run_directory(output_root)
        if directory is None or not trained(directory):
            raise RuntimeError(f'Training artifacts incomplete: {spec_run["id"]}')
        verify_inputs(suite)
        verify_checkpoint(directory, config)
        write_json(suite / 'state/current.json', {**state, 'phase': 'test', 'started_at': now(), 'run_dir': str(directory)})
        if not ((directory / 'test_predictions.csv').exists() and (directory / 'test_metrics.json').exists()):
            execute(['scripts/eval_test.py', '--run-dir', str(directory), '--batch-size', '4', '--num-workers', '12'], suite / 'logs' / (spec_run['id'] + '_test.log'))
        report_run(spec_run, directory, suite / 'reports' / spec_run['id'])
        write_json(marker, {**state, 'run_dir': str(directory), 'finished_at': now()})
        aggregate(suite)
        print(f'{now()} COMPLETED {number}/16: {spec_run["id"]}', flush=True)
    status = aggregate(suite)
    if not status['complete']:
        raise RuntimeError('Suite incomplete')
    write_json(suite / 'state/current.json', {'phase': 'complete', 'completed': 16, 'finished_at': now(), 'job_id': os.environ['SLURM_JOB_ID']})
    print(f'ALL 16 RUNS COMPLETE. Plots and tables: {suite / "reports"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    args = parser.parse_args()
    suite_path = args.suite.resolve()
    with suite_lock(suite_path):
        try:
            run(suite_path)
        except Exception as exc:
            write_json(suite_path / 'state/failure.json', {'at': now(), 'error': str(exc), 'traceback': traceback.format_exc()})
            traceback.print_exc()
            sys.exit(1)
