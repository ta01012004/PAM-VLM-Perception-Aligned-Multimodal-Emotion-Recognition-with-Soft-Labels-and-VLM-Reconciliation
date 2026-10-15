"""Preflight SLURM test-only, then submit validated non-overwriting jobs."""
from __future__ import annotations

import argparse
import getpass
from copy import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone


def job_id(reply):
    identifier = reply.strip().split(';')[0]
    if not identifier.isdigit():
        raise ValueError(f'Unexpected sbatch --parsable output: {reply!r}')
    return identifier


def resources(args, gpu=True):
    if not args.partition or args.cpus < 1 or not args.memory or not args.time:
        raise ValueError('Set partition/cpus/memory/time after current cluster inspection')
    command = [f'--partition={args.partition}', f'--cpus-per-task={args.cpus}',
               f'--mem={args.memory}', f'--time={args.time}']
    if gpu:
        if not args.gpu_request:
            raise ValueError('Set verified GPU GRES request, e.g. gpu:1')
        command.append(f'--gres={args.gpu_request}')
    for name in ('account', 'qos'):
        if getattr(args, name):
            command.append(f'--{name}={getattr(args, name)}')
    return command


def checked(command, cwd, env):
    result = subprocess.run(command, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(f'{command[0]} failed: {result.stderr.strip()}')
    return result.stdout.strip()


def submit(args):
    if os.name == 'nt':
        raise RuntimeError('Run this script after SSH on the Linux HPC login node')
    if os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Submit from login node, not a nested compute allocation')
    suite = args.suite.resolve()
    if (suite / 'PREPARATION_FAILED.txt').exists():
        raise RuntimeError('Preparation failed')
    spec = json.loads((suite / 'suite.json').read_text())
    conda = args.conda_init.resolve()
    if not conda.is_file():
        raise FileNotFoundError(conda)
    env = dict(os.environ, PAM_CONDA_INIT=str(conda), PAM_CONDA_ENV=args.conda_env)
    sys.path.insert(0, str(suite / 'code/scripts/published16'))
    from run_suite import verify_inputs
    verify_inputs(suite)
    # Read-only queue/policy record; let sbatch --test-only validate limits.
    queue = checked(['squeue', '-u', getpass.getuser(), '-o', '%i|%j|%T|%M|%R'], suite, env)
    policy = checked(['scontrol', 'show', 'partition', args.partition], suite, env)
    log = suite / 'logs'
    log.mkdir(exist_ok=True)
    if args.kind == 'reviewer':
        serial = getattr(args, 'serial', False)
        if not args.report_partition and not serial:
            raise ValueError('Specify a verified --report-partition for CPU reporting')
        report_args = copy(args)
        report_args.partition = args.report_partition
        report_args.cpus, report_args.memory, report_args.time = 4, '16G', '02:00:00'
        wrapper = Path(__file__).with_name('submit.sbatch').resolve()
        total = len(spec['runs'])
        if total < 1 or args.parallel < 1:
            raise ValueError('Empty suite or invalid array concurrency')
        commands = [] if serial else [
            resources(args) + ['--job-name=pam-review-smoke', f'--output={log}/smoke-%j.out',
                                str(wrapper), str(suite), 'smoke'],
            resources(args) + ['--job-name=pam-review', f'--array=0-{total-1}%{args.parallel}',
                                f'--output={log}/run-%A_%a.out', str(wrapper), str(suite), 'run'],
            resources(report_args, gpu=False) + ['--job-name=pam-review-report',
                                f'--output={log}/report-%j.out', str(wrapper), str(suite), 'analyze'],
        ]
        if serial:
            wrapper = Path(__file__).with_name('submit_serial.sbatch').resolve()
            commands = [resources(args) + ['--job-name=pam-review-serial',
                        f'--output={log}/serial-%j.out', str(wrapper), str(suite)]]
    else:
        if len(spec['runs']) != 10 or any(r['protocol'] != 'M10' for r in spec['runs']):
            raise ValueError('Expected ten prepared M10 controls')
        if args.repo is None or not args.repo.is_dir():
            raise ValueError('M10 needs --repo to resolve original offline cache')
        env['PAM_REPO'] = str(args.repo.resolve())
        wrapper = Path(__file__).parent.parent / 'pam_hard_ablation/submit.sbatch'
        commands = [resources(args) + ['--job-name=pam-hard-m10', f'--output={log}/hard-%j.out',
                                       str(wrapper.resolve()), str(suite)]]
    # Older clusters may ignore the environment-only dependency setting.
    external_dependency = env.get('SBATCH_DEPENDENCY')
    if external_dependency and not re.fullmatch(r'afterok:\d+(?::\d+)*', external_dependency):
        raise ValueError('External dependency must be afterok followed by numeric job IDs')
    initial_dependency = [f'--dependency={external_dependency}'] if external_dependency else []
    # No job is queued until EVERY proposed resource request passes test-only.
    validation = [checked(['sbatch', '--test-only', '--export=ALL', *initial_dependency, *command], suite, env) for command in commands]
    if args.dry_run:
        print(json.dumps({'policy': policy, 'queue': queue, 'commands': commands,
                          'test_only': validation, 'submitted': []}, indent=2))
        return
    lock = suite / '.submission.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    jobs = []
    record = {'at': datetime.now(timezone.utc).isoformat(), 'kind': args.kind,
              'policy': policy, 'queue_before': queue, 'test_only': validation,
              'jobs': jobs, 'commands': commands, 'cancelled': []}
    record['mode'] = 'serial' if getattr(args, 'serial', False) else 'standard'
    record['external_dependency'] = env.get('SBATCH_DEPENDENCY')
    # Keep this lock even on partial submission; never accidentally queue twice.
    with os.fdopen(fd, 'w') as f:
        f.write(json.dumps({'at': record['at'], 'pid': os.getpid()}))
    try:
        for i, command in enumerate(commands):
            dependency = [f'--dependency=afterok:{jobs[-1]}'] if i else initial_dependency
            reply = checked(['sbatch', '--parsable', '--export=ALL', *dependency, *command], suite, env)
            jobs.append(job_id(reply))
            (suite / 'submission.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(json.dumps(record, indent=2))
        print(checked(['squeue', '-j', ','.join(jobs), '-o', '%i|%j|%T|%R'], suite, env))
    except Exception as error:
        record['error'] = str(error)
        (suite / 'submission.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--kind', choices=('reviewer', 'm10-hard'), required=True)
    parser.add_argument('--partition', required=True)
    parser.add_argument('--report-partition')
    parser.add_argument('--gpu-request', required=True)
    parser.add_argument('--cpus', type=int, required=True)
    parser.add_argument('--memory', required=True)
    parser.add_argument('--time', required=True)
    parser.add_argument('--conda-init', type=Path, required=True)
    parser.add_argument('--conda-env', default='ta')
    parser.add_argument('--parallel', type=int, default=1)
    parser.add_argument('--serial', action='store_true',
                        help='Reviewer smoke/runs/report in one GPU job; respects low submit quotas')
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--account')
    parser.add_argument('--qos')
    parser.add_argument('--dry-run', action='store_true')
    submit(parser.parse_args())
