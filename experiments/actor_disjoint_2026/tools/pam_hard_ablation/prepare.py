"""Prepare ten original-PAM hard-label M10 controls; never submit/cancel jobs."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import yaml


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def make_hard(soft: dict, output: Path, identifier: str) -> dict:
    """Keep architecture, CE smoothing, seed, optimizer and budget unchanged."""
    if soft.get('baseline') != 'fusion' or not soft['loss']['soft_labels_enabled']:
        raise ValueError('Expected the original fusion CE+KL configuration.')
    hard = deepcopy(soft)
    hard['loss']['soft_labels_enabled'] = False
    hard['loss']['soft_weights'] = {key: 0.0 for key in soft['loss']['soft_weights']}
    hard['run_name'] = 'pam_hard_control_' + identifier
    hard['output_root'] = str(output / 'runs' / identifier)
    hard['data']['decode_error_report'] = str(output / 'logs' / (identifier + '_decode.csv'))
    return hard


def prepare(repo: Path, baseline: Path, output: Path, code_root: Path) -> None:
    repo, baseline, output, code_root = [p.resolve() for p in (repo, baseline, output, code_root)]
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if output.is_relative_to(baseline) or output.is_relative_to(code_root):
        raise ValueError('New outputs must not be inside the baseline or source snapshot.')
    spec = json.loads((baseline / 'suite.json').read_text())
    original_hashes = json.loads((baseline / 'input_hashes.json').read_text())
    runs = sorted((r for r in spec['runs'] if r['protocol'] == 'M10'), key=lambda r: r['fold'])
    if len(runs) != 10 or [r['fold'] for r in runs] != list(range(1, 11)):
        raise ValueError('Need exactly the ten original M10 folds.')
    # Check every frozen Python source, not merely files present in the candidate.
    expected_code = {}
    for name, hash_value in original_hashes.items():
        path = Path(name)
        try:
            relative = path.relative_to(repo)
        except ValueError:
            continue
        if relative.parts[0] == 'src' or str(relative).replace('\\', '/') in {
            'scripts/train.py', 'scripts/eval_test.py', 'scripts/plot_results.py'
        }:
            expected_code[relative] = hash_value
    if not any(p.parts[0] == 'src' for p in expected_code):
        raise ValueError('--repo does not match the repository paths in baseline hashes.')
    for relative, expected in expected_code.items():
        candidate = code_root / relative
        if not candidate.is_file() or digest(candidate) != expected:
            raise ValueError(f'Original source mismatch: {relative}; do not reuse soft results.')
    reporting = ['aggregate.py', 'protocols.py', 'run_suite.py']
    for filename in reporting:
        relative = Path('scripts/published16') / filename
        expected = original_hashes.get(str(repo / relative))
        if expected is None or digest(code_root / relative) != expected:
            raise ValueError(f'Original reporting/helper source mismatch: {relative}')
        expected_code[relative] = expected
    # Validate all referenced immutable baseline inputs. Ignore modified live source:
    # training will use the byte-identical, isolated source snapshot instead.
    frozen = {}
    for name, expected in original_hashes.items():
        path = Path(name)
        # Only original live source paths may differ: execution uses the checked
        # byte-identical snapshot. Freeze all other inputs, even outside baseline.
        if path.is_relative_to(repo) and path.relative_to(repo) in expected_code:
            continue
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Baseline input changed: {path}')
        frozen[str(path)] = expected
    prepared = []
    for run in runs:
        marker = baseline / 'state' / (run['id'] + '.done.json')
        directory = Path(json.loads(marker.read_text())['run_dir'])
        soft = yaml.safe_load(Path(run['config']).read_text())
        resolved = json.loads((directory / 'resolved_config.json').read_text())
        recorded = {k: v for k, v in resolved.items() if k not in {'run_id', 'slurm_job_id'}}
        if recorded != soft:
            raise ValueError(f'Soft run resolved configuration differs: {run["id"]}')
        for filename in ('test_predictions.csv', 'test_metrics.json', 'history.json',
                         'metrics.json', 'reproducibility.json', 'resolved_config.json',
                         'checkpoints/best.pt'):
            path = directory / filename
            if not path.is_file():
                raise ValueError(f'Soft run incomplete: {path}')
            frozen[str(path)] = digest(path)
        frozen[str(marker)] = digest(marker)
        prepared.append((run, directory, soft))
    output.mkdir(parents=True)
    try:
        for name in ('configs', 'logs', 'state', 'reports', 'code'):
            (output / name).mkdir()
        for relative in expected_code:
            destination = output / 'code' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(code_root / relative, destination)
        shutil.copyfile(Path(__file__).with_name('run.py'), output / 'run.py')
        controls = []
        for run, directory, soft in prepared:
            hard = make_hard(soft, output, run['id'])
            path = output / 'configs' / (run['id'] + '.yaml')
            path.write_text(yaml.safe_dump(hard, sort_keys=False), encoding='utf-8')
            item = deepcopy(run)
            item.update(config=str(path), output_root=hard['output_root'],
                        soft_run_dir=str(directory), soft_config=run['config'])
            controls.append(item)
        (output / 'suite.json').write_text(json.dumps({
            'runs': controls, 'baseline_suite': str(baseline),
            'experiment': 'Original PAM: CE-only versus existing CE+KL, M10, seed42',
            'changes': ['disable perceptual KL only; retain CE label smoothing'],
            'resume_policy': 'no automatic partial-checkpoint resume',
            'limitations': 'Across-fold SD is not seed uncertainty; historical tuning remains.'
        }, indent=2), encoding='utf-8')
        for directory in ('code', 'configs'):
            for path in (output / directory).rglob('*'):
                if path.is_file():
                    frozen[str(path)] = digest(path)
        for filename in ('run.py', 'suite.json'):
            frozen[str(output / filename)] = digest(output / filename)
        (output / 'input_hashes.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    except Exception:
        # Keep the partial preparation for diagnosis; never recursively erase it.
        (output / 'PREPARATION_FAILED.txt').write_text('Do not submit this directory.\n')
        raise
    print(f'Prepared 10 hard controls; 10 existing soft runs remain untouched: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('repo', 'baseline', 'output', 'code-root'):
        parser.add_argument('--' + option, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.repo, args.baseline, args.output, args.code_root)
