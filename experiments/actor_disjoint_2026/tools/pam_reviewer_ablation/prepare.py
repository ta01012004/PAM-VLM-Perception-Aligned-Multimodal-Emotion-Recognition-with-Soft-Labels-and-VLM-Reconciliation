"""Recover immutable T602020 inputs and create a NEW paired reviewer suite."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import pandas as pd
import yaml

from variants import ARMS, variant


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')


def validate_partition(paths):
    frames = {split: pd.read_csv(path) for split, path in paths.items()}
    actors = {split: set(frame.actor_id) for split, frame in frames.items()}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if actors[a] & actors[b]:
            raise ValueError(f'actor overlap: {a}/{b}')
        if set(frames[a].clip_id) & set(frames[b].clip_id):
            raise ValueError('clip overlap')
    if [len(actors[s]) for s in ('train', 'val', 'test')] != [55, 18, 18]:
        raise ValueError('Expected original T602020 55/18/18 actors')
    for split, frame in frames.items():
        if frame.clip_id.duplicated().any() or set(frame.split) != {split}:
            raise ValueError(f'Invalid manifest: {split}')
    return frames


def snapshot(path, model_type):
    path = path.absolute()
    if path.parent.name != 'snapshots' or not re.fullmatch('[0-9a-f]{40}', path.name):
        raise ValueError('Use a pinned HF cache snapshots/<40-character commit> directory')
    expected_repo = {'wavlm': 'models--microsoft--wavlm-base-plus',
                     'wav2vec2': 'models--facebook--wav2vec2-base'}[model_type]
    if path.parent.parent.name != expected_repo:
        raise ValueError(f'Expected pinned repository cache: {expected_repo}')
    config = json.loads((path / 'config.json').read_text())
    if config.get('model_type') != model_type or config.get('hidden_size') != 768:
        raise ValueError(f'Expected {model_type} base hidden_size=768')
    weights = list(path.glob('*.safetensors')) + list(path.glob('pytorch_model*.bin'))
    if not weights:
        raise ValueError('Offline snapshot has no model weights')
    if model_type == 'wav2vec2':
        processor = json.loads((path / 'preprocessor_config.json').read_text())
        if (config.get('feat_extract_norm') != 'group' or
                processor.get('sampling_rate') != 16000 or
                not processor.get('do_normalize') or processor.get('return_attention_mask', False)):
            raise ValueError('Alternative expects wav2vec2-base group norm,16kHz, normalize,no forward mask')
    return path


def prepare(args):
    baseline, output = args.baseline.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing overwrite: {output}')
    if output.is_relative_to(baseline):
        raise ValueError('Output must be separate from original suite')
    original_hashes = json.loads((baseline / 'input_hashes.json').read_text())
    verified_paths = {}
    def verified(path):
        path = Path(path)
        expected = original_hashes.get(str(path))
        if expected is None or digest(path) != expected:
            raise ValueError(f'Immutable input missing/hash mismatch: {path}')
        verified_paths[str(path.absolute())] = expected
        return path
    verified(baseline / 'suite.json')
    runs = json.loads((baseline / 'suite.json').read_text())['runs']
    holdouts = [r for r in runs if r['protocol'] == 'T602020']
    if len(holdouts) != 1:
        raise ValueError('Require exactly one original prespecified T602020 assignment')
    reference = holdouts[0]
    config = yaml.safe_load(verified(reference['config']).read_text())
    if config['baseline'] != 'fusion' or not config['loss']['soft_labels_enabled']:
        raise ValueError('Require original full PAM CE+KL config')
    if config['training']['primary_metric'] != 'macro_f1':
        raise ValueError('Preserve validation Macro-F1 checkpoint selection')
    model = json.loads(verified(config['model_config']).read_text())
    if model['audio']['backend'] != 'transformers' or model['video']['backend'] != 'torchvision':
        raise ValueError('Real suite cannot use tiny test backbones')
    if not model['audio']['encoder_id'].endswith('wavlm-base-plus') or model['video']['backbone_id'] != 'efficientnet_b0':
        raise ValueError('Unexpected original backbone pair; inspect rather than guess')
    frames = validate_partition({s: verified(config['data'][s + '_manifest'])
                                 for s in ('train', 'val', 'test')})
    loader = json.loads(verified(config['data']['loader_config']).read_text())
    if loader['sample_rate'] != 16000:
        raise ValueError('Pretrained raw-waveform encoders require16kHz')
    for key in ('label_mapping', 'split_checksum'):
        verified(config['data'][key])
    for path in config['data']['soft_labels'].values():
        verified(path)
    # Assert corpus availability, not only a manifest that names missing files.
    raw = Path(config['data']['raw_root'])
    raw_paths = set()
    for frame in frames.values():
        for column in ('audio_path', 'video_path'):
            for value in frame[column]:
                p = Path(value)
                media = p if p.is_absolute() else raw / p
                if not media.is_file():
                    raise FileNotFoundError(f'Missing raw input: {value}')
                raw_paths.add(media.absolute())
    original_audio = snapshot(args.original_audio_snapshot, 'wavlm')
    model['audio']['encoder_id'] = str(original_audio)
    if any(a in args.arms for a in ('A7', 'A9')) and args.audio_snapshot is None:
        raise ValueError('A7/A9 require --audio-snapshot')
    alternative = snapshot(args.audio_snapshot, 'wav2vec2') if any(a in args.arms for a in ('A7', 'A9')) else None
    resnet = args.resnet_weights.resolve() if args.resnet_weights else None
    if any(a in args.arms for a in ('A8', 'A9')):
        if resnet is None or not resnet.is_file() or not digest(resnet).startswith('f37072fd'):
            raise ValueError('Need official ResNet18 IMAGENET1K_V1 resnet18-f37072fd.pth')
    if not args.torch_cache.is_dir():
        raise FileNotFoundError('Original Torch pretrained cache missing')
    efficient = args.torch_cache / 'hub/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth'
    if not efficient.is_file() or not digest(efficient).startswith('7f5810bc'):
        raise ValueError('Require offline original EfficientNetB0 DEFAULT weights')
    model['video']['weights_path'] = str(efficient.absolute())
    source = Path(__file__).parent
    # Arm A0 is always fresh in this modified source, NEVER a historical checkpoint.
    output.mkdir(parents=True)
    try:
        for name in ('configs', 'models', 'logs', 'state', 'reports'):
            (output / name).mkdir()
        shutil.copytree(source / 'code_bundle', output / 'code',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for name in ('run.py', 'analyze.py'):
            shutil.copyfile(source / name, output / name)
        canonical = output / 'models/canonical_A0.json'
        write(canonical, model)
        planned = []
        for seed in args.seeds:
            for arm in args.arms:
                name = f'{arm}_seed{seed}'
                cfg, mdl = variant(arm, config, model, audio_snapshot=alternative,
                                   resnet_weights=resnet)
                model_path = output / 'models' / (name + '.json')
                write(model_path, mdl)
                cfg.update(seed=seed, run_name='reviewer_' + name,
                           model_config=str(model_path), canonical_model_config=str(canonical),
                           output_root=str(output / 'runs' / name))
                cfg['published_protocol']['training_seed'] = seed
                cfg['reviewer_variant'] = {'arm': arm, 'seed': seed, 'original_assignment': reference['id']}
                cfg['data']['decode_error_report'] = str(output / 'logs' / (name + '_decode.csv'))
                cfg_path = output / 'configs' / (name + '.yaml')
                cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
                planned.append({**reference, 'id': name, 'arm': arm, 'seed': seed,
                                'training_seed': seed,
                                'config': str(cfg_path), 'output_root': cfg['output_root'],
                                'comparability': 'fresh_paired_holdout_not_published_leaderboard'})
        write(output / 'suite.json', {
            'runs': planned, 'arms': {a: ARMS[a] for a in args.arms}, 'seeds': args.seeds,
            'baseline_suite': str(baseline), 'torch_cache': str(args.torch_cache.absolute()),
            'expected': len(planned), 'historical_results_reused': False,
            'limitations': ['one retrospective actor assignment', 'fold SD is not seed SD',
                           'H0 vs A0 changes KL and smoothing; H0 vs A1 isolates smoothing',
                           'A6 changes objective scale', 'backbone pretraining/capacity differ',
                           'Replacement encoders include checkpoint-required waveform/frame normalization',
                           'Original A0 video preprocessing retained; ResNet uses ImageNet channel normalization',
                           'unused fusion modules allocated but frozen; not a smaller allocated model'],
        })
        frozen = {str(verified(p)): h for p, h in original_hashes.items()
                  if Path(p).is_relative_to(baseline)}
        frozen.update(verified_paths)
        frozen.update({str(p): digest(p) for p in raw_paths})
        for folder in (output / 'code', output / 'configs', output / 'models',
                       original_audio, alternative):
            if folder is not None:
                frozen.update({str(p.absolute()): digest(p) for p in folder.rglob('*') if p.is_file()})
        for p in (output / 'suite.json', output / 'run.py', output / 'analyze.py', efficient, resnet):
            if p is not None:
                frozen[str(p.absolute())] = digest(p)
        write(output / 'input_hashes.json', frozen)
    except Exception:
        (output / 'PREPARATION_FAILED.txt').write_text('Do not submit. Inspect retained partial preparation.\n')
        raise
    print(f'Prepared {len(planned)} FRESH paired runs. No jobs submitted. {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'output', 'original-audio-snapshot', 'torch-cache'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--audio-snapshot', type=Path)
    parser.add_argument('--resnet-weights', type=Path)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    args = parser.parse_args()
    if 'A0' not in args.arms or len(set(args.arms)) != len(args.arms) or len(set(args.seeds)) != len(args.seeds):
        parser.error('Require A0 and unique arms/seeds')
    prepare(args)
