"""Pre-registered actor splits; no model scores enter split construction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

ACTORS = list(range(1001, 1092))
LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
EXCLUDED_H5 = ['1076_MTI_NEU_XX', '1076_MTI_SAD_XX', '1064_TIE_SAD_XX', '1064_IEO_DIS_MD']


def validate_partition(actors: dict[str, list[int]]) -> None:
    flat = [a for k in ('train', 'val', 'test') for a in actors[k]]
    if sorted(flat) != ACTORS or any(not actors[k] for k in actors):
        raise ValueError('Actors must partition all 91 identities without overlap')


def build_splits() -> list[dict]:
    splits = []
    shuffled = ACTORS.copy()
    random.Random(42).shuffle(shuffled)
    m10 = [shuffled[:10]] + [shuffled[10 + i * 9:19 + i * 9] for i in range(9)]
    for protocol, folds in [('H5', [ACTORS[i * 18:(i + 1) * 18] if i < 4 else ACTORS[72:] for i in range(5)]), ('M10', m10)]:
        for index, test in enumerate(folds, 1):
            remaining = sorted(set(ACTORS) - set(test))
            random.Random(4200 + index).shuffle(remaining)
            actors = {'train': sorted(remaining[9:]), 'val': sorted(remaining[:9]), 'test': sorted(test)}
            validate_partition(actors)
            splits.append({'id': f'{protocol.lower()}_fold{index:02d}', 'protocol': protocol,
                           'fold': index, 'training_seed': 42, 'split_seed': None if protocol == 'H5' else 42,
                           'validation_seed': 4200 + index, 'actors': actors,
                           'excluded_clips': EXCLUDED_H5 if protocol == 'H5' else [],
                           'comparability': 'exact_public_outer_test_only_reduced_train' if protocol == 'H5' else 'protocol_family_new_actor_assignment'})
    holdout = ACTORS.copy()
    random.Random(0).shuffle(holdout)
    actors = {'train': sorted(holdout[:55]), 'val': sorted(holdout[55:73]), 'test': sorted(holdout[73:])}
    validate_partition(actors)
    splits.append({'id': 't602020_seed0', 'protocol': 'T602020', 'fold': 1, 'training_seed': 42,
                   'split_seed': 0, 'validation_seed': None, 'actors': actors, 'excluded_clips': [],
                   'comparability': 'protocol_family_new_actor_assignment_single_trial'})
    return splits


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)
