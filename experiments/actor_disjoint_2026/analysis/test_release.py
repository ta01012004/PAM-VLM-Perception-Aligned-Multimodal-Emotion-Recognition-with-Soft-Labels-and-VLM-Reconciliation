"""CPU checks for the public evidence package; no datasets or GPUs required."""
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_release import (calculate_metrics, safe_file, validate_catalog,
                              validate_release, validate_reviewer_aggregates)

ROOT = Path(__file__).resolve().parents[1]


def test_complete_release_recomputes_all_59_runs():
    result = validate_release(ROOT)
    assert result['run_counts'] == {'primary': 16, 'ce_only': 10, 'reviewer': 33}
    assert result['max_metric_error'] < 1e-7
    assert result['primary_pooled_counts'] == {'H5': 7438, 'M10': 7442}
    assert result['reviewer_unique_test_clips'] == 1476


def test_catalog_rejects_missing_duplicate_or_unknown_runs():
    frame = pd.read_csv(ROOT / 'runs.csv')
    with pytest.raises(ValueError):
        validate_catalog(frame.iloc[:-1], ROOT)
    with pytest.raises(ValueError):
        validate_catalog(pd.concat([frame, frame.iloc[[0]]]), ROOT)
    changed = frame.copy()
    changed.loc[0, 'run'] = 'unknown'
    with pytest.raises(ValueError):
        validate_catalog(changed, ROOT)


@pytest.mark.parametrize('name', ['../outside.csv', '/tmp/outside.csv'])
def test_artifact_paths_cannot_escape_package(name):
    with pytest.raises(ValueError):
        safe_file(ROOT, name)


def test_ece_includes_unit_confidence_in_last_bin():
    labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
    frame = pd.DataFrame({'true': labels, 'pred': labels})
    for i, label in enumerate(labels):
        frame['p_' + label] = np.eye(6)[:, i]
    scores = calculate_metrics(frame)
    assert scores['ece'] == 0
    assert sum(b['count'] for b in scores['bins']) == 6
    assert scores['bins'][-1]['count'] == 6


@pytest.mark.parametrize('bad', [np.nan, -0.1, 1.1])
def test_reject_invalid_probability_vectors(bad):
    frame = pd.read_csv(ROOT / pd.read_csv(ROOT / 'runs.csv').iloc[0].prediction_path)
    frame.loc[0, 'p_angry'] = bad
    with pytest.raises(ValueError):
        calculate_metrics(frame)


def test_saved_label_must_match_probability_argmax():
    frame = pd.read_csv(ROOT / pd.read_csv(ROOT / 'runs.csv').iloc[0].prediction_path)
    frame.loc[0, 'pred'] = 'fear' if frame.loc[0, 'pred'] != 'fear' else 'angry'
    with pytest.raises(ValueError):
        calculate_metrics(frame)


@pytest.mark.parametrize('report', ['all_runs.csv', 'seed_mean_sd.csv',
                                   'paired_seed_deltas.csv', 'additional_paired_contrasts.csv'])
def test_modified_reviewer_aggregate_is_rejected(tmp_path, report):
    source = ROOT / 'evidence/reviewer/reports'
    destination = tmp_path / 'reports'
    shutil.copytree(source, destination)
    path = destination / report
    if report == 'seed_mean_sd.csv':
        frame = pd.read_csv(path, header=[0, 1], index_col=0)
        frame.iloc[0, 0] += .1
        frame.to_csv(path)
    else:
        frame = pd.read_csv(path)
        metric = 'macro_f1_minus_A0' if report == 'paired_seed_deltas.csv' else 'macro_f1'
        frame.loc[0, metric] += .1
        frame.to_csv(path, index=False)
    rows = pd.read_csv(ROOT / 'runs.csv')
    with pytest.raises((ValueError, AssertionError)):
        validate_reviewer_aggregates(rows[rows.group == 'reviewer'], destination)
