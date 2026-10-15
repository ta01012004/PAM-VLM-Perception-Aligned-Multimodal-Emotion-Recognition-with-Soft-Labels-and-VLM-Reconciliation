"""Validated per-run and per-protocol reporting; never pool protocols."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from protocols import LABELS, write_json


def calibration(probabilities: np.ndarray, targets: np.ndarray) -> dict:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(targets, dtype=int)
    if p.ndim != 2 or len(p) == 0 or y.shape != (len(p),) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-6):
        raise ValueError('Invalid probabilities/targets')
    if (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError('Invalid target IDs')
    confidence = p.max(1)
    correct = p.argmax(1) == y
    bins = []
    index = np.minimum(np.searchsorted(np.linspace(0, 1, 16), confidence, side='right') - 1, 14)
    for i in range(15):
        selected = index == i
        bins.append({'count': int(selected.sum()), 'confidence': float(confidence[selected].mean()) if selected.any() else 0.,
                     'accuracy': float(correct[selected].mean()) if selected.any() else 0.})
    if sum(b['count'] for b in bins) != len(p):
        raise ValueError('Calibration bins omitted samples')
    return {'ece': sum(b['count'] / len(p) * abs(b['confidence'] - b['accuracy']) for b in bins),
            'nll': float(-np.log(np.clip(p[np.arange(len(p)), y], 1e-12, 1)).mean()),
            'brier': float(((p - np.eye(p.shape[1])[y]) ** 2).sum(1).mean()), 'bins': bins}


def validate_predictions(pred: pd.DataFrame, manifest: pd.DataFrame) -> None:
    if pred.clip_id.duplicated().any() or manifest.clip_id.duplicated().any() or set(pred.clip_id) != set(manifest.clip_id):
        raise ValueError('Predictions must match every test clip exactly once')
    joined = pred.set_index('clip_id').join(manifest.set_index('clip_id')[['actor_id', 'emotion']], rsuffix='_expected')
    if not (joined.actor_id == joined.actor_id_expected).all() or not (joined['true'] == joined.emotion).all():
        raise ValueError('Actor or target mismatch')
    if not set(pred['true']).issubset(LABELS) or not set(pred.pred).issubset(LABELS):
        raise ValueError('Unexpected labels')
    p = pred[['p_' + lab for lab in LABELS]].to_numpy()
    y = pred['true'].map(dict(zip(LABELS, range(6)))).to_numpy()
    calibration(p, y)
    if not np.array_equal(np.asarray(LABELS)[p.argmax(1)], pred.pred.to_numpy()):
        raise ValueError('Prediction/probability mismatch')


def metrics(pred: pd.DataFrame) -> dict:
    ids = dict(zip(LABELS, range(6)))
    y = pred['true'].map(ids).to_numpy()
    hat = pred.pred.map(ids).to_numpy()
    result = calibration(pred[['p_' + label for label in LABELS]].to_numpy(), y)
    result.update(accuracy=float((y == hat).mean()),
                  macro_f1=float(f1_score(y, hat, labels=list(range(6)), average='macro', zero_division=0)),
                  uar=float(recall_score(y, hat, labels=list(range(6)), average='macro', zero_division=0)),
                  num_samples=len(pred), confusion_matrix=confusion_matrix(y, hat, labels=list(range(6))).tolist())
    return result


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def diagnostic_figure(result: dict, title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    cm = np.asarray(result['confusion_matrix'], dtype=float)
    norm = np.divide(cm, cm.sum(1, keepdims=True), out=np.zeros_like(cm), where=cm.sum(1, keepdims=True) != 0)
    im = axes[0].imshow(norm, vmin=0, vmax=1, cmap='Blues')
    for i in range(6):
        for j in range(6):
            axes[0].text(j, i, f'{norm[i, j]:.2f}', ha='center', va='center', fontsize=8, color='white' if norm[i, j] > .55 else 'black')
    axes[0].set(xticks=range(6), xticklabels=LABELS, yticks=range(6), yticklabels=LABELS, xlabel='Predicted', ylabel='True', title='Row-normalised confusion')
    axes[0].tick_params(axis='x', rotation=40)
    fig.colorbar(im, ax=axes[0], fraction=.046)
    nonempty = [b for b in result['bins'] if b['count']]
    axes[1].plot([0, 1], [0, 1], '--', color='gray')
    axes[1].plot([b['confidence'] for b in nonempty], [b['accuracy'] for b in nonempty], 'o-')
    axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean confidence', ylabel='Accuracy', title=f'Reliability (15 bins), ECE={result["ece"]:.4f}')
    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, path)


def report_run(run: dict, directory: Path, output: Path) -> tuple[dict, pd.DataFrame]:
    pred = pd.read_csv(directory / 'test_predictions.csv')
    validate_predictions(pred, pd.read_csv(run['test_manifest']))
    result = metrics(pred)
    original = json.loads((directory / 'test_metrics.json').read_text())
    for key in ('accuracy', 'macro_f1', 'uar', 'num_samples'):
        if not np.isclose(original[key], result[key], atol=1e-7):
            raise ValueError(f'Recomputed {key} disagrees with saved metrics: {run["id"]}')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'validated_metrics.json', result)
    diagnostic_figure(result, run['id'] + ' — test', output / 'test_diagnostics')
    history = json.loads((directory / 'history.json').read_text())['epochs']
    if not history:
        raise ValueError('Empty training history')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [r['epoch'] + 1 for r in history]
    for part in ('train', 'val'):
        axes[0].plot(epochs, [r[part + '_loss'] for r in history], label=part)
        axes[1].plot(epochs, [r[part + '_metrics']['macro_f1'] for r in history], label=part)
    axes[0].set(xlabel='Epoch', ylabel='Loss')
    axes[1].set(xlabel='Epoch', ylabel='Macro-F1', ylim=(0, 1))
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.2)
    fig.suptitle(run['id'] + ' — training / validation (not test)')
    fig.tight_layout()
    save_figure(fig, output / 'learning_curves')
    row = {'run': run['id'], 'protocol': run['protocol'], 'fold': run['fold'], 'directory': str(directory),
           'epochs': len(history), 'comparability': run['comparability'],
           **{k: result[k] for k in ('num_samples', 'accuracy', 'macro_f1', 'uar', 'ece', 'nll', 'brier')}}
    write_json(output / 'row.json', row)
    return row, pred


def aggregate(suite: Path) -> dict:
    spec = json.loads((suite / 'suite.json').read_text())
    output = suite / 'reports'
    output.mkdir(exist_ok=True)
    rows, predictions = [], {}
    for run in spec['runs']:
        marker = suite / 'state' / (run['id'] + '.done.json')
        if not marker.exists():
            continue
        directory = Path(json.loads(marker.read_text())['run_dir'])
        row, pred = report_run(run, directory, output / run['id'])
        rows.append(row)
        predictions[run['id']] = pred
    status = {'completed': len(rows), 'expected': 16, 'complete': len(rows) == 16, 'protocols': {}}
    if not rows:
        write_json(output / 'summary.json', status)
        return status
    all_rows = pd.DataFrame(rows)
    all_rows.to_csv(output / 'per_run_metrics.csv', index=False)
    for protocol in ('H5', 'M10', 'T602020'):
        expected = [r for r in spec['runs'] if r['protocol'] == protocol]
        block = all_rows[all_rows.protocol == protocol].sort_values('fold')
        if block.empty:
            continue
        complete = len(block) == len(expected)
        summary = {'complete': complete, 'n_completed': len(block), 'n_expected': len(expected),
                   'aggregation': 'unweighted fold mean; sample SD across folds (not training seeds)',
                   'metrics': {key: {'mean': float(block[key].mean()), 'sd': float(block[key].std(ddof=1)) if len(block) > 1 else None} for key in ('accuracy', 'macro_f1', 'uar', 'ece', 'nll', 'brier')}}
        label = f'{protocol}: {len(block)}/{len(expected)} runs' + ('' if complete else ' — PARTIAL')
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for key in ('accuracy', 'macro_f1', 'uar'):
            axes[0].plot(block.fold, block[key], 'o-', label=key)
        axes[0].set(xlabel='Fold / trial', ylabel='Score', ylim=(0, 1), xticks=block.fold)
        axes[0].legend()
        axes[1].bar(block.fold, block.ece, color='#e49145')
        axes[1].set(xlabel='Fold / trial', ylabel='ECE (lower is better)', xticks=block.fold)
        fig.suptitle(label)
        fig.tight_layout()
        save_figure(fig, output / (protocol + '_fold_metrics'))
        if complete:
            pooled = pd.concat([predictions[r['id']] for r in expected], ignore_index=True)
            manifests = pd.concat([pd.read_csv(r['test_manifest']) for r in expected], ignore_index=True)
            validate_predictions(pooled, manifests)
            pooled.to_csv(output / (protocol + '_predictions.csv'), index=False)
            pooled_metrics = metrics(pooled)
            summary['pooled_oof' if protocol != 'T602020' else 'holdout'] = pooled_metrics
            diagnostic_figure(pooled_metrics, protocol + (' — pooled out-of-fold (not fold mean)' if protocol != 'T602020' else ' — single holdout'), output / (protocol + '_diagnostics'))
        status['protocols'][protocol] = summary
    write_json(output / 'summary.json', status)
    lines = ['# PAM-Soft published-protocol results', '', f'Completed: {len(rows)}/16.', '', spec['notes'], '',
             'ECE is independently recomputed using exhaustive 15-bin assignment. NLL uses a 1e-12 probability floor; Brier is the class-summed multiclass score.', '',
             '| Protocol | Runs | Accuracy mean ± fold SD | Macro-F1 mean ± fold SD | UAR mean ± fold SD |', '|---|---:|---:|---:|---:|']
    for name, report in status['protocols'].items():
        cells = []
        for key in ('accuracy', 'macro_f1', 'uar'):
            value = report['metrics'][key]
            cells.append(f'{value["mean"]:.4f}' + (f' ± {value["sd"]:.4f}' if value['sd'] is not None else ' (SD unavailable)'))
        lines.append(f'| {name} | {report["n_completed"]}/{report["n_expected"]} | ' + ' | '.join(cells) + ' |')
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.suite.resolve()), indent=2))
