"""Independent metrics retained from the completed reviewer-suite audit.

ECE: 15 equal-width bins, top-1 confidence, final bin includes confidence=1.
NLL: natural logarithm, minimum probability 1e-12. Brier: sum over six classes.
"""
import numpy as np
import pandas as pd

LABELS = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad')
METRICS = ('macro_f1', 'uar', 'accuracy', 'ece', 'nll', 'brier')

def calculate_metrics(frame: pd.DataFrame) -> dict:
    """Recompute six metrics and confusion without importing the source runner."""
    p = frame[['p_' + label for label in LABELS]].to_numpy(dtype=float)
    mapping = dict(zip(LABELS, range(6)))
    y = frame['true'].map(mapping).to_numpy()
    if not len(frame) or np.isnan(y).any() or not np.isfinite(p).all():
        raise ValueError('Empty, nonfinite or unknown-label predictions')
    if (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-6, rtol=0):
        raise ValueError('Invalid probability distribution')
    y = y.astype(int)
    predicted = p.argmax(1)
    if not np.array_equal(np.asarray(LABELS)[predicted], frame['pred'].to_numpy()):
        raise ValueError('Saved prediction is not probability argmax')
    confusion = np.zeros((6, 6), dtype=int)
    np.add.at(confusion, (y, predicted), 1)
    tp = np.diag(confusion)
    support = confusion.sum(1)
    prediction_count = confusion.sum(0)
    f1 = np.divide(2 * tp, support + prediction_count,
                   out=np.zeros(6, dtype=float), where=(support + prediction_count) != 0)
    recall = np.divide(tp, support, out=np.zeros(6, dtype=float), where=support != 0)
    confidence = p.max(1)
    correct = predicted == y
    bins = []
    boundaries = np.linspace(0, 1, 16)
    for i in range(15):
        selected = (confidence >= boundaries[i]) & (
            (confidence <= boundaries[i + 1]) if i == 14 else (confidence < boundaries[i + 1]))
        bins.append({'count': int(selected.sum()),
                     'confidence': float(confidence[selected].mean()) if selected.any() else 0.,
                     'accuracy': float(correct[selected].mean()) if selected.any() else 0.})
    if sum(b['count'] for b in bins) != len(frame):
        raise ValueError('ECE bin partition omitted samples')
    return {'macro_f1': float(f1.mean()), 'uar': float(recall.mean()),
            'accuracy': float(correct.mean()),
            'ece': float(sum(b['count'] * abs(b['confidence'] - b['accuracy'])
                             for b in bins) / len(frame)),
            'nll': float(-np.log(np.maximum(p[np.arange(len(frame)), y], 1e-12)).mean()),
            'brier': float(np.square(p - np.eye(6)[y]).sum(1).mean()),
            'confusion_matrix': confusion.tolist(), 'num_samples': len(frame), 'bins': bins}


def validate_actor_partitions(actors: dict) -> None:
    for part in ('train', 'val', 'test'):
        if len(set(actors[part])) != len(actors[part]):
            raise ValueError('Duplicate actors in partition')
    if any(set(actors[a]) & set(actors[b])
           for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
        raise ValueError('Actor leakage between partitions')
