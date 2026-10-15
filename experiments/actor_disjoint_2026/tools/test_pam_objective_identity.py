"""Check manuscript algebra against the unchanged original PAM loss.

These deterministic float64 checks are not training or performance evidence.
"""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F


SOURCE = (
    Path(__file__).parent / "pam_hard_ablation" / "code_bundle"
    / "src" / "av_mer" / "training" / "losses.py"
)
SPEC = importlib.util.spec_from_file_location("pam_original_loss_audit", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
WEIGHTS = {"fusion": (0.75, 0.15), "audio": (0.20, 0.05), "video": (0.05, 0.02)}
OUTPUTS = {"fusion": "logits", "audio": "audio_logits", "video": "video_logits"}


def test_original_loss_matches_mixed_target_value_and_gradient():
    generator = torch.Generator().manual_seed(7103)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 0])
    intended = 0.95 * F.one_hot(labels, 6).double() + 0.05 / 6
    logits = {
        key: torch.randn(7, 6, generator=generator, dtype=torch.float64,
                         requires_grad=True)
        for key in OUTPUTS.values()
    }
    votes = {}
    for branch in WEIGHTS:
        counts = torch.rand(7, 6, generator=generator, dtype=torch.float64)
        votes[branch] = counts / counts.sum(dim=-1, keepdim=True)
    config = MODULE.LossConfig.from_mapping({
        "hard_weights": {k: ab[0] for k, ab in WEIGHTS.items()},
        "soft_weights": {k: ab[1] for k, ab in WEIGHTS.items()},
        "soft_labels_enabled": True,
        "label_smoothing": 0.05,
    })
    actual = MODULE.MultiBranchClassificationLoss(config)(logits, labels, votes).total
    expected = actual.new_zeros(())
    for branch, (a, b) in WEIGHTS.items():
        target = (a * intended + b * votes[branch]) / (a + b)
        log_prob = F.log_softmax(logits[OUTPUTS[branch]], dim=-1)
        entropy = -(votes[branch] * votes[branch].log()).sum(-1).mean()
        expected = expected + (a + b) * -(target * log_prob).sum(-1).mean() - b * entropy
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    gradients = torch.autograd.grad(actual, tuple(logits.values()))
    for (branch, (a, b)), gradient in zip(WEIGHTS.items(), gradients):
        z = logits[OUTPUTS[branch]]
        target = (a * intended + b * votes[branch]) / (a + b)
        expected_gradient = (a + b) / labels.numel() * (z.softmax(-1) - target)
        torch.testing.assert_close(gradient, expected_gradient, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("branch", WEIGHTS)
def test_intended_margin_holds_at_all_vote_simplex_vertices(branch):
    a, b = WEIGHTS[branch]
    for intended_class in range(6):
        r = 0.95 * F.one_hot(torch.tensor(intended_class), 6).double() + 0.05 / 6
        targets = (a * r + b * torch.eye(6, dtype=torch.float64)) / (a + b)
        other = [c for c in range(6) if c != intended_class]
        margins = targets[:, intended_class, None] - targets[:, other]
        bound = (a * 0.95 - b) / (a + b)
        assert bound > 0
        assert margins.min().item() == pytest.approx(bound, abs=1e-12)
        assert torch.all(targets.argmax(-1) == intended_class)


def test_class_guarantee_is_not_unconditional_on_weights():
    r = torch.tensor([0.95 + 0.05 / 6] + [0.05 / 6] * 5)
    q = F.one_hot(torch.tensor(1), 6)
    target = (r + 2 * q) / 3
    assert target.argmax().item() != 0
