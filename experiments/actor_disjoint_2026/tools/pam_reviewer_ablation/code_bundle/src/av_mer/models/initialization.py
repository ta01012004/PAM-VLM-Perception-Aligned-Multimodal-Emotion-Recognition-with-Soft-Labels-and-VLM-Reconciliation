"""Copy untrained canonical task tensors; never use a trained checkpoint."""
from __future__ import annotations

import gc
import hashlib
from pathlib import Path

import torch

from av_mer.models.config import PamVlmConfig
from av_mer.models.pam_vlm import PamVlmModel


def task_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            if '.backbone.' not in key}


def build_paired_model(model_config, training_config, seed):
    canonical_path = Path(training_config['canonical_model_config'])
    canonical_config = PamVlmConfig.from_json(canonical_path)
    torch.manual_seed(seed)
    canonical = PamVlmModel(canonical_config)
    initial = task_state(canonical)
    del canonical
    gc.collect()
    torch.manual_seed(seed)
    model = PamVlmModel(model_config)
    current = model.state_dict()
    copied, differing = {}, []
    for key, value in initial.items():
        if key in current and current[key].shape == value.shape:
            copied[key] = value
        else:
            differing.append(key)
    model.load_state_dict(copied, strict=False)
    hashes = {key: hashlib.sha256(value.numpy().tobytes()).hexdigest()
              for key, value in copied.items()}
    return model, {
        'seed': seed, 'canonical_model_config': str(canonical_path),
        'method': 'untrained canonical A0 task tensors; same-shape non-backbone only',
        'copied_tensor_hashes': hashes, 'differently_shaped_task_tensors': differing,
        'warning': 'Backbone changes alter capacity/pretraining and training stochasticity.',
    }
