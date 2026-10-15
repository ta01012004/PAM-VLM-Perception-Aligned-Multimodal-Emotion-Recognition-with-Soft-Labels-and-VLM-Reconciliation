"""Prespecified one-factor controls; no result-dependent arm selection."""
from copy import deepcopy

ARMS = {
    'A0': 'Full PAM CE+KL (fresh reference)',
    'A1': 'CE-only; retain original smoothing (isolates KL)',
    'A2': 'All KL branches use AV target',
    'A3': 'Swap voice/face targets; fused retains AV',
    'A4': 'Bypass cross-attention; retain modality norms/gate',
    'A5': 'Equal 0.5 gate; retain cross-attention/output norm',
    'A6': 'Remove auxiliary CE+KL; fused coefficients unchanged',
    'A7': 'Replace audio encoder with pinned wav2vec2-base',
    'A8': 'Replace visual encoder with ResNet-18 IMAGENET1K_V1',
    'A9': 'Replace both encoders (portability/interaction)',
    'H0': 'Literal one-hot CE-only, smoothing=0 (two changes vs A0)',
}


def variant(arm, config, model, *, audio_snapshot=None, resnet_weights=None):
    if arm not in ARMS:
        raise ValueError(f'Unknown arm: {arm}')
    config, model = deepcopy(config), deepcopy(model)
    config['target_routing'] = 'matched'
    model['fusion'].update(cross_attention_enabled=True, gate_mode='learned')
    if arm in {'A1', 'H0'}:
        config['loss']['soft_labels_enabled'] = False
        config['loss']['soft_weights'] = {key: 0.0 for key in config['loss']['soft_weights']}
    if arm == 'H0':
        config['loss']['label_smoothing'] = 0.0
    if arm == 'A2':
        config['target_routing'] = 'shared_av'
    if arm == 'A3':
        config['target_routing'] = 'swapped'
    if arm == 'A4':
        model['fusion']['cross_attention_enabled'] = False
    if arm == 'A5':
        model['fusion']['gate_mode'] = 'equal'
    if arm == 'A6':
        for kind in ('hard_weights', 'soft_weights'):
            for branch in ('audio', 'video'):
                config['loss'][kind][branch] = 0.0
    if arm in {'A7', 'A9'}:
        if not audio_snapshot:
            raise ValueError('A7/A9 require pinned offline audio snapshot')
        model['audio'].update(encoder_id=str(audio_snapshot), backbone_dim=768,
                              normalize_waveform=True, pass_attention_mask=False)
    if arm in {'A8', 'A9'}:
        if not resnet_weights:
            raise ValueError('A8/A9 require pinned ResNet weights')
        model['video'].update(backbone_id='resnet18', backbone_dim=512,
                              weights_path=str(resnet_weights), pretrained=True,
                              normalize_frames=True)
    return config, model
