"""CPU-only tests; no scheduler, corpus, download or training job."""
from dataclasses import replace
from pathlib import Path
import json
import tempfile
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent / 'code_bundle/src'))
from av_mer.models.config import AudioEncoderConfig, VideoEncoderConfig, FusionConfig
from av_mer.models.audio_encoder import AudioEncoder
from av_mer.models.video_encoder import VideoEncoder
from av_mer.models.fusion import BidirectionalCrossAttentionFusion
from av_mer.training.loop import _soft_targets
from av_mer.models.config import PamVlmConfig
from av_mer.models.initialization import build_paired_model


class VariantTests(unittest.TestCase):
    def tiny_config(self):
        return {'num_classes': 6, 'head_dropout': .1,
                'audio': dict(encoder_id='tiny', backend='tiny', backbone_dim=8,
                              hidden_dim=8, freeze_backbone=True, dropout=0., tiny_stride=2),
                'video': dict(backbone_id='tiny', backend='tiny', backbone_dim=8,
                              hidden_dim=8, temporal_layers=1, num_heads=2, dropout=0.,
                              freeze_backbone=True, pretrained=False),
                'fusion': dict(hidden_dim=8, num_heads=2, dropout=0.)}

    def test_paired_full_models_and_backbone_dimension_change(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'canonical.json'
            original = self.tiny_config()
            path.write_text(json.dumps(original))
            _, reference = build_paired_model(PamVlmConfig.from_json(path),
                                              {'canonical_model_config': str(path)}, 42)
            for changes in ({'cross_attention_enabled': False}, {'gate_mode': 'equal'}, {}):
                modified = self.tiny_config()
                modified['fusion'].update(changes)
                if not changes:
                    modified['video']['backbone_dim'] = 12
                candidate = Path(folder) / 'candidate.json'
                candidate.write_text(json.dumps(modified))
                model, record = build_paired_model(PamVlmConfig.from_json(candidate),
                                                   {'canonical_model_config': str(path)}, 42)
                for key, value in record['copied_tensor_hashes'].items():
                    self.assertEqual(value, reference['copied_tensor_hashes'][key], key)
                if not changes:
                    self.assertIn('video_encoder.projector.0.weight', record['differently_shaped_task_tensors'])
                output = model(torch.randn(2, 16), torch.ones(2, 16, dtype=torch.bool),
                               torch.rand(2, 3, 3, 8, 8), torch.ones(2, 3, dtype=torch.bool))
                self.assertEqual(output['logits'].shape, (2, 6))

    def test_resnet_normalization_preserves_padding(self):
        class FakeFrames(nn.Module):
            def __init__(self):
                super().__init__()
                self.last = None
            def forward(self, frames):
                self.last = frames
                return frames.new_zeros(len(frames), 8)
        config = VideoEncoderConfig('fake', 'torchvision', 8, 8, 1, 2, 0., True,
                                    True, normalize_frames=True)
        fake = FakeFrames()
        model = VideoEncoder(config, fake)
        mean = torch.tensor([.485, .456, .406])[None, None, :, None, None]
        frames = mean.expand(1, 2, 3, 4, 4).clone()
        model(frames, torch.tensor([[True, False]]))
        self.assertTrue(torch.equal(fake.last, torch.zeros_like(fake.last)))

    def test_target_routes(self):
        batch = dict(soft_voice=torch.tensor([1]), soft_face=torch.tensor([2]),
                     soft_av=torch.tensor([3]))
        self.assertIs(_soft_targets(batch, 'matched')['audio'], batch['soft_voice'])
        self.assertIs(_soft_targets(batch, 'shared_av')['video'], batch['soft_av'])
        self.assertIs(_soft_targets(batch, 'swapped')['audio'], batch['soft_face'])
        self.assertIs(_soft_targets(batch, 'swapped')['video'], batch['soft_voice'])
        with self.assertRaises(ValueError):
            _soft_targets(batch, 'typo')

    def test_structural_initialization_and_gradients(self):
        config = FusionConfig(8, 2, 0.0)
        models = []
        for overrides in ({}, {'cross_attention_enabled': False}, {'gate_mode': 'equal'}):
            torch.manual_seed(42)
            models.append(BidirectionalCrossAttentionFusion(replace(config, **overrides)))
        for model in models[1:]:
            for key, value in models[0].state_dict().items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
        a, v = torch.randn(2, 4, 8), torch.randn(2, 3, 8)
        am, vm = torch.ones(2, 4, dtype=torch.bool), torch.ones(2, 3, dtype=torch.bool)
        for model in models:
            model(a, am, v, vm).square().sum().backward()
        self.assertIsNone(models[1].audio_queries_video.in_proj_weight.grad)
        self.assertFalse(models[1].audio_queries_video.in_proj_weight.requires_grad)
        self.assertIsNone(models[2].gate.weight.grad)
        self.assertFalse(models[2].gate.weight.requires_grad)
        self.assertIsNotNone(models[0].gate.weight.grad)

    def test_resnet_stage_policy(self):
        class FakeResnet(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Linear(8, 8)
                for i in range(1, 5):
                    setattr(self, f'layer{i}', nn.Linear(8, 8))
        config = VideoEncoderConfig('resnet18', 'torchvision', 8, 8, 1, 2, 0.,
                                    False, True, 2)
        model = VideoEncoder(config, FakeResnet())
        self.assertFalse(model.backbone.layer2.weight.requires_grad)
        self.assertTrue(model.backbone.layer3.weight.requires_grad)
        model.train()
        self.assertFalse(model.backbone.layer2.training)
        self.assertTrue(model.backbone.layer3.training)

    def test_audio_normalization_and_mask_policy(self):
        class FakeSpeech(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(1))
                self.kwargs = None
            def forward(self, **kwargs):
                self.kwargs = kwargs
                return type('Output', (), {'last_hidden_state':
                    kwargs['input_values'].unsqueeze(-1).expand(-1, -1, 4)})()
            def _get_feature_vector_attention_mask(self, n, mask):
                return mask.bool()
        config = AudioEncoderConfig('test', 'transformers', 4, 4, True, 0., 2,
                                     normalize_waveform=True, pass_attention_mask=False)
        fake = FakeSpeech()
        model = AudioEncoder(config, fake)
        model(torch.tensor([[1., 3., 0., 0.]]), torch.tensor([[True, True, False, False]]))
        self.assertNotIn('attention_mask', fake.kwargs)
        self.assertTrue(torch.allclose(fake.kwargs['input_values'],
                                        torch.tensor([[-1., 1., 0., 0.]]), atol=1e-6))


if __name__ == '__main__':
    unittest.main()
