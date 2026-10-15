"""Test the prespecified scientific contract before implementing preparation."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from variants import ARMS, variant
from prepare import validate_partition


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {'seed': 42, 'loss': {'label_smoothing': .05,
                    'soft_labels_enabled': True,
                    'hard_weights': {'fusion': .75, 'audio': .2, 'video': .05},
                    'soft_weights': {'fusion': .15, 'audio': .05, 'video': .02}}}
        self.model = {'fusion': {}, 'audio': {}, 'video': {}}

    def test_hard_controls_distinct(self):
        ce, _ = variant('A1', self.cfg, self.model)
        onehot, _ = variant('H0', self.cfg, self.model)
        self.assertEqual(ce['loss']['label_smoothing'], .05)
        self.assertEqual(onehot['loss']['label_smoothing'], 0.)
        self.assertEqual(ce['loss']['hard_weights'], self.cfg['loss']['hard_weights'])
        self.assertFalse(ce['loss']['soft_labels_enabled'])

    def test_aux_does_not_renormalize_fused(self):
        config, _ = variant('A6', self.cfg, self.model)
        self.assertEqual(config['loss']['hard_weights'], {'fusion': .75, 'audio': 0., 'video': 0.})
        self.assertEqual(config['loss']['soft_weights']['fusion'], .15)

    def test_all_arms_and_original_preserved(self):
        original = deepcopy(self.cfg)
        for arm in ARMS:
            variant(arm, self.cfg, self.model, audio_snapshot='/snapshot', resnet_weights='/weights')
        self.assertEqual(original, self.cfg)
        with self.assertRaises(ValueError):
            variant('A9', self.cfg, self.model)

    def test_actor_overlap_rejected(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as folder:
            paths = {}
            for split in ('train', 'val', 'test'):
                path = Path(folder) / (split + '.csv')
                pd.DataFrame({'clip_id': [split], 'actor_id': [1001], 'split': [split]}).to_csv(path, index=False)
                paths[split] = path
            with self.assertRaisesRegex(ValueError, 'actor overlap'):
                validate_partition(paths)


if __name__ == '__main__':
    unittest.main()
