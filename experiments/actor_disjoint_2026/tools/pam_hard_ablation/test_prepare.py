"""CPU-only regression tests: no scheduler, GPU or real experiment touched."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import yaml
from prepare import digest, make_hard, prepare


class PreparationTests(unittest.TestCase):
    def config(self):
        return {'baseline': 'fusion', 'seed': 42, 'run_name': 'soft',
                'output_root': 'old', 'model_config': '/original/model.json',
                'training': {'epochs': 40, 'early_stopping_patience': 8},
                'optimizer': {'learning_rate': 0.0002},
                'data': {'decode_error_report': 'old'},
                'loss': {'soft_labels_enabled': True, 'label_smoothing': 0.05,
                         'hard_weights': {'fusion': 0.75, 'audio': 0.2, 'video': 0.05},
                         'soft_weights': {'fusion': 0.15, 'audio': 0.05, 'video': 0.02}}}

    def test_only_kl_and_output_change(self):
        soft = self.config()
        original = deepcopy(soft)
        hard = make_hard(soft, Path('/new'), 'm10_fold01')
        self.assertEqual(soft, original)
        for key in ('baseline', 'seed', 'model_config', 'training', 'optimizer'):
            self.assertEqual(hard[key], soft[key])
        self.assertEqual(hard['loss']['hard_weights'], soft['loss']['hard_weights'])
        self.assertEqual(hard['loss']['label_smoothing'], 0.05)
        self.assertFalse(hard['loss']['soft_labels_enabled'])
        self.assertTrue(all(v == 0 for v in hard['loss']['soft_weights'].values()))

    def fixture(self, root):
        repo, baseline, code = root / 'repo', root / 'baseline', root / 'snapshot'
        baseline.mkdir()
        hashes = {}
        for relative in ['src/core.py', 'scripts/train.py', 'scripts/eval_test.py',
                         'scripts/plot_results.py', 'scripts/published16/aggregate.py',
                         'scripts/published16/protocols.py', 'scripts/published16/run_suite.py']:
            p = code / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('# original fixture\n')
            hashes[str(repo / relative)] = digest(p)
        runs = []
        for fold in range(1, 11):
            identifier = f'm10_fold{fold:02d}'
            directory = baseline / 'runs' / identifier
            directory.mkdir(parents=True)
            config = self.config()
            cfg_path = baseline / 'configs' / (identifier + '.yaml')
            cfg_path.parent.mkdir(exist_ok=True)
            cfg_path.write_text(yaml.safe_dump(config))
            hashes[str(cfg_path)] = digest(cfg_path)
            resolved = {**config, 'run_id': identifier, 'slurm_job_id': '123'}
            for filename in ['test_predictions.csv', 'test_metrics.json', 'history.json',
                             'metrics.json', 'reproducibility.json', 'checkpoints/best.pt']:
                p = directory / filename
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('{}')
            (directory / 'resolved_config.json').write_text(json.dumps(resolved))
            marker = baseline / 'state' / (identifier + '.done.json')
            marker.parent.mkdir(exist_ok=True)
            marker.write_text(json.dumps({'run_dir': str(directory)}))
            runs.append({'protocol': 'M10', 'id': identifier, 'fold': fold,
                         'config': str(cfg_path), 'output_root': str(directory)})
        (baseline / 'suite.json').write_text(json.dumps({'runs': runs}))
        hashes[str(baseline / 'suite.json')] = digest(baseline / 'suite.json')
        (baseline / 'input_hashes.json').write_text(json.dumps(hashes))
        return repo, baseline, code

    def test_prepares_ten_and_preserves_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, baseline, code = self.fixture(root)
            before = digest(baseline / 'suite.json')
            output = root / 'new'
            prepare(repo, baseline, output, code)
            self.assertEqual(digest(baseline / 'suite.json'), before)
            self.assertEqual(len(json.loads((output / 'suite.json').read_text())['runs']), 10)
            for name, expected in json.loads((output / 'input_hashes.json').read_text()).items():
                self.assertEqual(digest(Path(name)), expected)
            with self.assertRaises(FileExistsError):
                prepare(repo, baseline, output, code)

    def test_source_mismatch_refused_before_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, baseline, code = self.fixture(root)
            (code / 'src/core.py').write_text('# altered\n')
            with self.assertRaisesRegex(ValueError, 'source mismatch'):
                prepare(repo, baseline, root / 'new', code)
            self.assertFalse((root / 'new').exists())

    def test_incomplete_soft_fold_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, baseline, code = self.fixture(root)
            (baseline / 'runs/m10_fold03/test_predictions.csv').unlink()
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                prepare(repo, baseline, root / 'new', code)
            self.assertFalse((root / 'new').exists())


if __name__ == '__main__':
    unittest.main()
