"""Full preparation fixture: placeholder bytes, never real experiments."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import yaml

import prepare
from variants import ARMS


class PreparationTests(unittest.TestCase):
    def fixture(self, root):
        base = root / 'baseline'
        base.mkdir()
        raw = root / 'raw'
        raw.mkdir()
        hashes = {}
        def frozen(path, text):
            path.write_text(text)
            hashes[str(path)] = prepare.digest(path)
            return str(path)
        data = {'raw_root': str(raw), 'decode_error_report': 'old.csv'}
        actor = 1001
        for split, count in [('train', 55), ('val', 18), ('test', 18)]:
            rows = []
            for _ in range(count):
                filename = f'{actor}.wav'
                (raw / filename).write_text('fixture')
                rows.append(dict(clip_id=str(actor), actor_id=actor, split=split,
                                 audio_path=filename, video_path=filename))
                actor += 1
            path = base / (split + '.csv')
            pd.DataFrame(rows).to_csv(path, index=False)
            hashes[str(path)] = prepare.digest(path)
            data[split + '_manifest'] = str(path)
        for name, text in [('loader_config', '{"sample_rate":16000}'),
                           ('label_mapping', '{}'), ('split_checksum', 'checksum')]:
            data[name] = frozen(base / name, text)
        data['soft_labels'] = {key: frozen(base / (key + '.parquet'), 'fixture')
                               for key in ('voice', 'face', 'av')}
        model = {'num_classes': 6, 'head_dropout': .25, 'fusion': {},
                 'audio': {'backend': 'transformers', 'encoder_id': 'microsoft/wavlm-base-plus'},
                 'video': {'backend': 'torchvision', 'backbone_id': 'efficientnet_b0'}}
        cfg = {'baseline': 'fusion', 'seed':42, 'data':data,
               'training':{'primary_metric':'macro_f1'}, 'published_protocol':{'training_seed':42},
               'model_config': frozen(base / 'model.json', json.dumps(model)),
               'loss': {'soft_labels_enabled':True, 'label_smoothing':.05,
                        'hard_weights':{'fusion':.75,'audio':.2,'video':.05},
                        'soft_weights':{'fusion':.15,'audio':.05,'video':.02}}}
        cfgpath = frozen(base / 'config.yaml', yaml.safe_dump(cfg))
        reference = {'protocol':'T602020','id':'t602020_seed0','fold':1,'config':cfgpath,
                     'test_manifest':data['test_manifest'], 'training_seed':42}
        frozen(base / 'suite.json', json.dumps({'runs':[reference]}))
        (base / 'input_hashes.json').write_text(json.dumps(hashes))
        def snapshot(repo, kind):
            path = root / repo / 'snapshots' / ('a' * 40)
            path.mkdir(parents=True)
            (path / 'config.json').write_text(json.dumps({'model_type':kind,'hidden_size':768,
                                                        'feat_extract_norm':'group'}))
            (path / 'pytorch_model.bin').write_text('fixture')
            (path / 'preprocessor_config.json').write_text(json.dumps({'sampling_rate':16000,
                                                                      'do_normalize':True}))
            return path
        original = snapshot('models--microsoft--wavlm-base-plus','wavlm')
        alternative = snapshot('models--facebook--wav2vec2-base','wav2vec2')
        cache = root / 'torch'
        (cache / 'hub/checkpoints').mkdir(parents=True)
        efficient = cache / 'hub/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth'
        efficient.write_text('fixture')
        resnet = root / 'resnet18-f37072fd.pth'
        resnet.write_text('fixture')
        return argparse.Namespace(baseline=base, output=root/'new', original_audio_snapshot=original,
                                  audio_snapshot=alternative, resnet_weights=resnet,
                                  torch_cache=cache, arms=list(ARMS), seeds=[42,43,44])

    def test_full_suite_is_fresh_preserves_source_and_seed_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.fixture(Path(folder))
            digest = prepare.digest
            def fixture_digest(path):
                name = Path(path).name
                if name.startswith('resnet18-'): return 'f37072fd' + '0'*56
                if name.startswith('efficientnet_b0_'): return '7f5810bc' + '0'*56
                return digest(path)
            before = digest(args.baseline / 'suite.json')
            with patch('prepare.digest', side_effect=fixture_digest):
                prepare.prepare(args)
            spec = json.loads((args.output / 'suite.json').read_text())
            self.assertEqual(len(spec['runs']),33)
            self.assertFalse(spec['historical_results_reused'])
            self.assertEqual(digest(args.baseline / 'suite.json'), before)
            for run in spec['runs']:
                cfg = yaml.safe_load(Path(run['config']).read_text())
                self.assertEqual(cfg['seed'], run['training_seed'])
                self.assertEqual(cfg['published_protocol']['training_seed'],cfg['seed'])
                self.assertNotEqual(cfg['output_root'], str(args.baseline))
            with self.assertRaises(FileExistsError):
                prepare.prepare(args)

    def test_wrong_hf_repository_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'models--microsoft--wavlm-base' / 'snapshots' / ('a'*40)
            path.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'repository cache'):
                prepare.snapshot(path, 'wavlm')


if __name__ == '__main__':
    unittest.main()
