"""Reporting-flow test with mocked completed runs, NOT empirical results."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import yaml

from analyze import analyze


class AnalysisTests(unittest.TestCase):
    def test_hard_smoothing_contrast_reported_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'state').mkdir()
            (root / 'reports').mkdir()
            runs = []
            for arm in ('A0','A1','H0'):
                run = dict(id=arm, arm=arm, seed=42, config=str(root / (arm+'.yaml')))
                Path(run['config']).write_text(yaml.safe_dump({}))
                directory = root / arm
                directory.mkdir()
                (directory / 'resource_usage.json').write_text(json.dumps({
                    'training_seconds': 1, 'peak_cuda_bytes': 1,
                    'parameter_counts': {'total':1,'trainable':1}}))
                (directory / 'initialization.json').write_text(json.dumps({'copied_tensor_hashes':{'head':'hash'}}))
                (root / 'state' / (arm+'.done.json')).write_text(json.dumps({'run_dir':str(directory)}))
                runs.append(run)
            (root / 'suite.json').write_text(json.dumps({'runs':runs,'arms':{'A0':'full','A1':'ce','H0':'onehot'},
                'seeds':[42],'expected':3,'limitations':['fixture']}))
            def report(run, directory, output):
                value = {'A0':.7,'A1':.6,'H0':.5}[run['arm']]
                return {key:value for key in ['macro_f1','uar','accuracy','ece','nll','brier']}, None
            with patch.dict(sys.modules, {
                'aggregate':SimpleNamespace(report_run=report),
                'run_suite':SimpleNamespace(verify_inputs=lambda _:None,verify_checkpoint=lambda *a:None)}):
                result = analyze(root)
            self.assertTrue(result['complete'])
            contrasts = pd.read_csv(root / 'reports/additional_paired_contrasts.csv')
            self.assertEqual(contrasts.iloc[0]['contrast'],'H0_minus_A1_smoothing_only')
            self.assertAlmostEqual(contrasts.iloc[0]['macro_f1'],-.1)


if __name__ == '__main__':
    unittest.main()
