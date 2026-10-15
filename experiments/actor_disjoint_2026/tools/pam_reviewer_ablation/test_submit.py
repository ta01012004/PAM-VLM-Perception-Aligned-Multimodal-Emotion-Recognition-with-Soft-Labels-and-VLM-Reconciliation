"""Mock SLURM to check resource arguments/dependencies; no real submission."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from submit import resources, job_id
import submit as driver


class SubmitTests(unittest.TestCase):
    def mock_launch(self, root, dry_run=False, reject=False, serial=False):
        suite = root / 'suite'
        suite.mkdir()
        (suite / 'suite.json').write_text(json.dumps({'runs': [{}, {}, {}]}))
        conda = root / 'conda.sh'
        conda.write_text('# fixture')
        args = SimpleNamespace(suite=suite, kind='reviewer', conda_init=conda,
                    conda_env='ta', partition='gpu', report_partition='cpu',
                    gpu_request='gpu:1', cpus=16, memory='64G', time='12:00:00',
                    account=None, qos=None, parallel=1, dry_run=dry_run, repo=None,
                    serial=serial)
        if serial:
            args.report_partition = None
        calls = []
        jobs = []
        def checked(cmd, cwd, env):
            calls.append(cmd)
            if cmd[0] == 'sbatch':
                self.assertEqual(env['SBATCH_DEPENDENCY'], 'afterok:999')
            if cmd[0] == 'sbatch' and '--test-only' in cmd:
                if reject and 'analyze' in cmd:
                    raise RuntimeError('mock incompatible report partition')
                return 'test-only accepted'
            if cmd[0] == 'sbatch':
                jobs.append(str(12000 + len(jobs)))
                return jobs[-1]
            return 'mock queue/policy'
        # Replace module's os object, not global os.name (pathlib depends on it).
        posix = SimpleNamespace(name='posix', environ={'SBATCH_DEPENDENCY': 'afterok:999'}, open=os.open, fdopen=os.fdopen,
                                getpid=os.getpid, O_CREAT=os.O_CREAT, O_EXCL=os.O_EXCL,
                                O_WRONLY=os.O_WRONLY)
        with patch.object(driver, 'os', posix), patch.object(driver, 'checked', side_effect=checked), \
             patch.dict(sys.modules, {'run_suite': SimpleNamespace(verify_inputs=lambda _: None)}):
            if reject:
                with self.assertRaisesRegex(RuntimeError, 'incompatible'):
                    driver.submit(args)
            else:
                driver.submit(args)
        return suite, calls, jobs

    def test_actual_driver_dependency_chain_mocked(self):
        with tempfile.TemporaryDirectory() as folder:
            suite, calls, jobs = self.mock_launch(Path(folder))
            submitted = [c for c in calls if c[0] == 'sbatch' and '--parsable' in c]
            self.assertEqual(len(submitted), 3)
            self.assertIn('--dependency=afterok:12000', submitted[1])
            self.assertIn('--dependency=afterok:12001', submitted[2])
            self.assertIn('--array=0-2%1', submitted[1])
            self.assertIn('--partition=cpu', submitted[2])
            self.assertEqual(json.loads((suite / 'submission.json').read_text())['jobs'], jobs)
            self.assertTrue((suite / '.submission.lock').exists())

    def test_preflight_rejection_queues_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            suite, calls, jobs = self.mock_launch(Path(folder), reject=True)
            self.assertEqual(jobs, [])
            self.assertFalse((suite / '.submission.lock').exists())

    def test_dry_run_queues_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            suite, calls, jobs = self.mock_launch(Path(folder), dry_run=True)
            self.assertEqual(jobs, [])
            self.assertFalse((suite / 'submission.json').exists())

    def test_serial_uses_one_job_and_records_external_dependency(self):
        with tempfile.TemporaryDirectory() as folder:
            suite, calls, jobs = self.mock_launch(Path(folder), serial=True)
            submitted = [c for c in calls if c[0] == 'sbatch' and '--parsable' in c]
            self.assertEqual(len(submitted), 1)
            self.assertFalse(any(s.startswith('--array=') for s in submitted[0]))
            self.assertTrue(any(s.endswith('submit_serial.sbatch') for s in submitted[0]))
            self.assertIn('--dependency=afterok:999', submitted[0])
            validation = [c for c in calls if c[0] == 'sbatch' and '--test-only' in c]
            self.assertIn('--dependency=afterok:999', validation[0])
            record = json.loads((suite / 'submission.json').read_text())
            self.assertEqual(record['mode'], 'serial')
            self.assertEqual(record['external_dependency'], 'afterok:999')
            self.assertEqual(len(record['jobs']), 1)

    def test_serial_dry_run_queues_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            suite, calls, jobs = self.mock_launch(Path(folder), serial=True, dry_run=True)
            self.assertEqual(jobs, [])
            self.assertFalse((suite / '.submission.lock').exists())

    def test_invalid_job_reply_rejected(self):
        self.assertEqual(job_id('12345;cluster\n'), '12345')
        with self.assertRaises(ValueError):
            job_id('warning: submission failed')

    def test_explicit_partition_and_limits(self):
        args = SimpleNamespace(partition='gpu', gpu_request='gpu:1', cpus=16,
                               memory='64G', time='12:00:00', account=None, qos=None)
        command = resources(args, gpu=True)
        self.assertIn('--partition=gpu', command)
        self.assertIn('--gres=gpu:1', command)
        self.assertNotIn('--gres=gpu:1', resources(args, gpu=False))

    def test_resources_fail_closed(self):
        args = SimpleNamespace(partition='', gpu_request='gpu:1', cpus=16,
                               memory='64G', time='12:00:00', account=None, qos=None)
        with self.assertRaises(ValueError):
            resources(args, gpu=True)


if __name__ == '__main__':
    unittest.main()
