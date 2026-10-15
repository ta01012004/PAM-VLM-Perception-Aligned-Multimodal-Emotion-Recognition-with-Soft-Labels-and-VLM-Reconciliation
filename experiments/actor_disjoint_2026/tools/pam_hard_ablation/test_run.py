"""CPU-only runner guards and package provenance tests."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from run import check_packages, main, PackageNotFoundError


class RunnerTests(unittest.TestCase):
    def test_missing_optional_package_matches_record(self):
        with patch('run.version', side_effect=PackageNotFoundError):
            check_packages({'optional': 'not-installed'})

    def test_package_mismatch_fails(self):
        with patch('run.version', return_value='2.0'):
            with self.assertRaisesRegex(RuntimeError, 'Environment mismatch'):
                check_packages({'torch': '1.0'})

    def test_matching_package(self):
        with patch('run.version', return_value='1.0'):
            check_packages({'torch': '1.0'})

    def test_no_training_outside_slurm(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'only inside SLURM'):
                main(Path('/unused'))


if __name__ == '__main__':
    unittest.main()
