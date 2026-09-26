import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseToolsTests(unittest.TestCase):
    def test_preparation_defaults_and_explicit_paths(self):
        module = load_script('prepare_benchmarks')
        self.assertEqual(module.parse_args([]).snapshot, 'natural')
        args = module.parse_args(['--snapshot', 'serving', '--output', 'custom-data'])
        self.assertEqual(args.output, Path('custom-data'))
        self.assertEqual(args.snapshot, 'serving')
        self.assertEqual(len(module.DATASETS), 4)

    def test_source_fingerprint_rejects_drift(self):
        module = load_script('check_repository')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'code/natural/sample.py'
            source.parent.mkdir(parents=True)
            source.write_text('value = 1\n')
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            (root / 'SOURCE_MANIFEST.sha256').write_text(f'{digest}  code/natural/sample.py\n')
            self.assertEqual(module.verify_manifest(root), 1)
            source.write_text('value = 2\n')
            with self.assertRaises(ValueError):
                module.verify_manifest(root)

    def test_manifest_rejects_path_escape(self):
        module = load_script('check_repository')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'SOURCE_MANIFEST.sha256').write_text('invalid  ../outside.py\n')
            with self.assertRaises(ValueError):
                module.verify_manifest(root)

    def test_snapshot_sources_have_not_changed(self):
        report = load_script('check_repository').check_repository()
        self.assertTrue(report['passed'])
        self.assertGreater(report['frozen_source_files_verified'], 100)
        self.assertFalse(report['gpu_inference_run'])


if __name__ == '__main__':
    unittest.main()
