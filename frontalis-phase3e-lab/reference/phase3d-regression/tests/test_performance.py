import importlib.util
from pathlib import Path
import sys
import unittest
import tempfile
from unittest import mock
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('performance', ROOT / 'experiments' / 'performance.py')
performance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(performance)
from npnd.core import Personality

class PerformanceTests(unittest.TestCase):
    def test_rerun_paths_preserve_original_evidence_and_stay_in_reports(self):
        report, runtime = performance.output_paths('performance-rerun-1')
        self.assertEqual(report.parent, ROOT / 'reports')
        self.assertNotEqual(report, ROOT / 'reports' / 'performance.json')
        self.assertNotEqual(runtime, ROOT / 'reports' / 'runtime.json')
        for tag in ('../overwrite', '', 'C:\\report', 'x/y'):
            with self.assertRaises(ValueError):
                performance.output_paths(tag)

    def test_timing_units_and_percentiles(self):
        result = performance.timing_summary([.001, .003, .005])
        self.assertEqual(result['samples'], 3)
        self.assertAlmostEqual(result['medianMs'], 3.)
        self.assertAlmostEqual(result['p95Ms'], 4.8)

    def test_timing_rejects_invalid_samples(self):
        for values in ([], [-.1], [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError):
                performance.timing_summary(values)

    def test_cuda_timer_synchronizes_around_every_call(self):
        operations = []
        with mock.patch.object(performance, 'synchronize', side_effect=lambda d: operations.append('sync')):
            performance.measured(lambda: operations.append('call'), 'cuda', repeats=2, warmups=1)
        self.assertEqual(operations, ['call', 'sync', 'sync', 'call', 'sync', 'sync', 'call', 'sync'])

    def test_memory_array_counts_not_rss(self):
        model = Personality(width=64)
        result = performance.state_bytes(model)
        self.assertEqual(result['learnedTensorBytes'], 2062 * 8)
        self.assertEqual(result['fixedTensorBytes'], 56 * 8)
        self.assertEqual(result['memoryNumericPayloadBytes'], 0)

    def test_gpu_whole_device_memory_does_not_claim_model_memory(self):
        completed = mock.Mock(returncode=0, stdout='GPU, 500, 8192, 3\n')
        with mock.patch.object(performance.shutil, 'which', return_value='nvidia-smi'), mock.patch.object(performance.subprocess, 'run', return_value=completed):
            result = performance.nvidia_snapshot()
        self.assertIn('including other applications', result['scope'])
        self.assertEqual(len(result['rows']), 1)

    def test_missing_nvidia_tool_is_explicit(self):
        with mock.patch.object(performance.shutil, 'which', return_value=None):
            self.assertFalse(performance.nvidia_snapshot()['available'])

    def test_preserved_runtime_worker_does_not_need_psutil(self):
        with mock.patch.object(performance, 'psutil', None):
            self.assertFalse(performance.process_memory()['available'])
        # Output-only workers never call process memory. This exercises the
        # same CLI payload shape used by the two-runtime comparison.
        with mock.patch.object(performance, 'process_memory', side_effect=AssertionError('memory report used in worker')):
            result = performance.cpu_worker(16)
        self.assertEqual(np.asarray(result['outputs']).shape, (96, 7))
        self.assertTrue(np.isfinite(result['outputs']).all())

    def test_unavailable_cuda_does_not_create_portability_evidence(self):
        with mock.patch.object(performance.torch.cuda, 'is_available', return_value=False):
            result = performance.cpu_gpu_portability()
        self.assertFalse(result['available'])
        self.assertEqual(result['rows'], [])

    def test_float32_and_float64_parameter_storage_are_reported_separately(self):
        model = Personality(width=64, dtype='float32')
        result = performance.state_bytes(model)
        self.assertEqual(result['learnedTensorBytes'], 2062 * 4)
        self.assertEqual(result['fixedTensorBytes'], 56 * 4)

    def test_million_benchmark_is_gated(self):
        with self.assertRaises(ValueError):
            performance.measure_model(width=31250)

    def test_million_extension_requires_fixed_protocol_and_boolean_pass(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'runtime', prefix='performance-gate-test-') as folder:
            root = Path(folder)
            (root / 'reports').mkdir(); (root / 'docs').mkdir()
            gate = {'millionParameterGatePassed': True, 'finiteAndUnrelatedBranchPreserved': True, 'finalTestInspectedAtGate': False}
            path = root / 'reports' / 'capacity-gate.json'
            path.write_text(performance.json.dumps(gate), encoding='utf8')
            with self.assertRaises(ValueError):
                performance.require_million_gate(root)
            (root / 'docs' / '01b-million-protocol.md').write_text('Fixed extension conditions', encoding='utf8')
            self.assertEqual(performance.require_million_gate(root)['gate'], gate)
            for value in (False, 'true', 1):
                path.write_text(performance.json.dumps({**gate, 'millionParameterGatePassed': value}), encoding='utf8')
                with self.assertRaises(ValueError):
                    performance.require_million_gate(root)

    def test_million_gate_failure_prevents_model_allocation(self):
        with mock.patch.object(performance, 'require_million_gate', side_effect=ValueError('not passed')), mock.patch.object(performance, 'Personality') as constructor:
            with self.assertRaises(ValueError):
                performance.measure_model(width=31250, allow_million=True)
            constructor.assert_not_called()

    def test_recorded_run_contains_actual_devices_and_finite_metrics(self):
        report = ROOT / 'reports' / 'performance.json'
        if not report.exists():
            self.skipTest('Run experiments/performance.py first for artifact validation.')
        result = performance.json.loads(report.read_text(encoding='utf8'))
        self.assertEqual(len(result['models']), 15 if result['runtime']['cudaAvailable'] else 7)
        for row in result['models']:
            expected_device = 'cuda:0' if row['device'] == 'cuda' else 'cpu'
            self.assertEqual(row['actualTensorDevices'], [expected_device])
            self.assertLessEqual(row['learnedParameterCount'], 100014)
            for key in ('singleInference', 'batch96Inference', 'oneExperienceUpdate'):
                self.assertTrue(np.isfinite(row[key]['medianMs']))
                self.assertGreaterEqual(row[key]['medianMs'], 0.)
                self.assertGreaterEqual(row[key]['p95Ms'], row[key]['medianMs'])
        for row in result['portability']['rows']:
            self.assertTrue(row['gpuSnapshotLoadedOnCpu64']['parameterHashesEqual'])
            self.assertLess(row['gpuSnapshotLoadedOnCpu64']['maxAbsOutputDifference'], 1e-10)
            self.assertLess(row['cpu64VsGpu64']['maxAbsOutputDifference'], 1e-7)

    def test_million_record_preserves_original_evidence_and_saved_state(self):
        report = ROOT / 'reports' / 'million-performance.json'
        if not report.exists():
            self.skipTest('Run the gated million extension first for artifact validation.')
        result = performance.json.loads(report.read_text(encoding='utf8'))
        self.assertTrue(result['originalEvidenceUnchanged'])
        self.assertEqual(result['originalEvidenceHashesBefore'], result['originalEvidenceHashesAfter'])
        for name, digest in result['originalEvidenceHashesBefore'].items():
            self.assertEqual(performance.hashlib.sha256((ROOT / 'reports' / name).read_bytes()).hexdigest(), digest)
        for row in result['models']:
            self.assertEqual(row['learnedParameterCount'], 1000014)
            self.assertEqual(row['updatedParameterCountPerExperience'], 500007)
            self.assertEqual(row['actualTensorDevices'], ['cpu' if row['device'] == 'cpu' else 'cuda:0'])
            self.assertTrue(np.isfinite(row['singleInference']['medianMs']))
        for row in result['portability']['rows']:
            self.assertTrue(row['gpuSnapshotLoadedOnCpu64']['parameterHashesEqual'])
            self.assertLess(row['gpuSnapshotLoadedOnCpu64']['maxAbsOutputDifference'], 1e-10)

if __name__ == '__main__':
    unittest.main()
