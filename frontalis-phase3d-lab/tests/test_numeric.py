"""Data separation, gate provenance, and ordering checks for real experiments."""
import itertools
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import numeric
from experiments import gpu_diagnostic
from npnd import data
numeric.torch.set_num_threads(1)


class NumericTests(unittest.TestCase):
    def test_disjoint_training_validation_test_sources_and_coordinates(self):
        blocks = [data.frozen_inputs(n, phase) for n, phase in [(192, 'train'), (64, 'validation'), (256, 'test')]]
        ids = [{row['id'] for row in block} for block in blocks]
        coordinates = [{(row['space'], tuple(row['cue'])) for row in block} for block in blocks]
        for a, b in itertools.combinations(range(3), 2):
            self.assertFalse(ids[a] & ids[b])
            self.assertFalse(coordinates[a] & coordinates[b])

    def test_personas_receive_identical_cues_and_different_observed_facts(self):
        records = [data.training(4311, person) for person in range(4)]
        cues = [[(row['space'], row['cue']) for row in block] for block in records]
        self.assertTrue(all(cues[0] == cue for cue in cues[1:]))
        self.assertEqual(len({numeric.digest([row['observed'] for row in block]) for block in records}), 4)
        for row in records[0]:
            self.assertFalse({'seriesId', 'emotion', 'target', 'teacherCoefficients'} & row.keys())

    def test_stress_changes_facts_not_cues_or_supplied_emotion(self):
        a, b = data.stress_event(4311, 1), data.stress_event(4311, 1, True)
        self.assertEqual(a['cue'], b['cue'])
        np.testing.assert_allclose(a['observed'], 1 - np.asarray(b['observed']))
        self.assertNotEqual(a['id'], b['id'])
        self.assertNotIn('emotion', a)

    def gate_rows(self, large=.008):
        return [{'world': world, 'origin': origin, 'task': task, 'trainCount': 192,
                 'kind': 'nn', 'width': width, 'status': 'complete',
                 'validationMSE': .01 if width == 16 else large}
                for world, origin, task, width in itertools.product(('linear', 'quadratic', 'nonlinear'), ('nn', 'linear', 'quadratic', 'memory'), ('plain', 'composed'), (16, 3125))]

    def setup_gate(self, folder):
        rows = [{'configuration': {'kind': 'nn', 'representation': 'raw8', 'width': 3125},
                 'status': 'complete', 'interventions': {'unrelatedBranch': {'relationalParametersExact': True}}} for _ in range(24)]
        numeric.write_json(folder / 'numeric-main.json', {'conditions': rows})

    def test_gate_uses_validation_before_test_and_refuses_changed_evidence(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(numeric, 'REPORTS', Path(temporary)):
            folder = Path(temporary)
            self.setup_gate(folder)
            rows = self.gate_rows()
            gate = numeric.expansion_gate(rows)
            self.assertTrue(gate['millionParameterGatePassed'])
            self.assertFalse(gate['finalTestInspectedAtGate'])
            self.assertTrue((folder / 'capacity-gate.json').exists())
            self.assertEqual(gate, numeric.expansion_gate(rows))
            changed = self.gate_rows(.007)
            with self.assertRaises(ValueError):
                numeric.expansion_gate(changed)

    def test_gate_requires_full_teacher_coverage(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(numeric, 'REPORTS', Path(temporary)):
            folder = Path(temporary)
            self.setup_gate(folder)
            gate = numeric.expansion_gate(self.gate_rows()[:-1])
            self.assertFalse(gate['millionParameterGatePassed'])

    def test_gate_rejects_below_ten_percent_improvement(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(numeric, 'REPORTS', Path(temporary)):
            folder = Path(temporary)
            self.setup_gate(folder)
            gate = numeric.expansion_gate(self.gate_rows(.0095))
            self.assertFalse(gate['millionParameterGatePassed'])

    def test_gate_requires_improvement_in_two_worlds_not_just_pooled_mean(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(numeric, 'REPORTS', Path(temporary)):
            folder = Path(temporary)
            self.setup_gate(folder)
            rows = self.gate_rows()
            for row in rows:
                if row['width'] == 3125:
                    row['validationMSE'] = .001 if row['world'] == 'linear' else .011
            gate = numeric.expansion_gate(rows)
            self.assertGreater(gate['improvementFraction'], .10)
            self.assertFalse(gate['millionParameterGatePassed'])

    def test_cached_diagnostic_rejects_changed_training_evidence(self):
        source = ROOT / 'reports/numeric-diagnostic/linear/nn/plain/n48/nn-h16.json'
        if not source.exists():
            self.skipTest('Completed frozen-function fixture unavailable.')
        saved = numeric.read_json(source)
        saved['trainingLabelsHash'] = 'changed-evidence'
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(numeric, 'REPORTS', Path(temporary)):
            cache = Path(temporary) / 'numeric-diagnostic/linear/nn/plain/n48/nn-h16.json'
            numeric.write_json(cache, saved)
            with self.assertRaises(ValueError):
                numeric.fit_candidates('linear', 'nn', False, 48, 'nn', 16)
            self.assertEqual(numeric.read_json(cache), saved)

    def test_rank_agreement_distinguishes_reversal_and_student_ties(self):
        labels = [{'space': 'relational', 'target': [value] * 7, 'features': [.5] * 8} for value in [-.2, .4]]
        expected = np.asarray([row['target'] for row in labels])
        for predicted, accuracy, ties in [(expected, 1., 0), (-expected, 0., 0), (np.zeros_like(expected), 0., 7)]:
            with mock.patch.object(numeric, 'arrays', return_value=predicted):
                result = numeric.rank_agreement(None, labels)
                self.assertEqual(result['agreement'], accuracy)
                self.assertEqual(result['studentTies'], ties)

    def test_frozen_layer_gradient_is_read_only_and_traces_replay(self):
        model = numeric.Personality(width=16, loss='bounded_logit')
        for event in data.training(count=4):
            model.update(event)
        original = model.snapshot()
        result = numeric.frozen_replay_gradient(model)
        self.assertEqual(original, model.snapshot())
        self.assertEqual(result['parameterHash'], original['parameterHash'])
        self.assertEqual(result['replayHash'], numeric.digest(model.replay['relational']))
        self.assertEqual(set(result['layerGradients']), {'w1', 'b1', 'w2', 'b2'})
        self.assertTrue(all(row['allFinite'] for row in result['layerGradients'].values()))

    def test_final_test_does_not_mutate_validation_rows_or_gate_hash(self):
        row = {'world': 'linear', 'origin': 'nn', 'task': 'plain', 'snapshot': 'unused', 'status': 'complete'}
        before = numeric.digest([row])
        with mock.patch.object(numeric, 'load_origin', return_value=(None, None)), mock.patch.object(numeric.data, 'source_labels', return_value=[]), mock.patch.object(numeric.Personality, 'load', return_value=None), mock.patch.object(numeric, 'mse', return_value=.1), mock.patch.object(numeric, 'rank_agreement', return_value={'agreement': .5}):
            final = numeric.final_test_rows([row])
        self.assertEqual(numeric.digest([row]), before)
        self.assertNotIn('testMSE', row)
        self.assertEqual(final[0]['testMSE'], .1)

    def test_million_final_test_requires_all_expected_conditions(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(numeric, 'REPORTS', Path(temporary)), mock.patch.object(numeric, 'final_test_rows') as test:
            with self.assertRaises(ValueError):
                numeric.million_finalize()
            test.assert_not_called()

    def test_gpu_final_test_requires_all_48_validation_cells_before_model_access(self):
        incomplete = {'conditions': [], 'selectionHash': numeric.digest([]), 'all48ValidationSelectionsFixedBeforeTest': True}
        with mock.patch.object(gpu_diagnostic, 'frozen_bundle', return_value={}), mock.patch.object(numeric, 'read_json', return_value=incomplete), mock.patch.object(numeric.Personality, 'load') as load:
            with self.assertRaises(ValueError):
                gpu_diagnostic.finalize()
            load.assert_not_called()

    def test_gpu_labels_refuse_tampered_target_even_when_persona_targets_not_read(self):
        if not gpu_diagnostic.LABEL_PATH.exists():
            self.skipTest('Frozen GPU label fixture unavailable.')
        bundle = numeric.read_json(gpu_diagnostic.LABEL_PATH)
        first = next(iter(bundle['cells'].values()))
        first['blocks']['train'][0]['target'][0] += .001
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'labels.json'
            numeric.write_json(path, bundle)
            with mock.patch.object(gpu_diagnostic, 'LABEL_PATH', path), self.assertRaises(ValueError):
                gpu_diagnostic.frozen_bundle()

    def test_gpu_labels_require_all_teacher_world_and_task_cells(self):
        if not gpu_diagnostic.LABEL_PATH.exists():
            self.skipTest('Frozen GPU label fixture unavailable.')
        bundle = numeric.read_json(gpu_diagnostic.LABEL_PATH)
        bundle['cells'].pop(next(iter(bundle['cells'])))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'labels.json'
            numeric.write_json(path, bundle)
            with mock.patch.object(gpu_diagnostic, 'LABEL_PATH', path), self.assertRaises(ValueError):
                gpu_diagnostic.frozen_bundle()

    def test_gpu_validation_cache_rejects_runtime_or_labels_change_without_overwrite(self):
        bundle = {'cells': {'linear/nn/plain': {'sourceSnapshot': 'source', 'sourceSnapshotSha256': 's', 'blockHashes': {'train': 't', 'validation': 'v'}}}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'reports/million-gpu-diagnostic/linear/nn/plain/n192/nn-h3125.json'
            numeric.write_json(path, {'status': 'complete', 'device': 'cpu'})
            label_path = root / 'labels.json'; label_path.write_text('{}', encoding='utf8')
            before = path.read_bytes()
            with mock.patch.object(gpu_diagnostic, 'ROOT', root), mock.patch.object(gpu_diagnostic, 'LABEL_PATH', label_path), mock.patch.object(gpu_diagnostic, 'protocol_hash', return_value='p'), mock.patch.object(gpu_diagnostic.torch.cuda, 'get_device_name', return_value='test-device'):
                with self.assertRaises(ValueError):
                    gpu_diagnostic.validation_cell(bundle, 'linear', 'nn', 'plain', 3125)
            self.assertEqual(path.read_bytes(), before)

    def test_gpu_artifact_commit_is_repeatable_but_refuses_changed_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'result.json'
            gpu_diagnostic.commit_json(path, {'selectionHash': 'fixed'})
            before = path.read_bytes()
            gpu_diagnostic.commit_json(path, {'selectionHash': 'fixed'})
            with self.assertRaises(ValueError):
                gpu_diagnostic.commit_json(path, {'selectionHash': 'changed'})
            self.assertEqual(path.read_bytes(), before)

    def test_million_online_workers_run_exact_formation_and_stress_without_cpu_diagnostic(self):
        with mock.patch.object(numeric, 'read_json', return_value={'millionParameterGatePassed': True}), mock.patch.object(numeric, 'train_one') as train, mock.patch.object(numeric, 'stress_one') as stress, mock.patch.object(numeric, 'fit_candidates') as diagnostic, mock.patch('builtins.print'):
            for worker in range(4):
                numeric.million_worker(worker, include_diagnostic=False)
            self.assertEqual(train.call_count, 8)
            formed = {(call.args[0]['loss'], call.args[0]['series']) for call in train.call_args_list}
            self.assertEqual(formed, set(itertools.product(data.LOSSES, range(4))))
            self.assertTrue(all(call.args[0]['width'] == 31250 and call.args[0]['seed'] == 4311 and call.args[1] == 'million-main' for call in train.call_args_list))
            self.assertEqual(stress.call_count, 2)
            self.assertEqual({call.args[1] for call in stress.call_args_list}, set(data.LOSSES))
            diagnostic.assert_not_called()

    def test_gpu_full_finalization_does_not_require_optional_cpu_reference_caches(self):
        rows = [{'width': width, 'world': world, 'origin': origin, 'task': task, 'status': 'complete',
                 'snapshot': f'models/nn-h{width}', 'parameterHash': 'fixed', 'parameterCount': 32*width+14,
                 'validationMSE': .1, 'elapsedSeconds': 1.}
                for width, (world, origin, task) in itertools.product(gpu_diagnostic.WIDTHS, gpu_diagnostic.JOBS)]
        saved = {'conditions': rows, 'selectionHash': numeric.digest(rows), 'all48ValidationSelectionsFixedBeforeTest': True}
        bundle = {'cells': {gpu_diagnostic.key(*job): {'blocks': {'test': [], 'validation': []}, 'blockHashes': {'test': 't'}} for job in gpu_diagnostic.JOBS}}
        def load(prefix, device):
            width = int(Path(prefix).name.split('h')[-1])
            return mock.Mock(parameter_hash=lambda: 'fixed', parameter_count=lambda: 32*width+14)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(gpu_diagnostic, 'ROOT', Path(temporary)), mock.patch.object(gpu_diagnostic, 'frozen_bundle', return_value=bundle), mock.patch.object(gpu_diagnostic, 'protocol_hash', return_value='p'), mock.patch.object(numeric, 'read_json', return_value=saved), mock.patch.object(numeric.Personality, 'load', side_effect=load), mock.patch.object(numeric, 'mse', return_value=.1), mock.patch.object(numeric, 'rank_agreement', return_value={'agreement': .5}), mock.patch.object(gpu_diagnostic, 'calibration', return_value=None), mock.patch.object(gpu_diagnostic, 'commit_json'), mock.patch('builtins.print'):
            result = gpu_diagnostic.finalize()
        self.assertEqual(result['gpuComplete'], 48)
        self.assertEqual(result['deviceCalibration']['comparisons'], 0)
        self.assertIsNone(result['deviceCalibration']['validationMSEAbsoluteDifferenceMax'])
        self.assertIsNone(result['deviceCalibration']['hundredThousandFinalTestMSEAbsoluteDifferenceMax'])
        self.assertEqual(result['cpuMillionDiagnosticCompleted'], 0)


if __name__ == '__main__':
    unittest.main()
