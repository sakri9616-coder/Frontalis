"""Meaningful parity, isolation, saturation and persistence checks."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from npnd.core import Personality, AXES, CHANNELS, GUARD

spec = importlib.util.spec_from_file_location('phase3c_reference', ROOT / 'reference' / 'npn' / 'model.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
torch.set_num_threads(1)

def event(n=0, space='relational'):
    rng = np.random.default_rng(n + 1701)
    return {'id': 'e' + str(n), 'space': space, 'cue': rng.uniform(.02, .98, 8).tolist(), 'observed': rng.uniform(.02, .98, 8).tolist(), 'confirmed': True, 'origin': 'synthetic', 'text': '共同で行った試みと、その確認済みの結果。'}

def parameters(model, space):
    return {k: v.detach().cpu().numpy().copy() for k, v in model.branches[space].items() if isinstance(v, torch.Tensor)}

class CoreTests(unittest.TestCase):
    def test_actual_parameter_counts_raw(self):
        for width, count in [(16, 526), (64, 2062), (312, 9998), (3125, 100014), (31250, 1000014)]:
            with self.subTest(width=width):
                m = Personality(width=width)
                self.assertEqual(m.parameter_count(), count)
                self.assertEqual(m.diagnostics()['fixedBaseParameters'], 56)

    def test_actual_parameter_counts_expansion(self):
        for width, count in [(5, 534), (19, 1990), (96, 9998), (962, 100062)]:
            self.assertEqual(Personality(width=width, representation='quadratic44').parameter_count(), count)

    def test_baseline_counts(self):
        self.assertEqual(Personality(kind='linear').parameter_count(), 126)
        self.assertEqual(Personality(kind='quadratic').parameter_count(), 630)
        self.assertEqual(Personality(kind='memory').parameter_count(), 0)

    def test_all_scales_same_initial_readout(self):
        x = np.random.default_rng(80).uniform(size=(15, 8))
        base = Personality().evaluate_batch(x, 'relational')
        for width in (16, 64, 312, 3125):
            for repr_name in ('raw8', 'quadratic44'):
                for scaled in (True, False):
                    np.testing.assert_array_equal(Personality(width=width, representation=repr_name, scaled=scaled).evaluate_batch(x, 'relational'), base)
        for kind in ('linear', 'quadratic', 'memory'):
            np.testing.assert_array_equal(Personality(kind=kind).evaluate_batch(x, 'relational'), base)

    def test_phase3c_initial_parameter_and_response_bridge(self):
        old = reference.init(4311)
        new = Personality(loss='legacy_signal')
        np.testing.assert_array_equal(new.common_base.numpy(), old['commonBase'])
        for space in ('relational', 'acoustic'):
            for name, a in parameters(new, space).items():
                np.testing.assert_array_equal(a, old['branches'][space][name])
            x = event()['cue']
            np.testing.assert_allclose(new.evaluate(x, space)['raw'], reference.evaluate(old, x, space)['raw'], atol=1e-15, rtol=0)

    def test_phase3c_one_and_48_update_parity(self):
        old = reference.init(4311)
        new = Personality(loss='legacy_signal')
        queries = np.random.default_rng(1).uniform(size=(17, 8))
        for n in range(48):
            e = event(n, 'acoustic' if n % 4 == 0 else 'relational')
            old = reference.update(old, e)
            record = new.update(e)
            np.testing.assert_allclose(record['target'], old['appraisals'][-1]['target'], atol=1e-10, rtol=0)
            if n in (0, 47):
                for space in ('relational', 'acoustic'):
                    expected = np.asarray([reference.evaluate(old, x, space)['raw'] for x in queries])
                    np.testing.assert_allclose(new.evaluate_batch(queries, space), expected, atol=1e-10, rtol=0)
                    for name, a in parameters(new, space).items():
                        np.testing.assert_allclose(a, old['branches'][space][name], atol=1e-10, rtol=0)

    def test_old_effect_not_external_target_drives_update(self):
        m = Personality()
        e = event()
        before = m.evaluate(e['observed'], e['space'])['raw']
        record = m.update(e)
        expected = np.tanh(np.clip(1.25 * np.arctanh(np.clip(before, -.999999, .999999)), -6, 6))
        np.testing.assert_allclose(record['target'], expected, atol=1e-15, rtol=0)
        self.assertEqual(record['effectBefore'], before)
        self.assertNotEqual(record['parameterHashBefore'], record['parameterHashAfter'])

    def test_hidden_weights_actually_learn(self):
        m = Personality(width=64)
        before = parameters(m, 'relational')['w1']
        m.update(event())
        self.assertGreater(np.max(np.abs(before - parameters(m, 'relational')['w1'])), 1e-5)

    def test_other_branch_bitwise_unchanged(self):
        for kind in ('nn', 'linear', 'quadratic', 'memory'):
            m = Personality(kind=kind)
            before = copy.deepcopy(m.branches['acoustic'])
            q = m.evaluate(event()['cue'], 'acoustic')
            m.update(event())
            if kind == 'memory':
                self.assertEqual(before, m.branches['acoustic'])
            else:
                for key in before:
                    self.assertTrue(torch.equal(before[key], m.branches['acoustic'][key]))
            self.assertEqual(q['raw'], m.evaluate(event()['cue'], 'acoustic')['raw'])

    def test_disabled_learning_stores_only_evidence(self):
        m = Personality(learning=False)
        before = m.parameter_hash()
        m.update(event())
        self.assertEqual(before, m.parameter_hash())
        self.assertEqual(m.generation, 1)
        self.assertEqual(len(m.replay['relational']), 0)

    def test_zero_information_skip(self):
        m = Personality()
        e = event(); e['cue'] = [.5] * 8; e['observed'] = [.5] * 8
        h = m.parameter_hash()
        r = m.update(e)
        self.assertTrue(r['noInformation'])
        self.assertEqual(h, m.parameter_hash())

    def test_fixed_prediction_forever(self):
        m = Personality()
        for n in range(8):
            m.update(event(n))
            self.assertEqual(m.evaluate(event(n)['cue'], 'relational')['predictedOutcome'], [.5] * 8)
        self.assertEqual(m.fixed_prediction_guard, GUARD)

    def test_reactions_unsigned_finite(self):
        m = Personality()
        for n in range(4):
            m.update(event(n))
        for x in np.random.default_rng(20).uniform(size=(30, 8)):
            signals = m.evaluate(x, 'relational')['signals']
            for name in ('anger', 'joy', 'sadness', 'initiative', 'resistance', 'dislike', 'curiosity'):
                self.assertGreaterEqual(signals[name], 0.)
                self.assertLessEqual(signals[name], 1.)

    def test_all_external_subjective_targets_rejected(self):
        forbidden = ('reward', 'praise', 'userRating', 'valence', 'emotion', 'personalityTarget', 'teacherCoefficients', 'target', 'seriesId')
        for key in forbidden:
            m = Personality(); before = m.snapshot()
            e = event(); e[key] = 1
            with self.assertRaises(ValueError):
                m.update(e)
            self.assertEqual(m.snapshot(), before)

    def test_missing_observation_not_learned(self):
        m = Personality(); before = m.snapshot(); e = event(); del e['observed']
        with self.assertRaises(ValueError):
            m.update(e)
        self.assertEqual(before, m.snapshot())

    def test_invalid_features_rejected_atomically(self):
        for bad in ([float('nan')] * 8, [float('inf')] * 8, [1.1] * 8, [True] * 8, [.5] * 7):
            m = Personality(); before = m.snapshot(); e = event(); e['observed'] = bad
            with self.assertRaises(ValueError):
                m.update(e)
            self.assertEqual(before, m.snapshot())

    def test_invalid_retained_text_metadata_atomic(self):
        m = Personality(); before = m.snapshot(); e = event(); e['badExtra'] = float('nan')
        with self.assertRaises(ValueError):
            m.update(e)
        self.assertEqual(before, m.snapshot())

    def test_duplicate_rejected(self):
        m = Personality(); e = event(); m.update(e); before = m.snapshot()
        with self.assertRaises(ValueError):
            m.update(e)
        self.assertEqual(before, m.snapshot())

    def test_no_overwrite_snapshot(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'runtime') as directory:
            prefix = Path(directory) / 'generation-0'; m = Personality(); m.save(prefix)
            before = Path(str(prefix) + '.npz').read_bytes()
            with self.assertRaises(FileExistsError):
                m.save(prefix)
            self.assertEqual(before, Path(str(prefix) + '.npz').read_bytes())

    def test_restore_exact_and_continued_learning(self):
        for kind in ('nn', 'linear', 'quadratic', 'memory'):
            with tempfile.TemporaryDirectory(dir=ROOT / 'runtime') as directory:
                m = Personality(kind=kind)
                for n in range(4):
                    m.update(event(n))
                m.save(Path(directory) / 'state')
                restored = Personality.load(Path(directory) / 'state')
                self.assertEqual(m.snapshot(), restored.snapshot())
                m.update(event(8)); restored.update(event(8))
                self.assertEqual(m.snapshot(), restored.snapshot())

    def test_weight_tampering_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'runtime') as directory:
            prefix = Path(directory) / 'state'; Personality().save(prefix)
            file = Path(str(prefix) + '.npz'); contents = bytearray(file.read_bytes()); contents[-10] ^= 1; file.write_bytes(contents)
            with self.assertRaises(ValueError):
                Personality.load(prefix)

    def test_metadata_tampering_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'runtime') as directory:
            prefix = Path(directory) / 'state'; Personality().save(prefix)
            file = Path(str(prefix) + '.json'); data = json.loads(file.read_text('utf8')); data['metadata']['configuration']['gain'] = 1.7; file.write_text(json.dumps(data), encoding='utf8')
            with self.assertRaises(ValueError):
                Personality.load(prefix)

    def test_clone_branches_and_evidence_independent(self):
        m = Personality(); m.update(event()); clone = m.clone(); before = m.snapshot()
        clone.update(event(1))
        self.assertEqual(before, m.snapshot())
        self.assertNotEqual(m.parameter_hash(), clone.parameter_hash())

    def test_training_emergency_keeps_original(self):
        m = Personality()
        before = m.snapshot()
        bad_branch = {k: v.detach().clone().requires_grad_(True) for k, v in m.branches['relational'].items()}
        with torch.no_grad():
            bad_branch['b2'].fill_(100.)
        with mock.patch.object(m, '_train_branch', return_value=(bad_branch, {'losses': [], 'gradientNorm': 1.})):
            with self.assertRaises(ValueError):
                m.update(event())
        self.assertEqual(before, m.snapshot())

    def test_prediction_guard_tampering_rejected(self):
        m = Personality(); m.fixed_prediction_guard['outcomes'][0] = 0.
        with self.assertRaises(ValueError):
            m.evaluate([.5] * 8, 'relational')

    def test_bounded_variants_same_self_target(self):
        signal = Personality(loss='bounded_signal'); logit = Personality(loss='bounded_logit')
        a = signal.update(event()); b = logit.update(event())
        self.assertEqual(a['target'], b['target'])
        self.assertEqual(a['targetLogit'], b['targetLogit'])
        self.assertLessEqual(max(np.abs(a['targetLogit'])), 6.)

    def test_logit_loss_handles_saturated_readout_gradient(self):
        gradients = {}
        for loss in ('bounded_signal', 'bounded_logit'):
            m = Personality(loss=loss)
            with torch.no_grad():
                m.branches['relational']['b2'].fill_(12.)
            record = m.update(event())
            gradients[loss] = record['training']['gradientNormFirst']
            self.assertLessEqual(max(np.abs(record['targetLogit'])), 6.)
        self.assertLess(gradients['bounded_signal'], 1e-10)
        self.assertGreater(gradients['bounded_logit'], 1.)

    def test_full_frozen_diagnostic_not_truncated(self):
        labels = [{'space': 'relational', 'features': event(n)['cue'], 'target': [0.] * 7} for n in range(100)]
        labels[0]['target'] = [.8] * 7
        all_model = Personality(kind='linear'); tail_model = all_model.clone()
        all_model.fit_labels(labels, epochs=2); tail_model.fit_labels(labels[-64:], epochs=2)
        self.assertNotEqual(all_model.parameter_hash(), tail_model.parameter_hash())
        self.assertEqual(all_model.diagnostic_history[-1]['count'], 100)

    def test_frozen_fit_failure_atomic_both_branches(self):
        m = Personality(); before = m.snapshot()
        labels = [{'space': 'relational', 'features': [.1] * 8, 'target': [0.] * 7}, {'space': 'acoustic', 'features': [.1] * 7, 'target': [0.] * 7}]
        with self.assertRaises(ValueError):
            m.fit_labels(labels)
        self.assertEqual(before, m.snapshot())

    def test_float32_supported_with_explicit_precision_limit(self):
        f64, f32 = Personality(), Personality(dtype='float32')
        x = np.random.default_rng(8).uniform(size=(20, 8))
        np.testing.assert_allclose(f64.evaluate_batch(x, 'relational'), f32.evaluate_batch(x, 'relational'), atol=1e-7, rtol=0)
        f32.update(event())
        self.assertTrue(np.isfinite(f32.evaluate_batch(x, 'relational')).all())

    def test_scale_configuration_visible(self):
        self.assertEqual(Personality(width=64).diagnostics()['readoutScale'], .25)
        self.assertEqual(Personality(width=64, scaled=False).diagnostics()['readoutScale'], 1.)

if __name__ == '__main__':
    unittest.main()
