import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from npn.model import (AXES, CHANNELS, KINDS, SPACES, init, update, evaluate,
                       digest, parameter_digest, parameter_norm, replay,
                       train_on_labels, predict)
from npn.store import validate, save, load, Store

TEST_TEMP = Path(__file__).resolve().parents[1] / 'runtime' / 'tests'
TEST_TEMP.mkdir(parents=True, exist_ok=True)


def event(i=0, space='relational', cue=None, observed=None):
    return {'id': f'test-{i}', 'text': '研究用の確認済み出来事。結果を観測した。',
            'space': space, 'cue': cue or [.8, .2, .7, .2, .6, .8, .2, .7],
            'observed': observed or [.2, .8, .2, .7, .8, .3, .8, .4],
            'confirmed': True, 'origin': 'synthetic'}


class LearningTests(unittest.TestCase):
    def test_initial_outputs_identical_across_families(self):
        point = [.13, .85, .71, .26, .89, .37, .69, .24]
        reference = evaluate(init(4311), point, 'relational')['raw']
        for kind in KINDS:
            for space in SPACES:
                self.assertEqual(reference, evaluate(init(4311, kind), point, space)['raw'])

    def test_same_seed_same_state(self):
        self.assertEqual(init(4311), init(4311))
        self.assertNotEqual(init(4311)['commonBase'], init(4312)['commonBase'])

    def test_prediction_fixed_after_all_family_learning(self):
        for kind in KINDS:
            state = init(4311, kind)
            guard = copy.deepcopy(state['fixedPredictionGuard'])
            for i in range(3):
                state = update(state, event(i))
            self.assertEqual(state['fixedPredictionGuard'], guard)
            self.assertEqual(predict(state, event()['cue'], 'relational'), [.5] * 8)
            self.assertEqual(evaluate(state, event()['observed'], 'relational')['predictedOutcome'], [.5] * 8)

    def test_value_parameter_changes(self):
        for kind in KINDS:
            original = init(4311, kind)
            result = update(original, event())
            self.assertNotEqual(parameter_digest(original), parameter_digest(result))
            self.assertNotEqual(evaluate(original, event()['cue'], 'relational')['raw'],
                                evaluate(result, event()['cue'], 'relational')['raw'])

    def test_neural_hidden_representation_learns(self):
        before = init(4311)
        after = update(before, event())
        self.assertNotEqual(before['branches']['relational']['w1'], after['branches']['relational']['w1'])
        self.assertNotEqual(before['branches']['relational']['w2'], after['branches']['relational']['w2'])

    def test_target_comes_from_old_observed_appraisal(self):
        before = init(4311)
        prior = np.asarray(evaluate(before, event()['observed'], 'relational')['raw'])
        expected = np.tanh(1.25 * np.arctanh(prior))
        after = update(before, event())
        np.testing.assert_array_equal(expected, after['appraisals'][0]['target'])
        self.assertEqual(after['appraisals'][0]['sourceStateHash'], digest(before))
        np.testing.assert_array_equal(prior, after['appraisals'][0]['effectBefore'])

    def test_frozen_old_labels_do_not_rewrite(self):
        state = update(init(4311), event())
        previous = copy.deepcopy(state['replay']['relational'])
        state = update(state, event(1, observed=[.7, .2, .8, .1, .4, .9, .2, .6]))
        self.assertEqual(state['replay']['relational'][:2], previous)

    def test_self_amplification_magnitudes(self):
        after = update(init(4311), event())
        appraisal = after['appraisals'][0]
        self.assertTrue(np.all(np.abs(appraisal['target']) >= np.abs(appraisal['effectBefore'])))
        self.assertTrue(np.all(np.abs(appraisal['target']) <= 1))

    def test_gain_one_is_association_not_no_learning(self):
        before = init(4311, gain=1.)
        after = update(before, event())
        np.testing.assert_array_equal(after['appraisals'][0]['target'], after['appraisals'][0]['effectBefore'])
        self.assertNotEqual(parameter_digest(before), parameter_digest(after))

    def test_learning_off_keeps_parameters(self):
        before = init(4311, learning=False)
        after = update(before, event())
        self.assertEqual(parameter_digest(before), parameter_digest(after))
        self.assertEqual(after['generation'], 1)
        self.assertEqual(after['replay']['relational'], [])

    def test_same_cue_different_results_learn_different_values(self):
        first = update(init(4311), event())
        second = update(init(4311), event(observed=[1 - x for x in event()['observed']]))
        self.assertGreater(np.linalg.norm(np.asarray(evaluate(first, event()['cue'], 'relational')['raw']) -
                           np.asarray(evaluate(second, event()['cue'], 'relational')['raw'])), .05)
        self.assertEqual(first['fixedPredictionGuard'], second['fixedPredictionGuard'])

    def test_domain_update_does_not_touch_other_branch(self):
        for kind in KINDS:
            before = init(4311, kind)
            point = event()['cue']
            after = update(before, event())
            self.assertEqual(before['branches']['acoustic'], after['branches']['acoustic'])
            self.assertEqual(evaluate(before, point, 'acoustic'),
                             {**evaluate(after, point, 'acoustic'),
                              'generation': before['generation'],
                              'parameterHash': parameter_digest(before)})

    def test_no_information_no_parameter_drift_after_training(self):
        for kind in KINDS:
            state = update(init(4311, kind), event())
            parameters = parameter_digest(state)
            remembered = copy.deepcopy(state['replay'])
            for i in range(5):
                state = update(state, event(i + 1, cue=[.5] * 8, observed=[.5] * 8))
            self.assertEqual(parameter_digest(state), parameters)
            self.assertEqual(state['replay'], remembered)
            self.assertTrue(state['appraisals'][-1]['noInformation'])
            self.assertEqual(state['generation'], 6)

    def test_input_state_and_source_are_not_mutated(self):
        state = init(4311)
        source = event()
        before = digest(state)
        after = update(state, source)
        self.assertEqual(digest(state), before)
        source['observed'][0] = .95
        self.assertNotEqual(after['history'][0]['observed'], source['observed'])

    def test_replay_reproducible(self):
        sources = [event(i) for i in range(4)]
        self.assertEqual(replay(sources, 4311), replay(sources, 4311))

    def test_repeated_experience_can_strengthen(self):
        point = [.82, .23, .66, .29, .91, .68, .34, .77]
        state = init(4311)
        before = np.abs(evaluate(state, point, 'relational')['raw']).mean()
        for i in range(16):
            state = update(state, event(i, cue=point, observed=point))
        after = np.abs(evaluate(state, point, 'relational')['raw']).mean()
        self.assertGreater(after, before)
        self.assertLessEqual(after, 1.)

    def test_new_results_can_modify_an_old_cue(self):
        state = init(4311)
        for i in range(6):
            state = update(state, event(i))
        before = np.asarray(evaluate(state, event()['cue'], 'relational')['raw'])
        for i in range(6, 14):
            state = update(state, event(i, observed=[1 - x for x in event()['observed']]))
        after = np.asarray(evaluate(state, event()['cue'], 'relational')['raw'])
        self.assertGreater(np.linalg.norm(after - before), .01)

    def test_replay_window_bounded(self):
        state = init(4311, 'memory')
        for i in range(36):
            state = update(state, event(i))
        self.assertEqual(len(state['replay']['relational']), 64)
        self.assertEqual(len(state['branches']['relational']['memory']), 64)
        self.assertEqual(len(state['history']), 36)

    def test_nonnegative_emotional_intensities(self):
        state = replay([event(i) for i in range(4)], 4311)
        for point in (event()['cue'], event()['observed'], [0.] * 8, [1.] * 8):
            value = evaluate(state, point, 'relational')
            self.assertTrue(np.all(np.abs(value['raw']) <= 1))
            for channel in ('anger', 'joy', 'sadness', 'dislike', 'initiative', 'resistance'):
                self.assertGreaterEqual(value['signals'][channel], 0)
                self.assertLessEqual(value['signals'][channel], 1)

    def test_emotion_channels_not_affinity_relabeling(self):
        value = evaluate(init(4311), event()['cue'], 'relational')['channels']
        self.assertEqual(len(set(value.values())), 7)

    def test_frozen_labels_diagnostic_retains_provenance(self):
        state = init(4311)
        labels = [{'space': 'relational', 'features': event()['cue'],
                   'target': [.2, -.1, .4, -.3, .2, .3, -.2]}]
        result = train_on_labels(state, labels, 'matched-label-test')
        self.assertEqual(result['generation'], 0)
        self.assertEqual(result['history'], [])
        self.assertEqual(result['diagnostics'][0]['sourceStateHash'], digest(state))
        self.assertEqual(result['diagnostics'][0]['labelsHash'], digest(labels))
        self.assertEqual(result['branches']['acoustic'], state['branches']['acoustic'])
        self.assertTrue(validate(result))


class RejectionTests(unittest.TestCase):
    def test_duplicate_id_rejected(self):
        state = update(init(4311), event())
        original = digest(state)
        with self.assertRaises(ValueError):
            update(state, event())
        self.assertEqual(original, digest(state))

    def test_unconfirmed_or_unobserved_rejected(self):
        for patch in ({'confirmed': False}, {'origin': 'llm-guess'}, {'observed': None}):
            with self.assertRaises(ValueError):
                update(init(4311), {**event(), **patch})

    def test_external_subjective_teachers_rejected(self):
        for key in ('reward', 'praise', 'emotion', 'personalityTarget', 'userRating', 'seriesId'):
            with self.assertRaises(ValueError):
                update(init(4311), {**event(), key: .8})

    def test_bad_shape_range_and_nonfinite_rejected(self):
        for bad in ([.5] * 7, [2.] * 8, [float('nan')] * 8, [float('inf')] * 8):
            with self.assertRaises(ValueError):
                evaluate(init(4311), bad, 'relational')

    def test_changed_predictor_guard_rejected(self):
        state = init(4311)
        state['fixedPredictionGuard']['outcomes'][0] = .8
        with self.assertRaises(ValueError):
            update(state, event())
        with self.assertRaises(ValueError):
            validate(state)

    def test_emergency_rejection_does_not_mutate_original(self):
        state = init(4311)
        state['branches']['relational']['w2'][0][0] = 60.
        original = digest(state)
        with self.assertRaises(ValueError):
            update(state, event())
        self.assertEqual(digest(state), original)

    def test_invalid_branch_space_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(init(4311), [.5] * 8, 'unknown')
        with self.assertRaises(ValueError):
            update(init(4311), event(space='unknown'))


class StoreTests(unittest.TestCase):
    def test_all_family_snapshot_roundtrip(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            for kind in KINDS:
                state = update(init(4311, kind), event())
                target = Path(folder) / (kind + '.json')
                save(target, state)
                self.assertEqual(load(target), state)

    def test_hash_corruption_detected(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            path = Path(folder) / 'state.json'
            save(path, init(4311))
            wrapper = json.loads(path.read_text(encoding='utf8'))
            wrapper['payload']['branches']['relational']['b2'][0] = .1
            path.write_text(json.dumps(wrapper), encoding='utf8')
            with self.assertRaises(ValueError):
                load(path)

    def test_ledger_tampering_detected(self):
        state = update(init(4311), event())
        state['history'][0]['text'] = '改変'
        with self.assertRaises(ValueError):
            validate(state)

    def test_target_tampering_detected_even_with_rehashed_ledger(self):
        state = update(init(4311), event())
        state['appraisals'][0]['target'][0] = .99
        state['ledgerHash'] = digest([digest([]), state['history'][0], state['appraisals'][0]])
        with self.assertRaises(ValueError):
            validate(state)

    def test_shapes_nonfinite_and_schema_detected(self):
        for mutate in (lambda s: s.update(schema='wrong'),
                       lambda s: s['branches']['relational'].update(w1=[[0.]]),
                       lambda s: s['branches']['relational']['b2'].__setitem__(0, float('inf'))):
            state = init(4311); mutate(state)
            with self.assertRaises(ValueError):
                validate(state)

    def test_rollback_appends_and_preserves_old_states(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            store = Store(folder)
            original = init(4311)
            learned = update(original, event())
            self.assertEqual(store.commit(original, kind='initial'), 0)
            self.assertEqual(store.commit(learned, expected_head=0), 1)
            self.assertEqual(store.rollback(0, expected_head=1), 2)
            self.assertEqual(store.load(), original)
            self.assertEqual(store.load(1), learned)
            self.assertEqual(len(store.export()['manifest']['generations']), 3)

    def test_stale_generation_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            store = Store(folder)
            store.commit(init(4311))
            with self.assertRaises(ValueError):
                store.commit(init(4311), expected_head=None)

    def test_export_restore_and_empty_export(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            store = Store(Path(folder) / 'one')
            store.commit(init(4311), kind='initial')
            store.commit(update(init(4311), event()), expected_head=0)
            store.rollback(0, 1)
            restored = Store(Path(folder) / 'two')
            restored.restore(store.export())
            self.assertEqual(restored.export(), store.export())
            empty = Store(Path(folder) / 'empty')
            other = Store(Path(folder) / 'empty-restored')
            other.restore(empty.export())
            self.assertEqual(empty.export(), other.export())

    def test_bad_restore_ancestry_rejected_without_partial_write(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            first = Store(Path(folder) / 'first')
            first.commit(init(4311))
            bundle = first.export()
            bundle['manifest']['generations'][0]['parent'] = 19
            second = Store(Path(folder) / 'second')
            with self.assertRaises(ValueError):
                second.restore(bundle)
            self.assertIsNone(second.export()['manifest']['head'])

    def test_malformed_wrapper_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            path = Path(folder) / 'bad.json'
            path.write_text('{"sha256":"missing"}', encoding='utf8')
            with self.assertRaises(ValueError):
                load(path)
        with self.assertRaises(ValueError):
            validate([])

    def test_restore_bad_later_kind_rejected_before_any_write(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            first = Store(Path(folder) / 'first')
            first.commit(init(4311))
            first.commit(update(init(4311), event()), expected_head=0)
            bundle = first.export()
            bundle['manifest']['generations'][1]['kind'] = 'unknown'
            second = Store(Path(folder) / 'second')
            with self.assertRaises(ValueError):
                second.restore(bundle)
            self.assertIsNone(second.export()['manifest']['head'])


if __name__ == '__main__':
    unittest.main()

