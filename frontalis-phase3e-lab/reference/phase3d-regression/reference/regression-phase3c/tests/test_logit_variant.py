"""Isolated post-result numerical diagnosis; no main-state replacement."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from npn import model, logit_variant as variant

RUNTIME = Path(__file__).resolve().parents[1] / 'runtime' / 'logit-tests'
RUNTIME.mkdir(parents=True, exist_ok=True)


def event(i=0, cue=None, observed=None, space='relational'):
    return {'id': f'variant-test-{i}', 'text': '研究用の事実結果を観測した。感情教師なし。',
            'space': space, 'cue': [.3,.65,.8,.45,.7,.2,.25,.8] if cue is None else cue,
            'observed': [.7,.35,.2,.55,.3,.8,.75,.2] if observed is None else observed,
            'confirmed': True, 'origin': 'synthetic'}


class VariantTests(unittest.TestCase):
    def test_all_family_start_and_prediction_equal_to_primary(self):
        for kind in model.KINDS:
            initial = variant.init(4311, kind)
            primary = model.init(4311, kind)
            self.assertEqual(model.parameter_digest(initial), model.parameter_digest(primary))
            self.assertEqual(variant.evaluate(initial, event()['cue'], 'relational')['raw'],
                             model.evaluate(primary, event()['cue'], 'relational')['raw'])
            self.assertEqual(initial['fixedPredictionGuard'], primary['fixedPredictionGuard'])

    def test_old_raw_selftarget_same_as_primary_at_same_parameters(self):
        primary = model.update(model.init(4311), event())
        changed = variant.update(variant.init(4311), event())
        self.assertEqual(primary['appraisals'][0]['target'], changed['appraisals'][0]['target'])
        self.assertEqual(primary['appraisals'][0]['effectBefore'], changed['appraisals'][0]['effectBefore'])
        self.assertEqual(changed['appraisals'][0]['sourceStateHash'], model.digest(variant.init(4311)))

    def test_logit_cap_finite_at_extremes(self):
        result = variant._logit_target(np.array([-1., -.2, 0., .2, 1.]), 6.)
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(result[0], -6.)
        self.assertEqual(result[-1], 6.)
        self.assertEqual(result[2], 0.)

    def test_direct_preactivation_matches_core_output(self):
        for kind in model.KINDS:
            state = variant.update(variant.init(4311, kind), event())
            logits = variant.preactivation(state, model.centered(event()['cue']), 'relational')[0]
            np.testing.assert_array_equal(np.tanh(logits), variant.evaluate(state, event()['cue'], 'relational')['raw'])

    def test_all_families_learn_and_remain_finite(self):
        for kind in model.KINDS:
            original = variant.init(4311, kind)
            after = variant.update(original, event())
            self.assertNotEqual(model.parameter_digest(original), model.parameter_digest(after))
            self.assertTrue(variant.validate(after))
            self.assertTrue(np.isfinite(variant.evaluate(after, event()['cue'], 'relational')['raw']).all())
            self.assertEqual(original['fixedPredictionGuard'], after['fixedPredictionGuard'])

    def test_other_branch_parameters_unchanged(self):
        for kind in model.KINDS:
            original = variant.init(4311, kind)
            after = variant.update(original, event())
            self.assertEqual(original['branches']['acoustic'], after['branches']['acoustic'])

    def test_information_zero_noop_on_learned_state(self):
        for kind in model.KINDS:
            before = variant.update(variant.init(4311, kind), event())
            after = variant.update(before, event(1, cue=[.5]*8, observed=[.5]*8))
            self.assertEqual(model.parameter_digest(before), model.parameter_digest(after))
            self.assertEqual(before['replay'], after['replay'])

    def test_learning_off_and_missing_observation(self):
        before = variant.init(4311, learning=False)
        after = variant.update(before, event())
        self.assertEqual(model.parameter_digest(before), model.parameter_digest(after))
        with self.assertRaises(ValueError):
            variant.update(before, {**event(), 'observed': None})

    def test_duplicate_and_external_gold_rejected(self):
        before = variant.update(variant.init(4311), event())
        with self.assertRaises(ValueError):
            variant.update(before, event())
        with self.assertRaises(ValueError):
            variant.update(variant.init(4311), {**event(), 'emotion': 'joy'})

    def test_primary_state_cannot_silently_use_variant_update(self):
        with self.assertRaises(ValueError):
            variant.update(model.init(4311), event())

    def test_vanishing_output_gradient_distinct_from_flat_old_values(self):
        state = variant.init(4311)
        state['branches']['relational']['b2'] = [15.] * 7
        probe = variant.gradient_probe(state, event())
        self.assertLess(probe['cueEffectMeanAbsolute'], 1e-10)
        self.assertLess(probe['outputTanhDerivativeMean'], 1e-10)
        self.assertLess(probe['signalOutputGradientNorm'], 1e-10)
        self.assertGreater(probe['logitOutputGradientNorm'], 1.)
        # A nonzero logit derivative does not make an opposite value teacher.
        self.assertTrue(np.all(np.asarray(probe['boundedTargetRaw']) > .99))

    def test_emergency_rejection_retains_original(self):
        state = variant.init(4311)
        state['branches']['relational']['w2'][0][0] = 60.
        original = model.digest(state)
        with self.assertRaises(ValueError):
            variant.update(state, event())
        self.assertEqual(model.digest(state), original)

    def test_variant_snapshot_and_logit_provenance(self):
        state = variant.update(variant.init(4311), event())
        with tempfile.TemporaryDirectory(dir=RUNTIME) as directory:
            path = Path(directory) / 'variant.json'
            variant.save(path, state)
            self.assertEqual(variant.load(path), state)
        changed = copy.deepcopy(state)
        changed['replay']['relational'][0]['targetLogit'][0] += .1
        with self.assertRaises(ValueError):
            variant.validate(changed)


if __name__ == '__main__':
    unittest.main()

