import unittest
from npn import data


class DataTests(unittest.TestCase):
    def test_identical_cues_distinct_outcomes(self):
        worlds = [data.stream(4311, i) for i in range(4)]
        self.assertEqual([r['cue'] for r in worlds[0]], [r['cue'] for r in worlds[3]])
        self.assertEqual([r['event'] for r in worlds[0]], [r['event'] for r in worlds[3]])
        self.assertNotEqual([r['observed'] for r in worlds[0]], [r['observed'] for r in worlds[1]])

    def test_disjoint_splits(self):
        splits = [data.stream(4311, 0, phase=p) for p in ('train','validation','test')]
        for i, a in enumerate(splits):
            for b in splits[i+1:]:
                self.assertTrue({r['id'] for r in a}.isdisjoint({r['id'] for r in b}))
                self.assertNotEqual([r['cue'] for r in a], [r['cue'] for r in b])

    def test_no_value_gold_or_series_annotation(self):
        forbidden = {'reward','emotion','affinity','value','teacherCoefficients','series','personalityTarget'}
        for condition in data.CONDITIONS:
            for row in data.stream(4311, 0, condition):
                self.assertTrue(forbidden.isdisjoint(row))
                self.assertEqual(len(row['cue']), 8)
                self.assertEqual(len(row['observed']), 8)
                self.assertTrue(all(0 <= x <= 1 for x in row['cue'] + row['observed']))
                self.assertTrue(row['confirmed'])
                self.assertEqual(row['origin'], 'synthetic')

    def test_unrelated_is_other_space_only(self):
        self.assertEqual({r['space'] for r in data.additional(kind='unrelated')}, {'acoustic'})
        self.assertEqual({r['space'] for r in data.additional(kind='reversal')}, {'relational'})

    def test_repeatable_fixture(self):
        self.assertEqual(data.stream(), data.stream())
        self.assertNotEqual(data.stream(4311), data.stream(4312))

    def test_probes_no_optimal_answer_and_same_forecast(self):
        probes = data.probes()
        self.assertEqual(len({r['id'] for r in probes}), len(probes))
        for row in probes:
            self.assertEqual(row['fixedPrediction'], [.5] * 8)
            self.assertNotIn('target', row)
            self.assertNotIn('expectedChoice', row)
            self.assertEqual(len(row['features']), 8)


if __name__ == '__main__':
    unittest.main()
