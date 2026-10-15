import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from npne.meaning import validate,encode
from npne.data import training,random_meanings,teacher
from experiments.run import baseline,basis

class ExperimentalContract(unittest.TestCase):
    def test_same_cues_distinct_observed_results(self):
        groups=[training(5311,s) for s in range(4)]
        for i in range(48):
            for g in groups[1:]:np.testing.assert_array_equal(encode(groups[0][i]['meaning'],observed=False),encode(g[i]['meaning'],observed=False))
        self.assertFalse(np.array_equal(encode(groups[0][0]['meaning']),encode(groups[1][0]['meaning'])))
    def test_authored_corpus_splits_and_paraphrases(self):
        data=json.loads((ROOT/'data/natural.json').read_text(encoding='utf8'));self.assertEqual(len(data),32)
        self.assertEqual(sum(c['split']=='heldout' for c in data),8)
        self.assertEqual(sum(c['split']=='formation' and c['outcomeConfirmed'] for c in data),15)
        for c in data:validate(c['gold'])
        for family in range(8):self.assertTrue(all(c['gold']==data[4*family]['gold'] for c in data if c['family']==family))
    def test_ridge_dual_matches_primal(self):
        rng=np.random.default_rng(15);x=rng.uniform(0,1,(10,3));q=rng.uniform(0,1,(4,3));y=rng.uniform(-.5,.5,(10,7))
        for quadratic in [False,True]:
            a=basis(x,quadratic);b=basis(q,quadratic);z=np.arctanh(y);expected=np.tanh(b@np.linalg.solve(a.T@a+.01*np.eye(a.shape[1]),a.T@z))
            np.testing.assert_allclose(baseline(x,y,q,'quadratic' if quadratic else 'linear'),expected,atol=1e-12)
    def test_diagnostic_frozen_data(self):
        ms=random_meanings(99,20);x=np.asarray([encode(m) for m in ms])
        for kind in ['linear','quadratic','interaction']:
            a=teacher(x,kind,5311);np.testing.assert_array_equal(a,teacher(x,kind,5311));self.assertTrue(np.isfinite(a).all());self.assertEqual(a.shape,(20,7))

if __name__=='__main__':unittest.main()
