import copy,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from npnf.core import *
from npnf.data import *
torch.set_num_threads(1)

class Tests(unittest.TestCase):
    def test_parameter_counts_and_equal_initial_capacity(self):
        expected=None
        for cap,w in enumerate(WIDTHS):
            n=TemporalNPN(capacity=cap);self.assertEqual(n.count(),102*w)
            y=n.evaluate(present_event(0)['interpretation'])['longTerm']
            if expected is None:expected=y
            self.assertEqual(y,expected)
    def test_old_npn_target_not_external_emotion(self):
        n=TemporalNPN();e=past_event(1);x=encode(e['interpretation']);x[8:]=.5;old=n.logits(x)[0]
        r=n.update(e);np.testing.assert_allclose(r['targetLogit'],np.clip(old*1.1,-4,4),atol=1e-15)
        self.assertNotEqual(r['before'],r['after'])
    def test_unknown_belief_can_form_personality(self):
        n=TemporalNPN();n.update(past_event(2));self.assertEqual(n.clock,1);self.assertNotEqual(n.episodes[0]['episode']['facts'],n.episodes[0]['episode']['interpretation'])
    def test_correction_keeps_old_record(self):
        n=TemporalNPN();n.update(past_event(2));old=copy.deepcopy(n.episodes[0]);n.update(present_event(2,0,'past-2-0'));self.assertEqual(n.episodes[0],old)
    def test_read_is_time_and_weight_pure(self):
        n=TemporalNPN();n.update(past_event(1));before=n.parameter_hash();t=n.clock;tool_evaluate(n,present_event(0)['interpretation']);self.assertEqual(n.clock,t);self.assertEqual(n.parameter_hash(),before)
    def test_learning_off_has_emotion_without_weight_changes(self):
        n=TemporalNPN(learning=False);before=n.parameter_hash();n.update(past_event(1));n.advance(5);self.assertEqual(n.parameter_hash(),before);self.assertGreater(np.linalg.norm(n.mood),0)
    def test_time_separates_short_and_mood(self):
        n=TemporalNPN(decay='none');n.update(past_event(1));short=np.linalg.norm(n.short);mood=np.linalg.norm(n.mood);h=n.parameter_hash();n.advance(16)
        self.assertLess(np.linalg.norm(n.short),.01*short);self.assertLess(np.linalg.norm(n.mood),mood);self.assertEqual(h,n.parameter_hash())
    def test_reactivation_uses_contemporaneous_interpretation(self):
        n=TemporalNPN();n.update(past_event(2));before=copy.deepcopy(n.episodes[0]);n.reactivate('past-2-0','recall');self.assertEqual(n.episodes[-1]['episode']['interpretation'],before['episode']['interpretation']);self.assertEqual(n.episodes[0],before)
    def test_invalid_episode_does_not_advance(self):
        for mutate in [lambda e:e.update(confidence=float('nan')),lambda e:e.update(emotion='angry'),lambda e:e.update(links=[dict(id='missing',relation='follows')])]:
            n=TemporalNPN();e=past_event(0);mutate(e);h=n.parameter_hash()
            with self.assertRaises(ValueError):n.update(e)
            self.assertEqual(n.parameter_hash(),h);self.assertEqual(n.clock,0)
    def test_duplicates_rejected(self):
        n=TemporalNPN();n.update(past_event(0));h=n.parameter_hash()
        with self.assertRaises(ValueError):n.update(past_event(0))
        self.assertEqual(n.parameter_hash(),h)
    def test_domain_independence_without_global_decay(self):
        n=TemporalNPN(decay='none');n.update(acoustic());h=n.branch_hash('acoustic');n.update(past_event(0));self.assertEqual(n.branch_hash('acoustic'),h)
    def test_save_restores_continuation(self):
        n=TemporalNPN();n.update(past_event(2))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state';n.save(p);m=TemporalNPN.load(p);self.assertEqual(m.parameter_hash(),n.parameter_hash());self.assertEqual(m.clock,n.clock)
            n.update(present_event(2,0,'past-2-0'));m.update(present_event(2,0,'past-2-0'));self.assertEqual(m.parameter_hash(),n.parameter_hash());self.assertEqual(m.episodes,n.episodes)
            with self.assertRaises(FileExistsError):n.save(p)
    def test_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state';TemporalNPN().save(p);a=Path(str(p)+'.npz');a.write_bytes(a.read_bytes()+b'x')
            with self.assertRaises(ValueError):TemporalNPN.load(p)
    def test_long_term_evaluation_survives_history_removal(self):
        n=TemporalNPN();n.update(past_event(1));x=encode(past_event(1)['interpretation'],observed=False);y=n.logits(x);n.episodes=[];n.ticks=[];np.testing.assert_array_equal(n.logits(x),y)
    def test_emotion_channels_can_coexist(self):
        n=TemporalNPN();n.short[5:]=1;n.mood[5:]=1;m=meaning(cue=[.5]*8,outcome=[.5]*8);self.assertGreater(n.evaluate(m)['coactivation'],0)
    def test_decay_preserves_episode_and_centers(self):
        n=TemporalNPN();n.update(past_event(1));old=copy.deepcopy(n.episodes);centers=n.centers.clone();w=n.weights.clone();n.advance(32);self.assertEqual(old,n.episodes);self.assertTrue(torch.equal(centers,n.centers));self.assertLess(float(torch.linalg.norm(n.weights)),float(torch.linalg.norm(w)))
    def test_natural_annotations_are_valid(self):
        for e in natural_cases():validate_episode(e)

if __name__=='__main__':unittest.main()
