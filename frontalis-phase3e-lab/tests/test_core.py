import copy,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from npne.core import Personality,experience,tool_evaluate
from npne.data import training,WIDTHS,context_pairs
from npne.meaning import encode,meaning,event,digest
from tools.evaluate import dispatch
torch.set_num_threads(1)

class Tests(unittest.TestCase):
    def test_counts_and_common_initial_judgment(self):
        m=training(5311,0,1)[0]['meaning'];expected=None
        for rep in WIDTHS:
            for cap,w in enumerate(WIDTHS[rep]):
                n=Personality(width=w,representation=rep);self.assertEqual(n.count(),2*(n.dim+8)*w+14)
                y=n.evaluate_batch([encode(m,rep)])[0]
                if expected is None:expected=y
                np.testing.assert_array_equal(y,expected)
    def test_dimensions_and_no_outcome_leak(self):
        m=meaning([event(),event('deceive')],causal='causes',delayed=True,temporal='after');a=encode(m,observed=False)
        b=copy.deepcopy(m);b['events'][1]=event('explore');b['causal']='prevents';b['delayed']=False;b['temporal']='simultaneous';b['outcomeFacts8']=[1]*8
        np.testing.assert_array_equal(a,encode(b,observed=False));self.assertFalse(np.array_equal(encode(m),encode(b)))
    def test_context_information_is_added(self):
        a,b=context_pairs(count=1)[0];np.testing.assert_array_equal(encode(a,'raw8'),encode(b,'raw8'));self.assertFalse(np.array_equal(encode(a),encode(b)))
    def test_subjective_meaning_forbidden(self):
        m=meaning();m['emotion']='happy'
        with self.assertRaises(ValueError):encode(m)
    def test_uncertain_learning_rejected(self):
        r=training(5311,0,1)[0];r['meaning']['events'][0]['uncertain']=True
        with self.assertRaises(ValueError):experience(r,'raw8')
    def test_old_evaluation_supplies_target(self):
        n=Personality();e=experience(training(5311,0,1)[0],'raw8');old=n.evaluate(e['observed'])['channels'];r=n.update(e)
        expected=np.clip(1.25*np.arctanh(list(old.values())),-6,6);np.testing.assert_allclose(r['targetLogit'],expected,atol=1e-14)
        self.assertNotEqual(r['before'],r['after'])
    def test_same_target_different_loss(self):
        e=experience(training(5311,0,1)[0],'raw8');a=Personality(loss='bounded_signal');b=Personality(loss='bounded_logit')
        np.testing.assert_array_equal(a.update(e)['targetLogit'],b.update(e)['targetLogit'])
    def test_branch_hold(self):
        n=Personality();before=n.branch_hash('acoustic');n.update(experience(training(5311,0,1)[0],'raw8'));self.assertEqual(n.branch_hash('acoustic'),before)
    def test_learning_off(self):
        n=Personality(learning=False);before=n.parameter_hash();n.update(experience(training(5311,0,1)[0],'raw8'));self.assertEqual(n.parameter_hash(),before)
    def test_neutral(self):
        n=Personality();before=n.parameter_hash();e=dict(id='zero',cue=[.5]*8,observed=[.5]*8,confirmed=True,origin='synthetic',space='relational',sourceHash=digest('zero'));self.assertTrue(n.update(e)['noInformation']);self.assertEqual(n.parameter_hash(),before)
    def test_rejected_updates_transactional(self):
        n=Personality();e=experience(training(5311,0,1)[0],'raw8');before=n.parameter_hash()
        for field,value in [('target',[1]*7),('confirmed',False),('observed',[float('nan')]*8),('space','bad')]:
            bad=copy.deepcopy(e);bad[field]=value
            with self.assertRaises(ValueError):n.update(bad)
            self.assertEqual(n.parameter_hash(),before);self.assertEqual(n.generation,0)
    def test_duplicate(self):
        n=Personality();e=experience(training(5311,0,1)[0],'raw8');n.update(e);before=n.parameter_hash()
        with self.assertRaises(ValueError):n.update(e)
        self.assertEqual(before,n.parameter_hash())
    def test_stability_rejection_retains_state(self):
        n=Personality();e=experience(training(5311,0,1)[0],'raw8');before=n.parameter_hash();n.lr=1e9
        with self.assertRaises(ValueError):n.update(e)
        self.assertEqual(before,n.parameter_hash());self.assertEqual(n.history,[])
    def test_save_restore_resume_rollback(self):
        n=Personality(width=5,representation='semantic44');events=training(5311,0,3)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'v0';n.save(p);n.update(experience(events[0],'semantic44'));q=Path(d)/'v1';n.save(q);r=Personality.load(q)
            self.assertEqual(n.parameter_hash(),r.parameter_hash());n.update(experience(events[1],'semantic44'));r.update(experience(events[1],'semantic44'));self.assertEqual(n.parameter_hash(),r.parameter_hash());self.assertEqual(Personality.load(p).generation,0)
            with self.assertRaises(FileExistsError):n.save(q)
    def test_tampered_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'s';Personality().save(p);j=Path(str(p)+'.json');m=json.loads(j.read_text());m['metadata']['generation']=99;j.write_text(json.dumps(m))
            with self.assertRaises(ValueError):Personality.load(p)
    def test_tampered_arrays(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'s';Personality().save(p);a=Path(str(p)+'.npz');a.write_bytes(a.read_bytes()+b'x')
            with self.assertRaises(ValueError):Personality.load(p)
    def test_tool_distinguishes_scene_and_target(self):
        n=Personality();m=meaning();a=meaning(outcome=[1]*8);r=tool_evaluate(n,m,[dict(id='A',meaning=a)]);self.assertEqual(r['selection'],'A');self.assertIsNotNone(r['selectedTargetAppraisal']);self.assertNotEqual(r['sceneAppraisal']['channels'],r['selectedTargetAppraisal']['channels'])
    def test_tool_read_only(self):
        n=Personality();before=n.parameter_hash();tool_evaluate(n,meaning());self.assertEqual(n.parameter_hash(),before);self.assertEqual(n.generation,0)
    def test_path_and_training_tool_forbidden(self):
        for id in ['../../secret','C:/x','stress-5311','main-5311-raw8-c0-bounded_logit-h0/../x']:
            with self.assertRaises(ValueError):dispatch(dict(modelId=id,meaning=meaning(),options=[]))
        with self.assertRaises(ValueError):dispatch(dict(modelId='ok',meaning=meaning(),options=[],target=[1]*7))
    def test_invalid_booleans_and_ranges(self):
        m=meaning();m['events'][0]['negated']=1
        with self.assertRaises(ValueError):encode(m)
        for x in [[True]*8,[1.1]*8,[-.1]*8,['.5']*8,[.5]*7]:
            with self.assertRaises(ValueError):Personality().evaluate(x)

if __name__=='__main__':unittest.main()
