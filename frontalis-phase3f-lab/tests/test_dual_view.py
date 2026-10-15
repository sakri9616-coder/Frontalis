import copy,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from npnf.core import *
from npnf.dual_view import *
from npnf.data import *

class DualTests(unittest.TestCase):
    def test_same_weights_different_explicit_architecture(self):
        a=TemporalNPN();b=DualViewAgent()
        for i in range(4):e=past_event(1,i);a.update(e);b.update(e)
        self.assertEqual(a.parameter_hash(),b.n.parameter_hash());self.assertNotEqual(a.parameter_hash(),b.parameter_hash())
    def test_records_keep_actual_dual_appraisal_and_old_episode(self):
        n=DualViewAgent();n.update(past_event(2));old=copy.deepcopy(n.episodes[0]);m=present_event(2,0,'past-2-0');pre=n.evaluate(m['interpretation']);row=n.update(m)
        self.assertEqual(row['appraisalBefore'],pre);self.assertEqual(n.episodes[0],old)
    def test_snapshot_requires_view_and_preserves_continuation(self):
        n=DualViewAgent();n.update(past_event(1))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state';n.save(p);m=DualViewAgent.load(p);self.assertEqual(n.parameter_hash(),m.parameter_hash());n.advance(8);m.advance(8);self.assertEqual(n.evaluate(present_event(0)['interpretation']),m.evaluate(present_event(0)['interpretation']))
            Path(str(p)+'.view.json').unlink()
            with self.assertRaises(FileNotFoundError):DualViewAgent.load(p)
