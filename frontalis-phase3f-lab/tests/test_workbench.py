import copy,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from playground import server as s

class Workbench(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.p=patch.object(s,'STATES',Path(self.tmp.name));self.p.start();s.init()
    def tearDown(self):self.p.stop();self.tmp.cleanup()
    def body(self,**kwargs):return dict(session=s.SESSION['session'],revision=s.SESSION['revision'],**kwargs)
    def test_saved_generations_restore_and_continue(self):
        s.mutate(self.body(action='preset',group='past',index=2));h=s.NPN.parameter_hash();s.NPN=None;s.restore();self.assertEqual(s.NPN.parameter_hash(),h);self.assertEqual(s.NPN.clock,1)
        s.mutate(self.body(action='preset',group='current',index=2,sourceId='ui-1'));self.assertEqual(s.NPN.episodes[-1]['episode']['kind'],'correction')
    def test_stale_request_is_rejected(self):
        b=self.body(action='elapsed',count=1);s.mutate(b)
        with self.assertRaises(ValueError):s.mutate(b)
        self.assertEqual(s.NPN.clock,1)
    def test_failed_save_does_not_commit_ram(self):
        h=s.NPN.parameter_hash()
        with patch.object(s.TemporalNPN,'save',side_effect=OSError('test only')):
            with self.assertRaises(OSError):s.mutate(self.body(action='preset',group='past',index=0))
        self.assertEqual(s.NPN.parameter_hash(),h);self.assertEqual(s.NPN.clock,0)
    def test_orphan_snapshot_does_not_block_retry(self):
        with patch.object(s,'json_write',side_effect=OSError('test only')):
            with self.assertRaises(OSError):s.mutate(self.body(action='preset',group='past',index=0))
        self.assertEqual(s.NPN.clock,0);s.mutate(self.body(action='preset',group='past',index=0));self.assertEqual(s.NPN.clock,1)
    def test_elapsed_updates_only_time_and_states(self):
        s.mutate(self.body(action='preset',group='past',index=1));old=copy.deepcopy(s.NPN.episodes);s.mutate(self.body(action='elapsed',count=16));self.assertEqual(s.NPN.clock,17);self.assertEqual(s.NPN.episodes,old);self.assertEqual(len(s.TRACE),18)
    def test_unconfirmed_and_unknown_drafts_rejected(self):
        for confirmed in [False,True]:
            with self.assertRaises(ValueError):s.mutate(self.body(action='learn',confirmed=confirmed,draftId='missing'))
        self.assertEqual(s.NPN.clock,0)
    def test_invalid_preset_does_not_change_live_state(self):
        with self.assertRaises(ValueError):s.mutate(self.body(action='preset',group='past',index=4))
        self.assertEqual(s.NPN.clock,0)
    def test_readout_does_not_expose_initial_description(self):
        r=s.result();self.assertNotIn('profile',r);self.assertNotIn('history',r);self.assertNotIn('name',r);self.assertEqual(s.NPN.clock,0)
    def test_dual_architecture_restores_and_remains_explicit(self):
        s.init(evaluation_view='dual50');s.mutate(self.body(action='preset',group='past',index=2));h=s.NPN.parameter_hash();s.NPN=None;s.restore()
        self.assertEqual(s.NPN.parameter_hash(),h);self.assertEqual(s.SESSION['evaluationView'],'dual50');self.assertIn('cueAppraisal',s.result()['scene'])

if __name__=='__main__':unittest.main()
