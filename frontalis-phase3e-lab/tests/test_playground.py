import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from playground import server
from npne.meaning import meaning,event

class Workbench(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.store=patch.object(server,'STORE',Path(self.temp.name));self.store.start();server.SESSIONS.clear()
        self.state=server.dispatch('/api/new',dict(representation='semantic44',capacityIndex=0));self.id=self.state['sessionId']
    def tearDown(self):self.store.stop();self.temp.cleanup();server.SESSIONS.clear()
    def interpret(self,m=None):
        with patch.object(server,'interpretation',return_value=(m or meaning([event()],outcome=[1,.1,.7,.1,1,.8,.2,.6]),{'model':'test-only'})):
            return server.dispatch('/api/interpret',dict(sessionId=self.id,text='確認済みの架空経験。'))
    def test_create_and_resume_exact(self):
        h=self.state['parameterHash'];server.SESSIONS.clear();self.assertEqual(server.dispatch('/api/resume',{'sessionId':self.id})['parameterHash'],h)
    def test_interpret_and_expression_do_not_learn(self):
        h=self.state['parameterHash'];p=self.interpret()
        with patch.object(server,'haiku',return_value=('テスト用の表出。',{})):
            r=server.dispatch('/api/speak',dict(sessionId=self.id,pendingId=p['pendingId']))
        self.assertTrue(r['readOnlyExact']);self.assertEqual(r['state']['parameterHash'],h);self.assertEqual(r['state']['generation'],0)
    def test_learning_changes_state_and_old_generation_remains(self):
        p=self.interpret();r=server.dispatch('/api/learn',dict(sessionId=self.id,pendingId=p['pendingId'],confirmed=True,expectedHash=self.state['parameterHash']))
        self.assertNotEqual(r['state']['parameterHash'],self.state['parameterHash']);self.assertEqual(len(r['state']['versions']),2)
        old=server.dispatch('/api/rollback',dict(sessionId=self.id,prefix=self.state['prefix']));self.assertEqual(old['parameterHash'],self.state['parameterHash'])
    def test_unconfirmed_and_stale_requests_rejected(self):
        p=self.interpret()
        for confirmed,h in [(False,self.state['parameterHash']),(True,'wrong')]:
            with self.assertRaises(ValueError):server.dispatch('/api/learn',dict(sessionId=self.id,pendingId=p['pendingId'],confirmed=confirmed,expectedHash=h))
        self.assertEqual(server.state(self.id)['parameterHash'],self.state['parameterHash'])
    def test_no_external_subjective_teacher(self):
        p=self.interpret();bad=copy.deepcopy(p['meaning']);bad['emotion']='happy'
        with self.assertRaises(ValueError):server.dispatch('/api/correct',dict(sessionId=self.id,pendingId=p['pendingId'],meaning=bad))
    def test_uncertain_occurrence_rejected(self):
        p=self.interpret(meaning([event(uncertain=True)]))
        with self.assertRaises(ValueError):server.dispatch('/api/learn',dict(sessionId=self.id,pendingId=p['pendingId'],confirmed=True,expectedHash=self.state['parameterHash']))
    def test_duplicate_experience_rejected(self):
        p=self.interpret();b=dict(sessionId=self.id,pendingId=p['pendingId'],confirmed=True,expectedHash=self.state['parameterHash']);r=server.dispatch('/api/learn',b);b['expectedHash']=r['state']['parameterHash']
        with self.assertRaises(ValueError):server.dispatch('/api/learn',b)
    def test_save_failure_does_not_commit_to_live_personality(self):
        p=self.interpret()
        with patch.object(server.Personality,'save',side_effect=OSError('test-only')):
            with self.assertRaises(OSError):server.dispatch('/api/learn',dict(sessionId=self.id,pendingId=p['pendingId'],confirmed=True,expectedHash=self.state['parameterHash']))
        self.assertEqual(server.state(self.id)['parameterHash'],self.state['parameterHash'])
    def test_path_injection_rejected(self):
        for id in ['../../secret','C:/secret','a'*31]:
            with self.assertRaises(ValueError):server.folder(id)
        with self.assertRaises(ValueError):server.dispatch('/api/rollback',dict(sessionId=self.id,prefix='../file'))

    def test_exploratory_view_preserved_and_only_outcome_context_changed(self):
        from npne.learning import prepare_experience
        from npne.data import training
        from npne.core import experience
        r=training(5311,0,1)[0];standard=experience(r,'semantic44');self.assertEqual(prepare_experience(r,'semantic44'),standard)
        e=prepare_experience(r,'semantic44','facts8-neutral-context');self.assertEqual(e['cue'],standard['cue']);self.assertEqual(e['observed'][:8],standard['observed'][:8]);self.assertEqual(e['observed'][8:],[.5]*36);self.assertNotEqual(e['sourceHash'],standard['sourceHash'])
        s=server.dispatch('/api/new',dict(representation='semantic44',capacityIndex=0,outcomeView='facts8-neutral-context'));server.SESSIONS.clear();self.assertEqual(server.state(s['sessionId'])['outcomeView'],'facts8-neutral-context')

if __name__=='__main__':unittest.main()
