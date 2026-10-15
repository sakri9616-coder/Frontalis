import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

class HostDispatchTests(unittest.TestCase):
    def invoke(self,state,calls):
        return subprocess.run([sys.executable,str(ROOT/'tools/dispatch.py'),str(state)],
            input=json.dumps({'calls':calls}),text=True,capture_output=True,
            env={**os.environ,'PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1'},timeout=15)
    def test_wrapped_state_readonly_and_repeat_identical(self):
        state=ROOT/'models/nonlinear/seed-4311/nn-p0.json'
        before=state.read_bytes()
        call={'name':'frontalis_evaluate','arguments':{'query_id':'case-0','space':'relational','features':[.5]*8,'basis':'affinity'}}
        first=self.invoke(state,[call]);second=self.invoke(state,[call])
        self.assertEqual(first.returncode,0,first.stderr)
        self.assertEqual(first.stdout,second.stdout)
        self.assertEqual(state.read_bytes(),before)
        self.assertEqual(json.loads(first.stdout)['dispatchMode'],'host-json')
    def test_caller_cannot_choose_external_snapshot(self):
        result=self.invoke(ROOT/'reports/live-inputs.json',[])
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Research snapshot only',result.stderr)
    def test_unknown_tool_and_extra_argument_rejected(self):
        state=ROOT/'models/nonlinear/seed-4311/nn-p0.json';before=state.read_bytes()
        for call in [{'name':'update','arguments':{}},{'name':'frontalis_evaluate','arguments':{'path':'private'}}]:
            self.assertNotEqual(self.invoke(state,[call]).returncode,0)
        self.assertEqual(state.read_bytes(),before)
    def test_duplicate_query_rejected(self):
        state=ROOT/'models/nonlinear/seed-4311/nn-p0.json'
        call={'name':'frontalis_evaluate','arguments':{'query_id':'case-0','space':'relational','features':[.5]*8}}
        result=self.invoke(state,[call,call]);self.assertNotEqual(result.returncode,0)
        self.assertIn('Duplicate query',result.stderr)

if __name__=='__main__':unittest.main()
