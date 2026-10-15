import copy
import unittest
from npnd.core import Personality
from tools.evaluate import decision

class ToolTest(unittest.TestCase):
    def setUp(self):
        self.model=Personality();self.args={'query_id':'case-0','space':'relational','features':[.6]*8,'basis':'value'}
    def test_read_only(self):
        before=self.model.parameter_hash();out=decision(self.model,self.args)
        self.assertEqual(before,out['parameterHash']);self.assertEqual(before,self.model.parameter_hash())
        self.assertEqual(out['predictionGuard'],[.5]*8)
    def test_choices_can_hold(self):
        self.args['choices']=[{'id':'one','label':'一つ目','features':[.5]*8},{'id':'two','label':'二つ目','features':[.5]*8}]
        self.assertEqual(decision(self.model,self.args)['selection'],'hold')
    def test_unknown_args_rejected(self):
        self.args['statePath']='anything'
        with self.assertRaises(ValueError):decision(self.model,self.args)
    def test_prediction_request_rejected(self):
        self.args['target']=[1]*7
        with self.assertRaises(ValueError):decision(self.model,self.args)
    def test_duplicate_choices_rejected(self):
        self.args['choices']=[{'id':'one','label':'同じ','features':[.5]*8}]*2
        with self.assertRaises(ValueError):decision(self.model,self.args)
    def test_bad_features_rejected(self):
        for bad in ([True]*8,[float('nan')]*8,[1.1]*8,[.5]*7):
            self.args['features']=bad
            with self.assertRaises(ValueError):decision(self.model,self.args)
    def test_bad_basis_rejected(self):
        self.args['basis']='moralTruth'
        with self.assertRaises(ValueError):decision(self.model,self.args)
    def test_negative_emotion_intensity_impossible(self):
        out=decision(self.model,self.args)
        self.assertTrue(all(out['signals'][k]>=0 for k in ('anger','joy','sadness','initiative','dislike','resistance')))

if __name__=='__main__':unittest.main()
