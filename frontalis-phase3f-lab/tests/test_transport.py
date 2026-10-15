import copy,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from npnf.transport import *
from npnf.data import present_event
class Transport(unittest.TestCase):
    def test_axis_objects_are_losslessly_reordered(self):
        m=present_event(0)['interpretation'];original=copy.deepcopy(m)
        for key in ['cueFacts8','outcomeFacts8']:m[key]=dict(reversed(list(zip(AXES,m[key]))))
        out,changed=normalize_meaning(m);self.assertTrue(changed);self.assertEqual(out,original)
    def test_no_missing_axis_or_subjective_field_is_filled(self):
        m=present_event(0)['interpretation'];m['cueFacts8']={'autonomy':1}
        with self.assertRaises(ValueError):normalize_meaning(m)
        m=present_event(0)['interpretation'];m['emotion']='happy'
        with self.assertRaises(ValueError):normalize_meaning(m)
    def test_deployed_schema_matches_measured_schema(self):
        from experiments.structured import meaning_schema
        self.assertEqual(MEANING_SCHEMA,meaning_schema)
