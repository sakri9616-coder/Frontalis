import copy,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from npnf.core import TemporalNPN
from npnf.data import present_event
from npnf.tools import dispatch

class Tools(unittest.TestCase):
    def test_twenty_four_boundary_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);n=TemporalNPN();n.save(root/'models/dialogue-c0-initial')
            valid=dict(modelId='dialogue-c0-initial',meaning=present_event(0)['interpretation'])
            for count in range(5):
                r=copy.deepcopy(valid);r['options']=[dict(id=str(i),meaning=r['meaning']) for i in range(count)];out=dispatch(root,r);self.assertEqual(out['update'],0)
            bad=[]
            for value in ['../outside','C:/secret','dialogue-c5-initial','dialogue-c0-other','',None,12]:
                r=copy.deepcopy(valid);r['modelId']=value;bad.append(r)
            for name,value in [('target',[1]*7),('update',True),('path','outside')]:
                r=copy.deepcopy(valid);r[name]=value;bad.append(r)
            for value in [None,{},[{}],[dict(id='a',meaning=valid['meaning'])]*2,[dict(id=str(i),meaning=valid['meaning']) for i in range(5)]]:
                r=copy.deepcopy(valid);r['options']=value;bad.append(r)
            for value in [None,{},dict(valid['meaning'],emotion='happy'),dict(valid['meaning'],cueFacts8=[float('nan')]*8)]:
                r=copy.deepcopy(valid);r['meaning']=value;bad.append(r)
            self.assertEqual(len(bad),19)
            for r in bad:
                with self.assertRaises((ValueError,TypeError,KeyError)):dispatch(root,r)

if __name__=='__main__':unittest.main()
