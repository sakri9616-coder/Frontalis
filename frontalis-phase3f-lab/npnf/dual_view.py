"""An explicit, separately identified exploratory architecture.

The same learned RBF parameters appraise the cue and the observed scene separately.
Their logits have equal fixed weight. No LLM sentiment or desired reaction is used.
"""
import copy,hashlib,json
from pathlib import Path
import numpy as np
from .core import TemporalNPN,json_write
from .meaning import digest,encode
VIEW='frontalis-temporal-dual50-v1'

class DualViewAgent:
    def __init__(self,n=None,**kwargs):object.__setattr__(self,'n',n or TemporalNPN(**kwargs))
    def __getattr__(self,name):return getattr(self.n,name)
    def __setattr__(self,name,value):
        if name=='n':object.__setattr__(self,name,value)
        else:setattr(self.n,name,value)
    def clone(self):return DualViewAgent(self.n.clone())
    def parameter_hash(self):return digest(dict(architecture=VIEW,weights=self.n.parameter_hash()))
    def evaluate(self,meaning,space='relational'):
        observed=self.n.logits(encode(meaning),space)[0];cue=self.n.logits(encode(meaning,observed=False),space)[0]
        z=.5*observed+.5*cue;long=np.tanh(z)
        combined=np.tanh(z+.3*self.n.short+.2*self.n.mood) if self.n.config['emotional'] else long.copy()
        return dict(longTerm=long.tolist(),shortTerm=self.n.short.tolist(),mood=self.n.mood.tolist(),combined=combined.tolist(),
            stance='approve' if combined[0]>.08 else 'refuse' if combined[0]<-.08 else 'hold',
            valence=float((combined[5]-combined[6])/2),activation=float(np.mean(np.abs(combined[4:]))),
            coactivation=float(min(max(0,combined[5]),max(0,combined[6]))),update=self.n.clock,parameterHash=self.parameter_hash(),
            evaluationView=VIEW,cueAppraisal=np.tanh(cue).tolist(),observedAppraisal=np.tanh(observed).tolist())
    def update(self,e):
        pre=self.evaluate(e['interpretation'],e['space']);old_short=self.n.short.copy();old_mood=self.n.mood.copy();before=self.parameter_hash()
        row=self.n.update(e)
        if self.n.config['emotional']:
            self.n.short=.45*old_short+.55*np.asarray(pre['longTerm']);self.n.mood=.9*old_mood+.1*self.n.short
        # Complete the new record before publication. Never alter an earlier episode.
        row['rawObservedAppraisalBefore']=row['appraisalBefore'];row['appraisalBefore']=pre
        row['before']=before;row['after']=self.parameter_hash();row['evaluationView']=VIEW;row.pop('recordHash')
        row['recordHash']=digest(row);self.n.episodes[-1]=copy.deepcopy(row)
        self.n.ticks[-1].update(shortNorm=float(np.linalg.norm(self.n.short)),moodNorm=float(np.linalg.norm(self.n.mood)))
        return copy.deepcopy(row)
    def reactivate(self,source_id,new_id):
        old=next((r['episode'] for r in self.n.episodes if r['episode']['id']==source_id),None)
        if old is None:raise ValueError('Unknown reactivation source.')
        e=copy.deepcopy(old);e.update(id=new_id,kind='reactivation',confidence=min(.6,old['confidence']),links=[dict(id=source_id,relation='recalls')]);return self.update(e)
    def save(self,prefix):
        manifest=Path(str(prefix)+'.view.json')
        if manifest.exists():raise FileExistsError('Existing evaluation architecture is immutable.')
        self.n.save(prefix)
        m=dict(schema=VIEW,personalityHash=self.parameter_hash(),weightsHash=self.n.parameter_hash(),coreJsonSha256=hashlib.sha256(Path(str(prefix)+'.json').read_bytes()).hexdigest())
        json_write(manifest,dict(metadata=m,metadataSha256=digest(m)))
    @classmethod
    def load(cls,prefix,device='cpu'):
        env=json.loads(Path(str(prefix)+'.view.json').read_text(encoding='utf8'));m=env['metadata']
        if m['schema']!=VIEW or digest(m)!=env['metadataSha256'] or hashlib.sha256(Path(str(prefix)+'.json').read_bytes()).hexdigest()!=m['coreJsonSha256']:raise ValueError('Evaluation architecture integrity mismatch.')
        n=cls(TemporalNPN.load(prefix,device))
        if n.n.parameter_hash()!=m['weightsHash'] or n.parameter_hash()!=m['personalityHash']:raise ValueError('Evaluation architecture parameter mismatch.')
        return n
