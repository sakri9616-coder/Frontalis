"""Phase 3-D architecture/rule extended to explicit factual semantic input.

No text, persona labels, rewards, or LLM emotional judgments enter optimization.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from .meaning import CHANNELS,digest,vector

SPACES=('relational','acoustic')
SCHEMA='frontalis-meaning-npn-v1'

class Personality:
    def __init__(self,seed=5311,width=16,representation='raw8',loss='bounded_logit',device='cpu',learning=True):
        if type(seed) is not int or type(width) is not int or not 1<=width<=31250:raise ValueError('Invalid architecture.')
        if representation not in ['raw8','semantic44'] or loss not in ['bounded_signal','bounded_logit'] or type(learning) is not bool:raise ValueError('Invalid condition.')
        self.config=dict(seed=seed,width=width,representation=representation,loss=loss,learning=learning)
        self.device=str(device);self.dim=8 if representation=='raw8' else 44
        self.base=torch.tensor(np.random.default_rng(seed).normal(0,.16,(8,7)),dtype=torch.float64,device=device)
        h=np.random.default_rng(seed+100003).normal(0,.28*math.sqrt(8/self.dim),(self.dim,width))
        self.branches={s:{'w1':self.t(h,True),'b1':self.t(np.zeros(width),True),'w2':self.t(np.zeros((width,7)),True),'b2':self.t(np.zeros(7),True)} for s in SPACES}
        self.scale=16/width;self.replay={s:[] for s in SPACES};self.history=[];self.generation=0
        self.lr=.025;self.epochs=8;self.replay_limit=64;self._hash=None

    def t(self,v,grad=False):return torch.tensor(v,dtype=torch.float64,device=self.device,requires_grad=grad)
    def count(self):return sum(v.numel() for b in self.branches.values() for v in b.values())
    def arrays(self):return {'base':self.base.detach().cpu().numpy().copy(),**{s+'__'+k:v.detach().cpu().numpy().copy() for s,b in self.branches.items() for k,v in b.items()}}
    def parameter_hash(self):
        if self._hash is None:
            h=hashlib.sha256(digest({'schema':SCHEMA,'representation':self.config['representation'],'scale':self.scale}).encode())
            for k,v in sorted(self.arrays().items()):h.update(k.encode());h.update(str(v.shape).encode());h.update(np.ascontiguousarray(v,dtype='<f8').tobytes())
            self._hash=h.hexdigest()
        return self._hash
    def branch_hash(self,s):
        h=hashlib.sha256()
        for k,v in self.branches[s].items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    def forward(self,x,space='relational',branch=None):
        if space not in SPACES:raise ValueError('Unknown branch.')
        b=self.branches[space] if branch is None else branch
        hidden=torch.tanh(x@b['w1']+b['b1'])
        z=x[:,:8]@self.base+self.scale*(hidden@b['w2'])+b['b2']
        return torch.tanh(z),z,hidden
    def evaluate_batch(self,x,space='relational'):
        a=np.asarray(x)
        if a.ndim!=2 or a.shape[1]!=self.dim:raise ValueError('Wrong factual input width.')
        for row in a:vector(row,self.dim)
        with torch.no_grad():return self.forward(self.t(2*a-1),space)[0].cpu().numpy()
    def evaluate(self,x,space='relational'):
        raw=self.evaluate_batch([vector(x,self.dim)],space)[0]
        c=dict(zip(CHANNELS,raw.tolist()));a=c['affinity']
        return dict(channels=c,signals=dict(preference=a,initiative=max(0,a),dislike=max(0,-a),resistance=max(0,-a),curiosity=max(0,c['curiosity']),value=c['value'],anger=max(0,c['anger']),joy=max(0,c['joy']),sadness=max(0,c['sadness'])),stance='approve' if a>.08 else 'refuse' if a<-.08 else 'hold',generation=self.generation,parameterHash=self.parameter_hash())
    def stable(self,branches=None):
        b=self.branches if branches is None else branches
        norm=math.sqrt(sum(float((v.detach()**2).sum()) for p in b.values() for v in p.values()))
        if not math.isfinite(norm) or any(not bool(torch.isfinite(v).all()) for p in b.values() for v in p.values()) or norm>50*math.sqrt(self.count()/526):raise ValueError('Numerical stability gate failed; no commit.')
        return norm
    def _fit(self,space,labels,epochs,lr):
        x=self.t(2*np.asarray([r['x'] for r in labels])-1);z_target=self.t([r['z'] for r in labels])
        y_target=torch.tanh(z_target)
        b={k:v.detach().clone().requires_grad_(True) for k,v in self.branches[space].items()}
        m={k:torch.zeros_like(v) for k,v in b.items()};v={k:torch.zeros_like(p) for k,p in b.items()}
        stats={}
        for epoch in range(1,epochs+1):
            y,z,h=self.forward(x,space,b)
            objective=((z-z_target)**2).mean() if self.config['loss']=='bounded_logit' else ((y-y_target)**2).mean()
            grads=torch.autograd.grad(objective,list(b.values()))
            norm=math.sqrt(sum(float((g.detach()**2).sum()) for g in grads))
            if not math.isfinite(float(objective.detach())) or not math.isfinite(norm):raise ValueError('Nonfinite objective/gradient; no commit.')
            stats=dict(loss=float(objective.detach()),gradientNorm=norm,outputDerivative=float((1-y.detach()**2).mean()),hiddenDerivative=float((1-h.detach()**2).mean()))
            with torch.no_grad():
                for (k,p),g in zip(b.items(),grads):
                    g=g*min(1,5/(norm+1e-12));m[k].mul_(.9).add_(g,alpha=.1);v[k].mul_(.999).addcmul_(g,g,value=.001)
                    p.addcdiv_(m[k]/(1-.9**epoch),torch.sqrt(v[k]/(1-.999**epoch))+1e-8,value=-lr)
        candidate=dict(self.branches);candidate[space]=b;self.stable(candidate)
        return b,stats
    def update(self,event):
        required={'id','cue','observed','confirmed','origin','space','sourceHash'}
        if not isinstance(event,dict) or set(event)!=required or event['confirmed'] is not True or event['origin']!='synthetic':raise ValueError('Only confirmed synthetic facts, no subjective teacher.')
        if not isinstance(event['id'],str) or not event['id'] or any(r['id']==event['id'] for r in self.history):raise ValueError('Duplicate/empty experience ID.')
        if not isinstance(event['sourceHash'],str) or len(event['sourceHash'])!=64:raise ValueError('Source meaning hash required.')
        s=event['space']
        if s not in SPACES:raise ValueError('Unknown branch.')
        cue=vector(event['cue'],self.dim);obs=vector(event['observed'],self.dim)
        self.stable();before=self.parameter_hash()
        with torch.no_grad():
            old_y,old_z,_=self.forward(self.t(2*obs[None,:]-1),s)
            target=torch.clamp(1.25*old_z[0],-6,6).cpu().numpy()
        replay=(self.replay[s]+[dict(x=x.tolist(),z=target.tolist(),sourceStateHash=before,eventId=event['id']) for x in [cue,obs]])[-self.replay_limit:]
        noinfo=bool(np.all(cue==.5) and np.all(obs==.5))
        stats={};branch=self.branches[s]
        if self.config['learning'] and not noinfo:branch,stats=self._fit(s,replay,self.epochs,self.lr)
        self.branches[s]=branch
        if self.config['learning'] and not noinfo:self.replay[s]=replay
        self._hash=None;self.generation+=1
        row=dict(id=event['id'],sourceHash=event['sourceHash'],before=before,after=self.parameter_hash(),oldOutcome=old_y[0].cpu().numpy().tolist(),targetLogit=target.tolist(),noInformation=noinfo,training=stats)
        self.history.append(row)
        return copy.deepcopy(row)
    def fit_diagnostic(self,x,y,epochs=120):
        """Fixed-label capacity diagnostic. Never exposed as personality update tool."""
        a=np.asarray(x);targets=np.asarray(y)
        if targets.shape!=(len(a),7) or not np.isfinite(targets).all() or np.max(np.abs(targets))>1:raise ValueError('Invalid diagnostic labels.')
        for row in a:vector(row,self.dim)
        labels=[dict(x=p.tolist(),z=np.arctanh(np.clip(t,-.999999,.999999)).tolist()) for p,t in zip(a,targets)]
        self.branches['relational'],stats=self._fit('relational',labels,epochs,self.lr);self._hash=None
        return stats
    def save(self,prefix):
        p=Path(prefix);p.parent.mkdir(parents=True,exist_ok=True);a=Path(str(p)+'.npz');j=Path(str(p)+'.json')
        if a.exists() or j.exists():raise FileExistsError('Existing generation is immutable.')
        self.stable()
        with a.open('xb') as f:np.savez_compressed(f,**self.arrays())
        meta=dict(schema=SCHEMA,config=self.config,generation=self.generation,history=self.history,replay=self.replay,parameterHash=self.parameter_hash(),arraySha256=hashlib.sha256(a.read_bytes()).hexdigest(),lr=self.lr,epochs=self.epochs,replayLimit=self.replay_limit)
        try:
            with j.open('x',encoding='utf8') as f:json.dump(dict(metadata=meta,metadataSha256=digest(meta)),f,ensure_ascii=False,allow_nan=False)
        except Exception:a.unlink(missing_ok=True);raise
    @classmethod
    def load(cls,prefix,device='cpu'):
        p=Path(prefix);env=json.loads(Path(str(p)+'.json').read_text(encoding='utf8'));m=env['metadata'];a=Path(str(p)+'.npz')
        if m['schema']!=SCHEMA or digest(m)!=env['metadataSha256'] or hashlib.sha256(a.read_bytes()).hexdigest()!=m['arraySha256']:raise ValueError('Snapshot integrity mismatch.')
        n=cls(**m['config'],device=device);expected=n.arrays()
        with np.load(a,allow_pickle=False) as data:
            if set(data.files)!=set(expected):raise ValueError('Invalid parameter keys.')
            for k,x in data.items():
                if x.shape!=expected[k].shape or x.dtype.kind!='f' or not np.isfinite(x).all():raise ValueError('Invalid parameter shape/value.')
                if k=='base':n.base=n.t(x)
                else:s,name=k.split('__');n.branches[s][name]=n.t(x,True)
        n.history=m['history'];n.replay=m['replay'];n.generation=m['generation'];n.lr=m['lr'];n.epochs=m['epochs'];n.replay_limit=m['replayLimit']
        n.stable()
        if n.parameter_hash()!=m['parameterHash']:raise ValueError('Parameter hash mismatch.')
        return n

def experience(row,representation):
    from .meaning import encode,validate
    m=validate(row['meaning'])
    if any(e['uncertain'] for e in m['events']):raise ValueError('Uncertain occurrence is not a confirmed learning observation.')
    return dict(id=row['id'],cue=encode(m,representation,False).tolist(),observed=encode(m,representation,True).tolist(),space=row['space'],confirmed=row['confirmed'],origin=row['origin'],sourceHash=digest({'text':row['text'],'meaning':m}))

def tool_evaluate(n,meaning,options=None):
    """Read-only appraisal; scene and selected target remain separate."""
    from .meaning import encode
    scene=n.evaluate(encode(meaning,n.config['representation']))
    candidates=[]
    for option in options or []:
        r=n.evaluate(encode(option['meaning'],n.config['representation']))
        candidates.append(dict(id=option['id'],appraisal=r))
    selected=max(candidates,key=lambda x:x['appraisal']['channels']['value']) if candidates else None
    return dict(sceneAppraisal=scene,candidateAppraisals=candidates,selection=selected['id'] if selected else 'hold',selectedTargetAppraisal=selected['appraisal'] if selected else None,parameterHash=n.parameter_hash())
