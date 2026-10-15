# Minimal numpy-backed stand-in for the torch calls used by npnf.core (analysis only).
import numpy as np
float64=np.float64
class T(np.ndarray):
    def cpu(self): return self
    def numpy(self): return np.asarray(self)
    def clone(self): return self.copy()
    def sum(self,*a,keepdim=False,**k):
        if keepdim: k['keepdims']=True
        return np.ndarray.sum(self,*a,**k)
def _t(x): return np.array(x,dtype=np.float64).view(T)
def as_tensor(x,dtype=None,device=None): return _t(x)
def exp(x): return np.exp(x)
def tanh(x): return np.tanh(x)
def sqrt(x): return np.sqrt(x)
def abs(x): return np.abs(x)
def max(x): return np.max(x)
def argmin(x): return np.argmin(x)
def isfinite(x): return np.isfinite(x)
def clamp(x,lo,hi): return np.clip(x,lo,hi)
def ones_like(x): return np.ones_like(x)
class linalg:
    norm=staticmethod(np.linalg.norm)
def set_num_threads(n): pass
