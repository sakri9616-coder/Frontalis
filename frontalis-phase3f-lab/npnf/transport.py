"""Typed factual transport. A lossless axis-object adapter never invents values."""
import copy
from .meaning import AXES,ACTIONS,ACTORS,validate
def obj(properties):return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)
def enum(values):return dict(type='string',enum=values)
def array(items,description=''):return dict(type='array',items=items,description=description)
EVENT_SCHEMA=obj(dict(action=enum(ACTIONS),actor=enum(ACTORS),target=enum(['self','other','object']),negated=dict(type='boolean'),uncertain=dict(type='boolean'),intent=enum(['benefit','harm','unknown'])))
MEANING_SCHEMA=obj(dict(cueFacts8=array(dict(type='number'),'Exactly 8 values in [0,1]. Unknown is 0.5. Use the specified axis order.'),outcomeFacts8=array(dict(type='number'),'Exactly 8 values in [0,1]. Unknown is 0.5.'),events=array(EVENT_SCHEMA,'One or two events in text order.'),consent=enum(['yes','no','unknown']),trust=enum(['high','low','unknown']),causal=enum(['causes','prevents','unrelated','unknown']),temporal=enum(['after','simultaneous','unknown']),delayed=dict(type='boolean')))
EXTRACTION_SCHEMA=obj(dict(records=array(obj(dict(id=dict(type='string'),facts=MEANING_SCHEMA,interpretation=MEANING_SCHEMA)))))
def normalize_meaning(m):
    m=copy.deepcopy(m);changed=False
    if not isinstance(m,dict):raise ValueError('Expected a factual meaning object.')
    for key in ['cueFacts8','outcomeFacts8']:
        if isinstance(m.get(key),dict):
            if set(m[key])!=set(AXES):raise ValueError('All eight exact factual axis names are required; no filling is permitted.')
            m[key]=[m[key][a] for a in AXES];changed=True
    return validate(m),changed
