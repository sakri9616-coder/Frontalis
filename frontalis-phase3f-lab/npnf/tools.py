"""Strict read-only model boundary. No arbitrary paths or subjective teachers."""
import re
from pathlib import Path
from .core import TemporalNPN,tool_evaluate
from .meaning import validate

def dispatch(root,request):
    if not isinstance(request,dict) or not {'modelId','meaning'}<=set(request) or set(request)-{'modelId','meaning','options'}:raise ValueError('Unexpected tool request fields.')
    if not isinstance(request['modelId'],str) or not re.fullmatch(r'dialogue-c[0-4]-(initial|cooperation|misunderstanding|corrected)',request['modelId']):raise ValueError('Unknown model ID.')
    validate(request['meaning']);options=request.get('options',[])
    if not isinstance(options,list) or len(options)>4:raise ValueError('At most four options.')
    ids=set()
    for o in options:
        if not isinstance(o,dict) or set(o)!={'id','meaning'} or not isinstance(o['id'],str) or not o['id'] or o['id'] in ids:raise ValueError('Invalid option ID.')
        ids.add(o['id']);validate(o['meaning'])
    n=TemporalNPN.load(Path(root)/'models'/request['modelId']);return tool_evaluate(n,request['meaning'],options)
