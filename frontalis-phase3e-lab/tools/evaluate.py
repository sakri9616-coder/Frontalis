"""Read-only stdin JSON tool. Arbitrary file paths/training parameters rejected."""
import json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from npne.core import Personality,tool_evaluate

def dispatch(request):
    if not isinstance(request,dict) or set(request)!={'modelId','meaning','options'}:raise ValueError('Only modelId/meaning/options supported.')
    id=request['modelId']
    if not isinstance(id,str) or not re.fullmatch(r'main-531[123]-(raw8|semantic44)-c[0-4]-bounded_(signal|logit)-h[0-3]',id):raise ValueError('Only registered read-only model IDs.')
    options=request['options']
    if not isinstance(options,list) or len(options)>4:raise ValueError('At most four factual candidates.')
    if any(not isinstance(o,dict) or set(o)!={'id','meaning'} or not isinstance(o['id'],str) or not o['id'] for o in options) or len({o['id'] for o in options})!=len(options):raise ValueError('Unique candidate IDs and factual meanings required.')
    n=Personality.load(ROOT/'models'/id)
    return tool_evaluate(n,request['meaning'],options)

if __name__=='__main__':
    try:print(json.dumps(dispatch(json.load(sys.stdin)),ensure_ascii=False,allow_nan=False))
    except (ValueError,FileNotFoundError,KeyError,TypeError) as e:print(json.dumps({'error':str(e)},ensure_ascii=False));sys.exit(1)
