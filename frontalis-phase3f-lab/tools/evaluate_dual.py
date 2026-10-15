import json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from npnf.core import tool_evaluate
from npnf.dual_view import DualViewAgent
from npnf.meaning import validate
if __name__=='__main__':
    try:
        raw=sys.stdin.read(65537)
        if len(raw)>65536:raise ValueError('Request too large.')
        r=json.loads(raw)
        if not isinstance(r,dict) or not {'modelId','meaning'}<=set(r) or set(r)-{'modelId','meaning','options'}:raise ValueError('Unexpected request fields.')
        if not isinstance(r['modelId'],str) or not re.fullmatch(r'dual-c[0-4]-(initial|cooperation|misunderstanding|corrected)',r['modelId']):raise ValueError('Unknown dual-view model ID.')
        validate(r['meaning']);options=r.get('options',[])
        if not isinstance(options,list) or len(options)>4:raise ValueError('At most four options.')
        seen=set()
        for o in options:
            if set(o)!={'id','meaning'} or not isinstance(o['id'],str) or not o['id'] or o['id'] in seen:raise ValueError('Invalid option ID.')
            seen.add(o['id']);validate(o['meaning'])
        n=DualViewAgent.load(ROOT/'models'/r['modelId']);print(json.dumps(dict(ok=True,result=tool_evaluate(n,r['meaning'],options)),ensure_ascii=False,allow_nan=False))
    except (ValueError,TypeError,KeyError,FileNotFoundError) as e:
        print(json.dumps(dict(ok=False,error=str(e)),ensure_ascii=False));sys.exit(2)
