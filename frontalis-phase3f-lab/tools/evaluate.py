"""JSON-in/JSON-out read-only NPN tool for external LLM callers."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from npnf.tools import dispatch
if __name__=='__main__':
    try:
        raw=sys.stdin.read(65537)
        if len(raw)>65536:raise ValueError('Request too large.')
        result=dispatch(ROOT,json.loads(raw));print(json.dumps(dict(ok=True,result=result),ensure_ascii=False,allow_nan=False))
    except (ValueError,TypeError,KeyError,FileNotFoundError) as e:
        print(json.dumps(dict(ok=False,error=str(e)),ensure_ascii=False));sys.exit(2)
