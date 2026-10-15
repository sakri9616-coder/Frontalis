"""Host-dispatched JSON tool calls for CLIs without usable native MCP."""
import json
import sys
from pathlib import Path
from evaluate import decision,ROOT
from npn.store import load,validate

def main():
    path=Path(sys.argv[1]).resolve()
    if not path.is_relative_to(ROOT/'models'):raise ValueError('Research snapshot only.')
    value=json.loads(path.read_text(encoding='utf8'))
    state=load(path) if 'payload' in value else value;validate(state)
    calls=json.load(sys.stdin)['calls']
    if not isinstance(calls,list) or not 1<=len(calls)<=8:raise ValueError('Invalid batch.')
    results=[]
    for call in calls:
        if set(call)!={'name','arguments'} or call['name']!='frontalis_evaluate':raise ValueError('Read-only evaluation only.')
        results.append(decision(state,call['arguments']))
    if len({r['query_id'] for r in results})!=len(results):raise ValueError('Duplicate query.')
    print(json.dumps({'tool':'frontalis_evaluate','dispatchMode':'host-json','results':results},ensure_ascii=False,allow_nan=False))
if __name__=='__main__':main()
