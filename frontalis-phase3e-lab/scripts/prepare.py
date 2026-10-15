"""Create authored corpus and preservation manifest before any experiment."""
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from npne.meaning import meaning,event

def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def corpus():
    bases=[
      (['友人が私の許可を得て制作を手伝った。そのおかげで作品は完成し、失った物はなかった。','私は友人の制作の助けを受け入れた。手助けによって制作は成功し、損失は生じなかった。','友達は私に同意を確かめてから制作を助けた。それが完成につながり、何も失わなかった。','制作を助けたいと友人に言われ、私は承諾した。結果、その助けで作品が出来上がり、損失はなかった。'],
       meaning([event('help')],outcome=[.5,.5,.5,0,1,.5,.5,.5],consent='yes',causal='causes',temporal='after')),
      (['組織が私の同意なしに私の選択を制限した。そのため私は材料を失い、制作は失敗した。','私は承諾していないのに、組織が私の自由な選択を封じた。その制約で材料が失われ、完成しなかった。','私の許可を取らずに組織が私の選択肢を狭めた。その結果、材料に損失が出て制作は失敗した。','組織による私の選択の制限に、私は同意していなかった。その制限が原因で材料をなくし、作品は未完成となった。'],
       meaning([event('restrict','institution')],cue=[0,1,.5,.5,.5,.5,.5,.5],outcome=[.5,.5,.5,1,0,.5,.5,.5],consent='no',causal='causes',temporal='after')),
      (['友人は私をだましたわけではない。私はその友人を以前から信頼している。結果はまだ分からない。','以前から信頼している友人について、私への欺きはなかったと確認された。結果は未確認だ。','その友達への事前の信頼は高い。友達が私を欺いた、という話は否定された。その後の結果は不明だ。','私が長く信頼してきた友人は、私を欺いていなかった。何が起きた結果なのかは分からない。'],
       meaning([event('deceive',negated=True)],trust='high')),
      (['知らない人が私を手伝ったかもしれないと聞いた。事実かどうかも結果もまだ不明だ。','面識のない人が私を助けたという未確認の噂がある。実際にあったかも結果も確認できない。','私を助けてくれたのは知らない人だったらしい。ただし伝聞にすぎず、結果も不確かだ。','見知らぬ誰かが私を手助けした可能性はあるが、確認は取れていない。結果は分からない。'],
       meaning([event('help','stranger',uncertain=True)])),
      (['私は自分で選んで未知の場所を探索した。その探索が原因で道具を失ったが、新しい仕組みも発見した。','誰にも強制されず、私は未知の場所を調べに行った。その結果、道具に損失が出る一方、新発見があった。','私は自由に決めて未踏の場所を探索した。それによって道具をなくし、同時に未知の仕組みを見つけた。','未知の場所に出向くことは私の自由な選択だった。探索の結果として道具は失われ、新しい仕組みが分かった。'],
       meaning([event('explore','self','object')],cue=[1,0,.5,.5,.5,.5,.5,.5],outcome=[.5,.5,.5,1,.5,1,.5,.5],causal='causes',temporal='after')),
      (['友人が私の行動を制限した。その制限が事故を防ぎ、私は物を失わずに済んだ。','私の行動を友達が止めた。それにより事故が防がれ、損失はなかった。','友人による私の行動の制約は、事故の発生を防いだ。そのため私は何も失わなかった。','友達が私の行動を制限したおかげで、事故を回避できた。物の損失は出なかった。'],
       meaning([event('restrict')],cue=[.5,1,.5,.5,.5,.5,.5,.5],outcome=[.5,.5,.5,0,.5,.5,.5,.5],causal='prevents',temporal='after')),
      (['友人は私のためになるよう意図して私を手伝い、その後、私を傷つける目的で私をだました。数日後に材料の損失が判明した。二つの行為と損失の因果は不明だ。','友達は私を助けようとして手伝ったが、続いて私を害そうとだました。材料を失ったと分かったのは数日後で、因果関係は分からない。','私への善意を明言した友人の手助けの後、同じ友人による私を傷つける意図の欺きがあった。後日、材料の損失が分かったが原因は未確認だ。','友人は私に利益を与える意図で手伝い、その次に私を害するために欺いた。日が経って材料の損失が確認された。どの行為が原因かは分からない。'],
       meaning([event('help',intent='benefit'),event('deceive',intent='harm')],outcome=[.5,.5,.5,1,.5,.5,.5,.5],temporal='after',delayed=True)),
      (['私は物を作り、同時に知らない人が私を手伝った。二つの行為は無関係だと確認されている。完成したかは不明だ。','私が制作しているのと同じ時に、面識のない人が私を助けた。両者に因果関係はなく、制作の結果はまだ分からない。','私の制作と知らない人の私への手助けは同時だったが、二つは無関係と確認された。成功したかは未確認だ。','私は制作を行い、それと並行して見知らぬ人が私を助けた。二つの行為は互いに関係しない。完成の確認はない。'],
       meaning([event('create','self','object'),event('help','stranger')],causal='unrelated',temporal='simultaneous')),
    ]
    return [dict(id=f's{i*4+j:02d}',family=i,variant=j,split='heldout' if j==3 else 'formation',text=t,gold=m,outcomeConfirmed=i not in [2,3,7]) for i,(texts,m) in enumerate(bases) for j,t in enumerate(texts)]

def preserve():
    paths=set()
    for d in ROOT.parent.iterdir():
        if d.is_dir() and (d.name.startswith(('frontalis-phase3','nephia-'))) and d!=ROOT:
            for directory,dirs,files in os.walk(d):
                dirs[:]=[n for n in dirs if n not in {'.venv','__pycache__','runtime','sandbox','node_modules','.git'}]
                for name in files:paths.add(Path(directory)/name)
    prior=ROOT.parent/'frontalis-phase3d-lab/reference/prior-sha256.json'
    for row in json.loads(prior.read_text(encoding='utf8')):
        p=Path(row['path'])
        if p.is_file():paths.add(p)
    return [dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(paths)]

if __name__=='__main__':
    corpus_path=ROOT/'data/natural.json'
    if corpus_path.exists():
        if json.loads(corpus_path.read_text(encoding='utf8'))!=corpus():raise ValueError('Existing corpus differs.')
    else:write(corpus_path,corpus())
    rows=preserve();write(ROOT/'reference/prior-sha256.json',rows)
    print(json.dumps({'naturalCases':32,'independentStoryFamilies':8,'preservedFiles':len(rows)}))
