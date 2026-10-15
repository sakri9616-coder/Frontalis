"""Create, verify and restore separate archives without changing research bytes."""
import argparse,hashlib,json,os,shutil,stat,zipfile
from pathlib import Path,PurePosixPath
ARCH=Path(__file__).resolve().parent;ROOT=ARCH.parent;STAGE=ARCH/'.staging'
PLANS={
 'nephia-core-lab':'labs/core', 'nephia-npn-lab':'labs/npn',
 'nephia-phase3-lab':'labs/phase-3','nephia-phase35-lab':'labs/phase-3.5',
 'nephia-phase3b-lab':'labs/phase-3b','nephia-phase3b-self-lab':'labs/phase-3b-self',
 'frontalis-phase3c-lab':'labs/phase-3c',
 'frontalis-phase3d-repro-final-check':'reproductions/phase-3d',
 'frontalis-phase3d-repro-smoke-1':'reproductions/phase-3d',
 'frontalis-phase3d-repro-smoke-2':'reproductions/phase-3d',
 'frontalis-phase3e-repro-smoke':'reproductions/phase-3e',
 'frontalis-phase3f-repro-final':'reproductions/phase-3f',
 'frontalis-phase3f-repro-smoke':'reproductions/phase-3f'}

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def sha_file(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def sha_stream(f):return hashlib.file_digest(f,'sha256').hexdigest()
def paths(name):
    if name not in PLANS:raise ValueError('Unknown archive name.')
    folder=ARCH/PLANS[name];return folder/(name+'.zip'),folder/(name+'.manifest.json')

def snapshot(source):
    if not source.is_dir() or source.is_symlink() or source.stat().st_file_attributes&0x400:raise ValueError('Plain source directory required.')
    files={};dirs=['']
    for base,children,names in os.walk(source,followlinks=False):
        for name in children:
            p=Path(base)/name
            if p.is_symlink() or p.stat().st_file_attributes&0x400:raise ValueError('Source contains a reparse directory: '+str(p))
            dirs.append(p.relative_to(source).as_posix())
        for name in names:
            p=Path(base)/name;s=p.stat(follow_symlinks=False)
            if s.st_file_attributes&0x400:raise ValueError('Source contains a reparse file: '+str(p))
            files[p.relative_to(source).as_posix()]=dict(bytes=s.st_size,sha256=sha_file(p))
    return dict(files=files,directories=sorted(dirs))

def verify(name,source=None):
    archive,manifest=paths(name);m=read(manifest)
    if m['originalRoot']!=name or sha_file(archive)!=m['archiveSha256']:raise ValueError('Archive identity/hash mismatch.')
    expected={name+'/'+d+'/' if d else name+'/' for d in m['directories']}|{name+'/'+p for p in m['files']}
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist())!=expected or len(z.namelist())!=len(expected):raise ValueError('ZIP entry set mismatch.')
        for member in z.infolist():
            p=PurePosixPath(member.filename)
            if p.is_absolute() or '..' in p.parts or p.parts[0]!=name or stat.S_ISLNK(member.external_attr>>16):raise ValueError('Unsafe archive entry.')
        for path,expected_file in m['files'].items():
            info=z.getinfo(name+'/'+path)
            with z.open(info) as f:h=sha_stream(f)
            if info.file_size!=expected_file['bytes'] or h!=expected_file['sha256']:raise ValueError('Archived bytes mismatch: '+path)
    if source is not None:
        actual=snapshot(source)
        if actual['files']!=m['files'] or actual['directories']!=m['directories']:raise ValueError('Source changed since archiving: '+name)
    return m

def create():
    for name in PLANS:
        archive,manifest=paths(name);archive.parent.mkdir(parents=True,exist_ok=True)
        if archive.exists() or manifest.exists():verify(name,ROOT/name);continue
        source=ROOT/name;s=snapshot(source);temporary=archive.with_suffix('.partial.zip')
        with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9,allowZip64=True,strict_timestamps=False) as z:
            for d in s['directories']:z.write(source/d,arcname=name+'/'+d+'/' if d else name+'/')
            for p in sorted(s['files']):z.write(source/p,arcname=name+'/'+p)
        temporary.rename(archive)
        m=dict(originalRoot=name,archiveSha256=sha_file(archive),archiveBytes=archive.stat().st_size,**s)
        write(manifest,m);verify(name,source)
        print('Verified',name,len(s['files']),'files;',archive.stat().st_size,'archive bytes',flush=True)
    index()

def index():
    records=[]
    for name in PLANS:
        archive,manifest=paths(name)
        if not archive.exists():continue
        m=read(manifest);records.append(dict(name=name,archive=archive.relative_to(ARCH).as_posix(),manifest=manifest.relative_to(ARCH).as_posix(),
            files=len(m['files']),sourceBytes=sum(x['bytes'] for x in m['files'].values()),archiveBytes=m['archiveBytes'],
            archiveSha256=m['archiveSha256'],sourcePresent=(ROOT/name).exists(),stagingPresent=(STAGE/name).exists()))
    write(ARCH/'index.json',dict(schema='frontalis-archive-index-v1',retainedActive=['frontalis-phase3d-lab','frontalis-phase3e-lab','frontalis-phase3f-lab'],archives=records))

def restore(name):
    names=[n for n in PLANS if PLANS[n].startswith('labs/')] if name=='all-labs' else [name]
    for n in names:
        dest=ROOT/n
        if dest.exists():raise FileExistsError('Restore never overwrites an existing folder: '+n)
        verify(n);archive,_=paths(n);temporary=ARCH/'.restore'/n
        if temporary.exists():raise FileExistsError('Restore staging already exists.')
        temporary.mkdir(parents=True)
        try:
            with zipfile.ZipFile(archive) as z:z.extractall(temporary)
            verify(n,temporary/n)
            (temporary/n).rename(dest);temporary.rmdir()
        except Exception:
            print('Incomplete restoration retained at',temporary,flush=True);raise
        print('Restored',n,flush=True)
    index()

def audit():
    opened={};bad=[];checked=0;archived=0
    def check(p,size,expected):
        nonlocal checked,archived
        if p.is_file():actual=p.stat().st_size;h=sha_file(p)
        else:
            relative=p.relative_to(ROOT);name=relative.parts[0]
            if name not in PLANS:raise FileNotFoundError(str(p))
            if name not in opened:verify(name);opened[name]=zipfile.ZipFile(paths(name)[0])
            z=opened[name];member=relative.as_posix();actual=z.getinfo(member).file_size
            with z.open(member) as f:h=sha_stream(f)
            archived+=1
        checked+=1
        if actual!=size or h!=expected:bad.append(str(p))
    latest=read(ROOT/'frontalis-phase3f-lab/artifact-sha256.json')
    previous=read(ROOT/'frontalis-phase3f-lab/reference/phase3e-artifact-sha256.json')
    older=read(ROOT/'frontalis-phase3e-lab/reference/prior-sha256.json')
    try:
        for r in latest['files']:check(ROOT/'frontalis-phase3f-lab'/r['path'],r['bytes'],r['sha256'])
        for r in previous['files']:check(ROOT/'frontalis-phase3e-lab'/r['path'],r['bytes'],r['sha256'])
        for r in older:check(Path(r['path']),r['bytes'],r['sha256'])
    finally:
        for z in opened.values():z.close()
    result=dict(checked=checked,readFromArchives=archived,changed=len(bad),changedPaths=bad,
        latestSealedFiles=len(latest['files']),priorProtectedFiles=len(previous['files'])+len(older))
    write(ARCH/'preservation-audit.json',result);print(json.dumps(result,indent=2),flush=True)
    if bad:raise ValueError('Frozen research integrity mismatch.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['create','verify-stage','verify-source','index','verify-all','restore','audit']);p.add_argument('name',nargs='?');a=p.parse_args()
    if a.action=='create':create()
    elif a.action=='verify-stage':verify(a.name,STAGE/a.name);print('Staged source matches verified archive:',a.name,flush=True)
    elif a.action=='verify-source':verify(a.name,ROOT/a.name);print('Source matches verified archive:',a.name,flush=True)
    elif a.action=='index':index()
    elif a.action=='verify-all':
        for name in PLANS:verify(name);print('Verified archive:',name,flush=True)
    elif a.action=='restore':restore(a.name)
    else:audit()
