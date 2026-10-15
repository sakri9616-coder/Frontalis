import argparse,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args()
    if not a.name.replace('-','').isalnum():raise ValueError('Use an alphanumeric name.')
    dest=ROOT.parent/('frontalis-phase3f-repro-'+a.name)
    if dest.exists():raise FileExistsError('Fresh reproduction directory required.')
    dest.mkdir()
    for folder in ['npnf','experiments','scripts','tests','tools','examples','playground']:
        shutil.copytree(ROOT/folder,dest/folder,ignore=shutil.ignore_patterns('__pycache__','states','demo-output'))
    for folder in ['data','reference']:
        shutil.copytree(ROOT/folder,dest/folder,ignore=shutil.ignore_patterns('dialogue-plan.json','pipeline-*.json','future-*.json','dual-*.json'))
    for file in ['PREREGISTRATION.md','LITERATURE.md','start-workbench.ps1']:
        shutil.copyfile(ROOT/file,dest/file)
    (dest/'reports').mkdir();(dest/'models').mkdir()
    print(dest)
