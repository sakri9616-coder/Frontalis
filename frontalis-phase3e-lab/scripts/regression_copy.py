"""Copy previous tests/code/evidence into the new lab; never test-write old lab."""
import os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT.parent/'frontalis-phase3d-lab';DEST=ROOT/'reference/phase3d-regression'
if DEST.exists():raise FileExistsError(DEST)
DEST.mkdir(parents=True)
for folder in ['npnd','tests','experiments','scripts','tools','reference','docs','reports']:
    for directory,dirs,files in os.walk(SOURCE/folder):
        dirs[:]=[d for d in dirs if d not in ['__pycache__','pip-cache']]
        for file in files:
            p=Path(directory)/file
            if p.suffix not in ['.py','.json','.md','.mjs','.cjs','.html','.txt']:continue
            target=DEST/p.relative_to(SOURCE);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
(DEST/'runtime').mkdir()
print(DEST)
