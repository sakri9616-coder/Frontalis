import hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from npnf.data import natural_cases,CONTRACT,PAST,NOW
from npnf.core import json_write

def main():
    json_write(ROOT/'data/natural.json',natural_cases())
    (ROOT/'data/extraction-contract.txt').write_text(CONTRACT,encoding='utf8')
    json_write(ROOT/'data/scenarios.json',dict(past=PAST,current=NOW,combinations=16))
    parent=ROOT.parent/'frontalis-phase3e-lab'
    # Phase 3-E's fixed manifest excludes its user's mutable interactive sessions.
    shutil.copyfile(parent/'artifact-sha256.json',ROOT/'reference/phase3e-artifact-sha256.json')
    paths=['resources/skills/memory-templates/ja-JP/journal.md','src/shared/memory-format.ts','src/shared/memory-page.ts','src/shared/ipc.ts','src/main/services/memory-store.ts','src/main/services/memory-curation.ts','src/main/services/brain/memory-tools.ts','docs/adr/0052-the-memory-is-written-in-the-conversation-language-from-templates-in-every-language.md']
    asist=ROOT.parent/'asist'
    json_write(ROOT/'reference/asist-source-sha256.json',{p:hashlib.sha256((asist/p).read_bytes()).hexdigest() for p in paths})
    print('Prepared authored synthetic data and read-only source references.')

if __name__=='__main__':main()
