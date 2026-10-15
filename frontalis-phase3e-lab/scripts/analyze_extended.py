"""Original scoring, with separately named user-authorized Haiku arm."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import analyze
analyze.ENGINES=['mimo-v2.6-flash-free','nemotron-3-ultra-free','claude-haiku-5-5']
{'natural':analyze.natural,'numerical':analyze.numerical,'dialogue':analyze.dialogue}[sys.argv[1]]()
