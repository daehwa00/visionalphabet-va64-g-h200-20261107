"""Run the pinned source generator; trial.py re-keys only an observed H200 direct scope."""
from pathlib import Path
p=Path(__file__).with_name('variants.py')
src=p.read_text()
marker='sys.path.insert(0,str((d/"src").resolve()))'
assert src.count(marker)==1
exec(compile(src.split(marker)[0],str(p),"exec"),{"__name__":"__main__"})
