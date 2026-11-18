import io,json,statistics,sys
from contextlib import redirect_stdout
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from summarize_matched import compare
r=Path(sys.argv[1])
def eligible(arm):
    try:
        actual=json.loads((r/f'actual-{arm}.json').read_text())
        if actual['status']!='passed':return None
        times=[]
        for i in (1,2,3):
            n=r/f"{arm}-p{i}-{arm}.json";c=r/f"{arm}-p{i}-control.json"
            a=json.loads(n.read_text());b=json.loads(c.read_text())
            if a['median_ms']>=b['median_ms']:return None
            if any(v['failed_count'] for v in compare(n.with_suffix('.validation.pt'),c.with_suffix('.validation.pt')).values()):return None
            times.append(a['median_ms']/b['median_ms'])
        return statistics.median(times)
    except (OSError,KeyError):return None
if eligible('both') is not None:
    opts=[(x,k) for k in ('direct','coarse') if (x:=eligible(k)) is not None]
    if opts:print(min(opts)[1])
