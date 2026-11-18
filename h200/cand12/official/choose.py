import json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]);r=root/'results'
def ratio(arm):
    if not (root/('checked-'+arm)).exists():return None
    vals=[json.loads((r/f"{arm}-p{i}-{arm}.json").read_text())["median_ms"]/json.loads((r/f"{arm}-p{i}-control.json").read_text())["median_ms"] for i in (1,2,3)]
    return statistics.median(vals) if max(vals)<1 else None
if ratio('both') is not None:
    eligible=[(a,k) for k in ('direct','coarse') if (a:=ratio(k)) is not None]
    if eligible:print(min(eligible)[1])
