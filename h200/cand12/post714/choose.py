"""Rank eligible candidates only after all evaluation pairs have completed."""
import json
from pathlib import Path
import statistics
import sys

def choose(root):
    eligible=[]
    for arm in ('outdot','tree','native','rows2','rows8','buckets'):
        if not (root/f"eligible-{arm}").is_file():continue
        r=root/"results";ratios=[]
        for i in (1,2,3):
            c=json.loads((r/f"{arm}-p{i}-control.json").read_text())
            a=json.loads((r/f"{arm}-p{i}-{arm}.json").read_text())
            ratios.append(a["median_ms"]/c["median_ms"])
        if max(ratios)<1:eligible.append((statistics.median(ratios),arm))
    return min(eligible)[1] if eligible else ""
if __name__=="__main__":print(choose(Path(sys.argv[1])))
