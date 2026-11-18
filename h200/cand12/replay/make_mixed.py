"""One-factor follow-up to the 60.6 ms combination, chosen by validated operand replay."""
import ast
import json
from pathlib import Path
import statistics
import sys

root=Path(sys.argv[1]);r=json.loads((root/'micro/result.json').read_text())
base=r['samples']['special_double']
eligible=[]
for mode in ('special_reconstruct','special_adjoint'):
    tests=r['checks'].get(mode,{})
    sample=r['samples'].get(mode,[])
    if len(tests)==9 and all(v['passed'] for v in tests.values()) and len(sample)==len(base):
        ratios=[a/b for a,b in zip(sample,base)]
        if max(ratios)<1 and statistics.median(ratios)<.99:eligible.append((statistics.median(ratios),mode))
if not eligible: sys.exit()
best=min(eligible)[1];p=root/'source-mixed/src/lnet/direct_path_scan_kernels.py'
s=p.read_text();tree=ast.parse(s);fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_path_backward_kernel')
sites=sorted([n for n in ast.walk(fn) if isinstance(n,ast.Call) and ast.unparse(n.func)=='_prefix_pair'],key=lambda n:n.lineno)
assert len(sites)==4
ls=s.splitlines(True)
for i,n in enumerate(sites):
    flag=n.args[-1];assert ast.unparse(flag)=='DOUBLING'
    if (best=='special_reconstruct' and i>=2) or (best=='special_adjoint' and i<2):
        ls[flag.lineno-1]=ls[flag.lineno-1].replace('DOUBLING,','DOUBLING and COARSE,')
p.write_text(''.join(ls));(root/'mixed-mode').write_text(best)
print("MIXED_SELECTED",best)
