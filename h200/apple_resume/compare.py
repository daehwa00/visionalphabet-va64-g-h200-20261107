import json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'apple_checkpoint'))
import checkpointing as ck
import train

def main():
    a,b,out=map(Path,sys.argv[1:])
    x=ck.verified_load(a/'checkpoints/latest.pt');y=ck.verified_load(b/'checkpoints/latest.pt')
    for key in ('global_step','tokens_seen','skipped_optimizer_steps'):
        assert x['progress'][key]==y['progress'][key],key
    delta=norm=0.;maximum=0.
    for key,tensor in x['model'].items():
        d=tensor.double()-y['model'][key].double()
        delta+=d.square().sum().item();norm+=tensor.double().square().sum().item();maximum=max(maximum,d.abs().max().item())
    relative=(delta/norm)**.5
    assert math.isfinite(relative) and relative<1e-4,relative
    for folder in (a,b):
        assert all(json.loads((folder/'restore-audit.json').read_text())['matches'].values())
    train.atomic_json(out,{'model_relative_l2':relative,'model_max_abs':maximum,'threshold':1e-4,'tokens_steps_skips_match':True,'restores_match':True,'note':'Batch reduction differs; this is tolerance parity, not bitwise continuation.'})
    print('MIGRATION_COMPARISON='+out.read_text(),flush=True)
if __name__=='__main__':main()
