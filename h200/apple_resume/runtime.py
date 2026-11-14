"""Audited single-GPU continuation with only microbatch/GAS migration allowed."""
import copy, hashlib, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'apple_checkpoint'))
import torch
import train
import checkpointing as ck

def digest(value):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            x=x.detach().cpu().contiguous()
            h.update(str((x.dtype,list(x.shape))).encode());h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=repr):visit(k);visit(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(type(x).__name__.encode())
            for y in x:visit(y)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()

def convert(saved,expected,receipt):
    old=saved['contract']; expected=ck.plain(expected)
    assert saved['kind']=='training' and saved['schema']==2
    assert old['branch_source']=='residual' and expected['branch_source']=='residual'
    assert old['global_batch']==expected['global_batch']==1024
    for config in (old,expected):
        assert config['microbatch']*config['gradient_accumulation_steps']==1024
    allowed={'microbatch','gradient_accumulation_steps'}
    mismatch={k for k in set(old)|set(expected) if old.get(k)!=expected.get(k)}
    if mismatch-allowed:raise ValueError('Unexpected migration contract fields: '+repr(mismatch-allowed))
    progress=saved['progress']
    assert progress['tokens_seen']==progress['global_step']*1024*2048
    assert len(saved['rng']['cuda'])==1
    best=saved.get('best')
    if progress.get('best_val') is not None:
        assert best and best['kind']=='weights' and best['contract']==old
        assert best['progress']['val_loss']==progress['best_val']
    provenance={'source_receipt':receipt,'source_contract':old,'changed_fields':sorted(mismatch),
        'note':'Single RTX4090 to H200; global batch/cursor and optimizer states preserved; microbatch reduction order differs.'}
    output=dict(saved,contract=expected,migration_provenance=provenance)
    if best:output['best']=dict(best,contract=expected,migration_provenance=provenance)
    return output

def main():
    mode=sys.argv[1]
    args=train.parser().parse_args(sys.argv[2:])
    if mode=='migrate':
        path=Path(args.resume); saved=ck.verified_load(path)
        receipt=json.loads(path.with_suffix('.pt.json').read_text())
        model=train.build_model(args.seed,branch_source=args.branch_source)
        expected=train.checkpoint_contract(train.contract(args,model),args)
        migrated=convert(saved,expected,receipt)
        out=Path(args.output_root);out.mkdir(parents=True,exist_ok=False)
        new_receipt=ck.atomic_save(out/'latest.pt',migrated)
        loaded=ck.verified_load(out/'latest.pt')
        matches={k:digest(saved[k])==digest(loaded[k]) for k in ('model','optimizer','scaler','rng','progress','model_config')}
        if saved.get('best'):matches['best_model']=digest(saved['best']['model'])==digest(loaded['best']['model'])
        assert all(matches.values())
        train.atomic_json(out/'migration-report.json',dict(matches=matches,receipt=new_receipt,provenance=migrated['migration_provenance']))
        print('MIGRATION_VALIDATED='+json.dumps(matches),flush=True)
        return
    assert mode=='train'
    import stream
    original_stream=stream.TokenStream
    class ResilientStream(original_stream):
        def __init__(self,*a,**kw):
            kw.setdefault('retries',8);super().__init__(*a,**kw)
    stream.TokenStream=ResilientStream
    original_restore=ck.restore_training
    def restore(path,model,optimizer,scaler,**kw):
        progress=original_restore(path,model,optimizer,scaler,**kw)
        saved=ck.verified_load(path)
        actual={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'rng':ck.rng_state()}
        matches={k:digest(saved[k])==digest(actual[k]) for k in actual}
        assert all(matches.values()),matches
        train.atomic_json(Path(args.output_root)/'restore-audit.json',dict(matches=matches,global_step=progress['global_step'],checkpoint_sha256=ck.digest_file(path)))
        return progress
    ck.restore_training=restore
    return train.run(args)

if __name__=='__main__':sys.exit(main())
