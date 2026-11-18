"""Exercise the ordinary CLI's real contract without starting training or allocating a GPU."""
import hashlib
import json
import os
from pathlib import Path
import sys

def main():
    root=Path(os.environ['RUN_ROOT']).resolve()
    source=root/'source'
    sys.path[:0]=[str(source/'scripts'),str(source/'src')]
    import run_va_eight_ablations as entry
    import va_imagenet100_runtime as runtime
    from torchvision.datasets import ImageFolder
    assert Path(entry.__file__).resolve()==source/'scripts/run_va_eight_ablations.py'
    sys.argv=[str(source/'scripts/run_va_eight_ablations.py'),'--root',str(root/'preflight-run'),
              '--data-root',str(root/'data'),'--workers','0','--epochs','19']
    args=runtime.parse_args(variants=entry.VARIANTS,seeds=entry.SEEDS)
    contract=entry._contract(args)
    train,val=(ImageFolder(args.data_root/s) for s in ('train','val'))
    assert len(train)==contract['data']['train_images']==1024 and len(val)==contract['data']['validation_images']==100
    assert len(train.classes)==100 and train.class_to_idx==val.class_to_idx
    for ds in (train,val):
        for path,_ in ds.samples:
            with ds.loader(path) as im:
                im.load()
    recipe=contract['recipe']
    assert recipe['effective_batch_size']==1024 and recipe['precision']=='bfloat16' and recipe['epochs']==19
    paths=contract['source_closure']
    manifest=json.loads((Path(__file__).parent/'source.json').read_text())
    for name,h in manifest.items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==h,name
    expected={n for n in manifest if n=='pyproject.toml' or (n.endswith('.py') and str(Path(n).parent) in ('scripts','src/lnet'))}
    assert expected.issubset(paths) and 'pyproject.toml' in paths
    closure=hashlib.sha256()
    for n in paths:closure.update(n.encode());closure.update((source/n).read_bytes())
    assert closure.hexdigest()==contract['source_sha256']['source_closure']
    receipt=dict(status='passed',contract_entry='run_va_eight_ablations._contract',classes=len(train.classes),
                 decoded_images=len(train)+len(val),dataset_contract=contract['data'],closure_files=len(paths),
                 manifest_files=len(manifest),pyproject_sha256=manifest['pyproject.toml'],source_closure_sha256=closure.hexdigest())
    (root/'contract-preflight.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('ORDINARY_CONTRACT_PREFLIGHT',json.dumps(receipt,separators=(',',':')))
if __name__=='__main__':main()
