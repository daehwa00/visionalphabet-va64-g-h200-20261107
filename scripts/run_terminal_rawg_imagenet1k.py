"""Optimized A backbone, terminal raw G only; matched IN1K recipe, global batch1024."""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, shutil, signal, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import imagenet1k_runtime as runtime
import run_va_a_imcorr as rawg
from a2d_r2k3_capacity_factory import CapacityQ4OnlyAffineClassifier

MODEL_KEY='va_real_memory_a_terminal_rawg_opt'
PARAMETERS=3_910_504

class ImageNet1KRawG(rawg.AImaginaryCorrelation):
    def forward(self, inputs):
        return super().forward(inputs)[0]


def build_model(*, prepared=True):
    model=rawg._build(rawg.VARIANT,rawg.baseline.ComplexScanConfig(output_dim=100,stem_strides=(2,2)))
    # Fresh IN1K classifier; no trained IN100 weights are loaded.
    energy=CapacityQ4OnlyAffineClassifier(512,512,1000)
    head=CapacityQ4OnlyAffineClassifier(1024,1024,1000)
    with torch.no_grad():
        head.affine.linear.weight.zero_()
        head.affine.linear.weight[:,:512].copy_(energy.affine.linear.weight)
        head.affine.linear.bias.copy_(energy.affine.linear.bias)
    model.backbone.classifier=head
    model.__class__=ImageNet1KRawG
    if sum(p.numel() for p in model.parameters())!=PARAMETERS:raise RuntimeError('IN1K inventory mismatch')
    if prepared:model.prepare_for_compiled_training_()
    return model


def make_task(args):
    manifest=json.loads(args.data_manifest.read_text())
    if manifest['root']!=str(args.data_root.resolve()) or manifest['counts']!={'train':1281167,'val':50000}:
        raise ValueError('Full IN1K dataset manifest required')
    source=Path(__file__).resolve().parents[1]
    output=args.output_root.resolve()/MODEL_KEY/'seed_0'
    return runtime.BaselineTask(phase='full',model_key=MODEL_KEY,seed=0,learning_rate=0.003,epochs=100,
        data_root=args.data_root.resolve(),output_dir=output,result_path=output/'result.json',checkpoint_path=output/'checkpoint.pt',
        source_root=source,batch_size=args.batch_size,effective_batch_size=1024,workers=args.workers,resume=args.resume,
        num_classes=1000,precision='bfloat16',wandb_mode='disabled',
        execution=runtime.ExecutionOptions(compile_model=True,gpu_mixup=True,non_blocking_transfer=True,prefetch_factor=2,metric_sync_interval=25),
        source_sha256={'source_manifest':hashlib.sha256((source/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),
                      'data_manifest':hashlib.sha256(args.data_manifest.read_bytes()).hexdigest()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--data-manifest',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--batch-size',type=int,choices=(128,256),default=256)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--stop-file',type=Path,required=True);p.add_argument('--resume',action='store_true')
    args=p.parse_args();task=make_task(args);task.output_dir.mkdir(parents=True,exist_ok=True)
    managed=Path(os.environ['VA64_G_MANAGED_ROOT']) if os.environ.get('VA64_G_MANAGED_ROOT') else None
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:args.stop_file.touch())
    if managed:
        from va64_runtime_operations import current
        current(managed,'va64-g-seed0','training',task.output_dir)
    runtime._atomic_json(task.output_dir/'architecture.json',{'model':MODEL_KEY,'parameters':PARAMETERS,'seed':0,
        'epochs':100,'physical_batch':args.batch_size,'effective_batch':1024,'accumulation':task.gradient_accumulation_steps,
        'learning_rate':0.003,'channels':[64,96,128,256],'poles':[160,160,160,128],'depth':[2,2,6,2],
        'readout':'terminal only raw mean(2s*conj(u)-abs(u)^2), Re512+Im512; no readout normalization or logarithm',
        'classifier':'BatchNorm1024 affine=false plus Linear1024-to-1000; imaginary half zero initialized',
        'initialization':'scratch seed0; no trained checkpoint','protocol':'existing IN1K runtime; batch changed256-to1024; LR unchanged'})
    def builder(key,_source,classes):
        if key!=MODEL_KEY or classes!=1000:raise ValueError('model task mismatch')
        model=build_model()
        random.seed(1_000_000);np.random.seed(1_000_000);torch.manual_seed(1_000_000);torch.cuda.manual_seed_all(1_000_000)
        return model
    def step(row):
        if not math.isfinite(row['loss']):raise RuntimeError('nonfinite training loss')
        with (task.output_dir/'steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print('RAWG_STEP='+json.dumps(row),flush=True)
    def optimizer_builder(model,lr,device):
        opt=runtime._build_optimizer(model,lr,device)
        def check(opt,_args,_kwargs):
            grads=[p.grad for g in opt.param_groups for p in g['params'] if p.grad is not None]
            if not grads or not bool(torch.stack([torch.isfinite(g).all() for g in grads]).all()):raise RuntimeError('nonfinite gradients')
        opt.register_step_pre_hook(check);return opt
    def epoch(model,row):
        del model
        runtime._atomic_json(task.output_dir/'checkpoint-meta.json',{**row,'status':'local_checkpoint'})
        runtime._atomic_json(task.output_dir/'latest-epoch.json',row)
        with (task.output_dir/'epochs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        best_path=task.output_dir/'best.json';best=json.loads(best_path.read_text())['accuracy'] if best_path.exists() else -1
        if row['validation']['accuracy']>best:
            shutil.copyfile(task.checkpoint_path,task.output_dir/'.best.tmp');(task.output_dir/'.best.tmp').replace(task.output_dir/'best.pt')
            runtime._atomic_json(best_path,{'epoch':row['epoch'],'accuracy':row['validation']['accuracy']})
        print('RAWG_EPOCH='+json.dumps(row),flush=True)
    torch.set_float32_matmul_precision('high');torch.backends.cudnn.benchmark=True
    result=runtime.run_task(task,model_builder=builder,optimizer_builder=optimizer_builder,epoch_callback=epoch,step_callback=step,
        stop_requested=args.stop_file.is_file,stop_without_checkpoint=True)
    if managed:
        from va64_runtime_operations import backup_and_wait,complete_run
        backup_and_wait(managed,'va64-g-seed0',task.output_dir,100,result['global_step'],{'status':'completed'})
        complete_run(managed,task.output_dir,'completed')
    print('RAWG_FINAL='+json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)

if __name__=='__main__':main()
