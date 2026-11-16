"""Matched scratch A experiment with real + imaginary complex correlation readout."""
from __future__ import annotations
import os,time
from pathlib import Path
import torch
import run_va_night_a as anchor
from a2d_r2k3_capacity_factory import CapacityQ4OnlyAffineClassifier
from lnet.terminal_imaginary_correlation import complex_correlation_descriptor

baseline=anchor.baseline
VARIANT='VA-RealMemory-A-TerminalRawGOpt-IN100'
VARIANTS=(VARIANT,)
SEEDS=(501,)
PARAMETERS=2_988_004

class AImaginaryCorrelation(anchor.PyramidRealMemoryIN100):
    architecture='va-real-memory-a-terminal-rawg-opt-v1'

    def forward(self, inputs):
        state=self.forward_features(inputs)
        terminal=self.backbone.terminal
        drive=terminal.pole_input_projection(state)
        px,py=terminal.pole_coefficients((*drive.shape[:-1],terminal.modes))
        descriptor=complex_correlation_descriptor(px,py,drive,gain_normalization=terminal.product_gain_normalization)
        with torch.autocast(device_type=descriptor.device.type,enabled=False):
            return self.classifier(descriptor.float(),terminal_only=True)


def _build(variant,config):
    if variant!=VARIANT:raise ValueError(variant)
    model=anchor._build(anchor.VARIANT,config)
    old=model.classifier
    with torch.random.fork_rng(devices=[]):
        new=CapacityQ4OnlyAffineClassifier(1024,1024,100)
    with torch.no_grad():
        new.affine.linear.weight.zero_()
        new.affine.linear.weight[:,:512].copy_(old.affine.linear.weight)
        new.affine.linear.bias.copy_(old.affine.linear.bias)
        new.affine.standardizer.running_mean[:512].copy_(old.affine.standardizer.running_mean)
        new.affine.standardizer.running_var[:512].copy_(old.affine.standardizer.running_var)
        new.affine.standardizer.num_batches_tracked.copy_(old.affine.standardizer.num_batches_tracked)
    model.backbone.classifier=new
    model.backbone.descriptor_dim=1024
    model.__class__=AImaginaryCorrelation
    if sum(p.numel() for p in model.parameters())!=PARAMETERS:raise RuntimeError('parameter inventory changed')
    from lnet.real_memory_training import enable_real_memory_training_
    model._training_optimization_inventory=enable_real_memory_training_(model)
    return model


def _contract(args):
    payload=anchor._contract(args)
    root=Path(__file__).resolve().parents[1]
    paths=baseline.runtime.source_dependency_paths(root,(Path(__file__).stem,))
    config=dict(payload['variant_configs'][anchor.VARIANT])
    config.update(architecture=AImaginaryCorrelation.architecture,head='BatchNorm1024-affine-false Linear1024-to-100',readout='[mean Re(2s*conj(u)-abs(u)^2)4P; mean Im(2s*conj(u)-abs(u)^2)4P], u=gamma_y*horizontal_state',correlation_axes='vertical conditional on horizontal scan; four existing direction orders',correlation_normalization='none before classifier BatchNorm; no log on signed channels')
    payload.update(schema='lnet.real_memory_a_terminal_rawg_opt.in100.v1',variants=list(VARIANTS),priority=list(VARIANTS),variant_configs={VARIANT:config},parameter_counts={VARIANT:PARAMETERS},architecture={VARIANT:AImaginaryCorrelation.architecture},initialization='scratch seed501 A backbone/head real half; new correlation head weights zero; no trained checkpoint loaded',references={'baseline_variant':anchor.VARIANT,'baseline_parameters':anchor.PARAMETERS,'baseline_final_accuracy':0.8658000230789185,'parameter_delta':51200,'parameter_matched':False},source_closure=[p.relative_to(root).as_posix() for p in paths])
    payload['source_sha256'].update(imcorr_runner=baseline.runtime.digest(__file__),source_closure=baseline.runtime.source_fingerprint(root,paths))
    return payload


def _build_optimizer(model,recipe):
    optimizer=baseline.runtime.build_optimizer(model,recipe)
    destination=os.environ.get('VA_IN100_PROGRESS_PATH')
    if destination:
        steps=0
        def record(opt,args,kwargs):
            nonlocal steps
            steps+=1
            if steps!=1 and steps%100:return
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            finite=bool(grads) and bool(torch.stack([g.isfinite().all() for g in grads]).all())
            baseline.harness._atomic_json(Path(destination),dict(variant=VARIANT,seed=501,session_optimizer_steps=steps,finite_gradients=finite,current_learning_rates=[g['lr'] for g in opt.param_groups],unix_time=time.time()))
            if not finite:raise RuntimeError('nonfinite gradients')
        optimizer.register_step_post_hook(record)
    return optimizer


def main():
    baseline.runtime.configure(VARIANTS,SEEDS)
    baseline.heads.VARIANTS=VARIANTS;baseline.heads.SEEDS=SEEDS
    baseline.structured._training_objective=baseline.heads._training_objective
    baseline.structured._after_training_batch=baseline.heads._after_training_batch
    torch.set_float32_matmul_precision('high');torch.backends.cudnn.benchmark=True
    baseline.harness.main(baseline.harness.runner_bindings(variants=VARIANTS,seeds=SEEDS,model_config=baseline.ComplexScanConfig,build_model=_build,contract=_contract,build_optimizer=_build_optimizer,prepare_model=baseline.prepare._prepare_model,train_epoch=baseline.structured._train_epoch,evaluate=baseline.heads._evaluate,wandb_model_metrics=baseline._wandb_model_metrics,summarize=baseline.heads._summarize))

if __name__=='__main__':main()
