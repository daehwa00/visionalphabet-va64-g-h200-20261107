"""Independent seed501 IN100 ablations over the optimized terminal raw-G model."""
import os
import json
from pathlib import Path
import torch
import run_va_a_imcorr as raw
from va_ablation_model import build_backbone
from a2d_r2k3_capacity_factory import CapacityQ4OnlyAffineClassifier
from lnet.terminal_imaginary_correlation import complex_correlation_descriptor
from lnet.real_memory_training import enable_real_memory_training_

baseline=raw.baseline
RAW_VARIANT=raw.VARIANT
BASE=dict(channels=(64,96,128,256),poles=(160,160,160,128),depth=(2,2,6,2),normalize=False,cutmix=False,timm_ra=False)
SPECS={'10-dw-terminal-rmsnorm':dict(reader='DW3x3 -> pointwise C-to-2P',terminal_input_norm='same as stage4 same-resolution Reader',terminal_rms_matching=False)}
KEY=os.environ.get('VA_ABLATION','10-dw-terminal-rmsnorm')
SPEC=BASE|SPECS[KEY]
VARIANT='VA-Ablation-'+KEY
VARIANTS=(VARIANT,)
SEEDS=(501,)

class AblationModel(raw.AImaginaryCorrelation):
    architecture='va-terminal-rawg-dw-terminal-rmsnorm-v1'
    def forward(self,inputs):
        state=self.forward_features(inputs)
        terminal=self.backbone.terminal
        drive=terminal.pole_input_projection(state)
        px,py=terminal.pole_coefficients((*drive.shape[:-1],terminal.modes))
        descriptor=complex_correlation_descriptor(px,py,drive,normalize=self.normalize)
        with torch.autocast(device_type=descriptor.device.type,enabled=False):
            return self.classifier(descriptor.float(),terminal_only=True)

def _build(variant,config):
    if variant!=VARIANT:raise ValueError(variant)
    model=build_backbone(config,SPEC['channels'],SPEC['poles'],SPEC['depth'])
    old=model.classifier
    half=4*SPEC['poles'][-1]
    with torch.random.fork_rng(devices=[]):
        new=CapacityQ4OnlyAffineClassifier(2*half,2*half,100)
    with torch.no_grad():
        new.affine.linear.weight.zero_()
        new.affine.linear.weight[:,:half].copy_(old.affine.linear.weight)
        new.affine.linear.bias.copy_(old.affine.linear.bias)
        new.affine.standardizer.running_mean[:half].copy_(old.affine.standardizer.running_mean)
        new.affine.standardizer.running_var[:half].copy_(old.affine.standardizer.running_var)
        new.affine.standardizer.num_batches_tracked.copy_(old.affine.standardizer.num_batches_tracked)
    model.backbone.classifier=new
    model.backbone.descriptor_dim=2*half
    model.__class__=AblationModel
    model.normalize=SPEC['normalize']
    model._training_optimization_inventory=enable_real_memory_training_(model)
    from lnet.depthwise_memory_reader import replace_readers_
    model = replace_readers_(model)
    import copy
    terminal = model.backbone.terminal.pole_input_projection
    template = model.backbone.same_resolution_blocks['7'].reader.input_norm
    if template is None or template.weight.numel() != terminal.input_channels:
        raise RuntimeError('stage4 norm template mismatch')
    terminal.input_norm = copy.deepcopy(template)
    terminal.match_input_rms = False
    return model

def _contract(args):
    payload=raw._contract(args)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(501)
        model=_build(VARIANT,baseline.ComplexScanConfig(**payload['model']))
    parameters=sum(p.numel() for p in model.parameters())
    root=Path(__file__).resolve().parents[1]
    paths=baseline.runtime.source_dependency_paths(root,(Path(__file__).stem,'va_ablation_augment'))
    payload.update(schema='lnet.dw_terminal_rmsnorm.in100.v1',variants=list(VARIANTS),priority=list(VARIANTS),variant_configs={VARIANT:SPEC},parameter_counts={VARIANT:parameters},architecture={VARIANT:AblationModel.architecture},references={'baseline_variant':RAW_VARIANT,'baseline_parameters':raw.PARAMETERS,'baseline_accuracy':.8666,'parameter_delta':parameters-raw.PARAMETERS},initialization='scratch seed501; no trained weights; same-shaped reference tensors paired; imaginary head weights zero',source_closure=[p.relative_to(root).as_posix() for p in paths])
    payload['source_sha256'].update(ablation_runner=baseline.runtime.digest(__file__),source_closure=baseline.runtime.source_fingerprint(root,paths))
    payload['ablation_recipe']=dict(cutmix_alpha=1.0 if SPEC['cutmix'] else None,cutmix_switch_probability=.5 if SPEC['cutmix'] else 0,timm_randaugment='rand-m9-mstd0.5-inc1' if SPEC['timm_ra'] else None,terminal_denominator='mean(abs(u)^2).clamp_min(1e-6).detach()' if SPEC['normalize'] else None)
    return json.loads(json.dumps(payload))

def install():
    from va_ablation_augment import install as install_augment
    baseline.runtime.configure(VARIANTS,SEEDS)
    baseline.heads.VARIANTS=VARIANTS;baseline.heads.SEEDS=SEEDS
    baseline.structured._training_objective=baseline.heads._training_objective
    baseline.structured._after_training_batch=baseline.heads._after_training_batch
    install_augment(cutmix=SPEC['cutmix'],timm_ra=SPEC['timm_ra'])
    raw.VARIANT=VARIANT

def _build_optimizer(model,recipe):
    return raw._build_optimizer(model,recipe)

def main():
    install()
    torch.set_float32_matmul_precision('high');torch.backends.cudnn.benchmark=True
    baseline.harness.main(baseline.harness.runner_bindings(variants=VARIANTS,seeds=SEEDS,model_config=baseline.ComplexScanConfig,build_model=_build,contract=_contract,build_optimizer=_build_optimizer,prepare_model=baseline.prepare._prepare_model,train_epoch=baseline.structured._train_epoch,evaluate=baseline.heads._evaluate,wandb_model_metrics=baseline._wandb_model_metrics,summarize=baseline.heads._summarize))

if __name__=='__main__':main()
