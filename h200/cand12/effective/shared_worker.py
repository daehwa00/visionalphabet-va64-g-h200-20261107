"""Isolated, checkpoint-reset full-effective-batch benchmark worker."""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path
import statistics
import time
import traceback


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ('checkpoint', 'contract', 'fixture', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--microbatch', type=int, required=True)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--groups', type=int, default=3)
    p.add_argument('--samples', type=int, default=2)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--capture', action='store_true')
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(status='initializing', microbatch=a.microbatch, effective_batch=1024,
                  checkpoint_sha256=sha(a.checkpoint), contract_sha256=sha(a.contract),
                  fixture_sha256=sha(a.fixture), worker_sha256=sha(__file__),
                  warmup=a.warmup, groups=a.groups, samples=a.samples)
    def save():
        a.output.write_text(json.dumps(result, indent=2) + '\n')
    save()
    try:
        import torch
        import numpy as np
        import run_va_eight_ablations as target
        import va_imagenet100_runtime as rt
        import va_model_builder as mb
        import effective_batch_update as update
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
        recipe = json.loads(a.contract.read_text())['recipe']
        assert (checkpoint['epoch'], checkpoint['global_step']) == (18, 18270)
        torch.manual_seed(checkpoint['seed'])
        model = target._build(target.VARIANT, mb.model_config()).cuda().train()
        model.load_state_dict(checkpoint['model'], strict=True)
        model = rt.prepare_model(model, recipe)
        optimizer = target._build_optimizer(model, recipe)
        optimizer.load_state_dict(checkpoint['optimizer'])
        os.environ['LNET_COMPILE_MODE'] = 'default'
        runtime = rt._build_runtime(model, recipe)
        if a.fixture.suffix == '.json':
            meta = json.loads(a.fixture.read_text())
            folder = a.fixture.parent
            x = torch.load(folder / 'images.pt', map_location='cpu', weights_only=True)
            y = torch.load(folder / 'labels.pt', map_location='cpu', weights_only=True)
            permutation = torch.load(folder / 'permutation.pt', map_location='cpu', weights_only=True)
            mix = meta['mixup']['lambda']
            fixture = {'x': float(mix)*x + (1-float(mix))*x[permutation],
                       'y': y, 'other': y[permutation], 'mix': mix}
            result['fixture_tensors_sha256'] = {n:sha(folder/n) for n in ('images.pt','labels.pt','permutation.pt')}
        else:
            fixture = torch.load(a.fixture, map_location='cpu', weights_only=False)
        # The fixture contract accepts explicit preprocessed/mixed tensors only.
        if isinstance(fixture, dict):
            batch = (fixture['x'], fixture['y'], fixture['other'], fixture['mix'])
        else:
            batch = fixture
        x, y, other, mix = batch
        assert len(x) == len(y) == len(other) == 1024
        batch = (x.cuda().contiguous(memory_format=torch.channels_last),
                 y.cuda(), other.cuda(), float(mix))
        result.update(epoch=checkpoint['epoch'], global_step=checkpoint['global_step'],
                      gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                      parameters=sum(v.numel() for v in model.parameters()),
                      precision='bfloat16', primitive_sha256=sha(update.__file__),
                      learning_rates=[g['lr'] for g in optimizer.param_groups],
                      scope='whole 1024-image update: forward/objective/backward/clip/fused AdamW; reset excluded')
        source_root = Path(target.__file__).resolve().parents[1]
        closure = {str(p.relative_to(source_root)):sha(p) for folder in ('src/lnet','scripts')
                   for p in sorted((source_root/folder).glob('*.py'))}
        closure.update({str(p.relative_to(source_root)):sha(p)
                        for p in sorted((source_root/'src/lnet').glob('validated*.json'))})
        result['source_closure_sha256'] = hashlib.sha256(json.dumps(closure,sort_keys=True).encode()).hexdigest()
        result['workload_sha256'] = hashlib.sha256(json.dumps({k:result[k] for k in (
            'microbatch','effective_batch','checkpoint_sha256','contract_sha256','fixture_sha256',
            'primitive_sha256','precision','learning_rates')},sort_keys=True).encode()).hexdigest()
        a.output.with_suffix('.source.json').write_text(json.dumps(closure,indent=2)+'\n')
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone()
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        def restore():
            model.load_state_dict(checkpoint['model'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer.zero_grad(set_to_none=True)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
        def step(capture=False):
            return update.run_update(model, runtime, optimizer, batch, a.microbatch,
                                     'bfloat16', capture=capture, metrics=True)
        result['status'] = 'compiling'; save()
        start = time.perf_counter()
        for _ in range(a.warmup):
            restore()
            step()
            torch.cuda.synchronize()
        result['warmup_seconds'] = time.perf_counter() - start
        result['setup_peak_mib'] = torch.cuda.max_memory_allocated() / 2**20
        if a.capture:
            restore()
            captured = step(True)
            torch.cuda.synchronize()
            torch.save(captured, a.output.with_suffix('.capture.pt'))
            result['capture_keys'] = list(captured)
            checks = captured['checks']
            result['capture_counts'] = {k:len(v) for k,v in checks.items()}
            result['capture_finite'] = all(bool(t.isfinite().all()) for v in checks.values() for t in v.values())
            if len(checks['outputs']) != 5 or len(checks['grads']) != 193 or not result['capture_finite']:
                raise RuntimeError('Capture must contain five finite outputs and all 193 gradients')
            result['status'] = 'captured'; save()
        walls, peaks = [], []
        for _ in range(a.groups):
            group = []
            for _ in range(a.samples):
                restore()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                step()
                torch.cuda.synchronize()
                group.append((time.perf_counter() - start) * 1000)
                peaks.append(torch.cuda.max_memory_allocated() / 2**20)
            walls.append(group)
            result.update(status='timing', wall_samples_ms=walls); save()
        median = statistics.median(map(statistics.mean, walls))
        result.update(median_ms=median, images_s=1024000 / median,
                      peak_allocated_mib=max(peaks), reserved_mib=torch.cuda.memory_reserved()/2**20)
        if a.profile:
            restore(); torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True) as prof:
                step(); torch.cuda.synchronize()
            prof.export_chrome_trace(str(a.output.with_suffix('.trace.json')))
            a.output.with_suffix('.profile.txt').write_text(
                prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=80))
        result['status'] = 'passed'
    except Exception as error:
        result.update(status='oom' if 'out of memory' in str(error).lower() else 'error',
                      error=str(error), traceback=traceback.format_exc())
        save()
        raise
    save()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
