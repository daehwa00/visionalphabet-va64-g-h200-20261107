"""Profile unchanged official RMT with the paper's complete forward FLOP rules.

The frozen auditor supplies pointwise/reduction/normalization rules; PyTorch's
FlopCounterMode supplies MACs at two real FLOPs each. All high-level tensor
calls are classified, and unknown calls invalidate (rather than lower) totals.
This is an uncompiled, eval-mode, FP32, CPU arithmetic count, not latency.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
from torch.overrides import TorchFunctionMode
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDITOR = ROOT / 'releases/submission/anonymous-code.GBcMHI/visionalphabet/efficiency/rebuttal_full_flops_audit.py'


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def describe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {'shape': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, (list, tuple)):
        return [describe(item) for item in value]
    if isinstance(value, dict):
        return {key: describe(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# Metadata and discrete/indexing-only operations do not perform real arithmetic.
EXTRA_ZERO = {
    '__get__', '__getitem__', 'size', 'dim', 'numel', 'is_contiguous',
    'is_floating_point', 'get_device', 'arange', 'meshgrid', 'repeat',
    'type_as', 'float', 'int', 'long', 'item', 'unbind', 'broadcast_tensors',
}
ALIASES = {
    '__add__': 'add', '__radd__': 'add', '__sub__': 'sub', '__rsub__': 'rsub',
    '__mul__': 'mul', '__rmul__': 'mul', '__truediv__': 'div',
    '__neg__': 'neg', '__matmul__': 'matmul',
}


def classify(event: dict, auditor: Any) -> tuple[str, int]:
    """Use original rules unchanged; explicitly extend only absent RMT calls."""
    name = event['op'].split('.')[1]
    if name in EXTRA_ZERO:
        return 'zero_cost', 0
    if name == 'abs':
        return 'scalar_aten', auditor.tensor_numel(event['output'])
    if name == 'pow':
        # A scalar power is one scalar special-function equivalent, same as exp.
        return 'scalar_aten', auditor.tensor_numel(event['output'])
    if name in {'mm', 'bmm'}:
        return 'core_registry', 0
    return auditor.aten_flops(event)


class ArithmeticCensus(TorchFunctionMode):
    def __init__(self, auditor: Any):
        super().__init__()
        self.auditor = auditor
        self.components: dict[str, int] = defaultdict(int)
        self.histogram: Counter[str] = Counter()
        self.unsupported: Counter[str] = Counter()
        self.examples: dict[str, dict] = {}

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        output = func(*args, **kwargs)
        name = ALIASES.get(func.__name__, func.__name__)
        event = {'op': f'aten.{name}.default', 'args': describe(args),
                 'kwargs': describe(kwargs), 'output': describe(output)}
        # Functional batch_norm orders running stats before weight/bias;
        # the frozen ATen audit expects input, weight, bias, running stats.
        if name == 'batch_norm':
            event['args'] = [describe(args[0]),
                             describe(args[3] if len(args) > 3 else kwargs.get('weight')),
                             describe(args[4] if len(args) > 4 else kwargs.get('bias'))]
        self.histogram[event['op']] += 1
        try:
            category, count = classify(event, self.auditor)
        except ValueError:
            self.unsupported[event['op']] += 1
            self.examples.setdefault(event['op'], event)
        else:
            self.components[category] += count
        return output

    def total(self, core: int) -> int:
        if self.unsupported:
            raise ValueError(f'Uncovered operators: {dict(self.unsupported)}')
        if core <= 0:
            raise ValueError('Missing independently counted core arithmetic')
        return sum(self.components.values()) + core


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(model: torch.nn.Module, auditor: Any, image_size: int = 224) -> dict:
    model = model.cpu().float().eval()
    sample = torch.zeros(1, 3, image_size, image_size)
    census = ArithmeticCensus(auditor)
    core = FlopCounterMode(display=False)
    with torch.inference_mode(), core, census:
        output = model(sample)
    core_total = int(core.get_total_flops())
    result = {'params': sum(p.numel() for p in model.parameters()),
              'output_shape': list(output.shape), 'image_size': image_size,
              'batch': 1, 'device': 'cpu', 'dtype': 'float32', 'mode': 'eval',
              'unsupported_operators': dict(sorted(census.unsupported.items())),
              'unsupported_examples': census.examples,
              'operator_histogram': dict(sorted(census.histogram.items())),
              'components': dict(sorted(census.components.items())),
              'core_operator_flops': {str(op): int(count) for op, count in
                                      core.get_flop_counts()['Global'].items()}}
    result['components']['core_registry'] = core_total
    if census.unsupported:
        result.update(valid=False, flops_per_image=None)
    else:
        result.update(valid=True, flops_per_image=census.total(core_total))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rmt-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='RMT_T3')
    parser.add_argument('--auditor', type=Path, default=DEFAULT_AUDITOR)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    auditor = load_module(args.auditor, 'frozen_va_common_flops_auditor')
    official = load_module(args.rmt_source.resolve(), 'official_rmt_common_flops')
    model = getattr(official, args.model)(None)
    result = profile(model, auditor)
    result.update(schema='visionalphabet.rmt_common_forward_flops.v1', model=args.model,
                  convention=auditor.__doc__, torch_version=torch.__version__,
                  provenance={'rmt_source': {'path': str(args.rmt_source), 'sha256': sha256(args.rmt_source)},
                              'auditor': {'path': str(args.auditor), 'sha256': sha256(args.auditor)},
                              'profiler_sha256': sha256(Path(__file__))})
    try:
        result['provenance']['rmt_commit'] = subprocess.check_output(
            ['git', '-C', str(args.rmt_source.parent), 'rev-parse', 'HEAD'], text=True).strip()
    except subprocess.CalledProcessError:
        result['provenance']['rmt_commit'] = None
    reference_path = args.auditor.parent / 'rebuttal-full-flops-audit.json'
    reference_document = json.loads(reference_path.read_text())
    if reference_document['auditor_sha256'] != sha256(args.auditor):
        raise ValueError('Reference audit and active arithmetic rules differ')
    reference = next(row for row in reference_document['models']
                     if row['model'] == 'va_k64')
    result['comparison'] = {'reference_model': 'va_k64', 'reference_params': reference['params'],
                            'reference_flops_per_image': reference['flops_per_image'],
                            'reference_audit_sha256': sha256(reference_path),
                            'same_scalar_and_mac_rules': True,
                            'matched_tolerance_fraction': 0.10}
    if result['valid']:
        ratio = result['flops_per_image'] / reference['flops_per_image']
        result['comparison'].update(rmt_to_va_flops_ratio=ratio,
                                    within_10_percent=abs(ratio - 1) <= 0.10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ['model', 'valid', 'params', 'flops_per_image',
                                                  'unsupported_operators', 'comparison']}, indent=2))
    if not result['valid']:
        raise SystemExit('Incomplete arithmetic census; refusing total FLOPs')


if __name__ == '__main__':
    main()
