"""Readiness checks only. This program never submits or launches training."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path


def verify_source(root: Path) -> int:
    manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
    for relative, expected in manifest['sha256'].items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('unsafe source manifest member')
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'source hash mismatch: {relative}')
    return len(manifest['sha256'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--inspect-only', action='store_true')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    count = verify_source(args.source_root)
    if args.inspect_only:
        print(json.dumps({'status': 'source_verified_only', 'files': count, 'gpu_checked': False}))
        return
    if args.data_root is None or args.output_root is None:
        parser.error('GPU preflight requires explicit --data-root and --output-root')
    if args.output_root.exists():
        raise FileExistsError('use a fresh preflight output directory')
    import importlib.metadata
    import platform
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('one visible CUDA GPU is required')
    gpu = torch.cuda.get_device_properties(0)
    if 'H200' not in gpu.name or gpu.total_memory < 120 * 1024**3:
        raise RuntimeError('expected one whole H200; received a different GPU/allocation')
    versions = {key: importlib.metadata.version(key) for key in ('torch', 'torchvision', 'timm', 'triton')}
    expected = {'torch': '2.9.1', 'torchvision': '0.24.1', 'timm': '1.0.26', 'triton': '3.5.1'}
    if any(versions[k].split('+')[0] != v for k, v in expected.items()):
        raise RuntimeError(f'environment differs from the 4090 experiment: {versions}')
    if platform.python_version() != '3.13.11':
        raise RuntimeError('Python 3.13.11 is required to match the current environment')
    args.output_root.mkdir(parents=True)
    sys.path[:0] = [str(args.source_root / 'src'), str(args.source_root / 'scripts')]
    from validate_va64_roundtrip import dataset_manifest
    dataset_manifest(args.data_root, args.output_root / 'dataset-manifest.json')
    data = json.loads((args.output_root / 'dataset-manifest.json').read_text())
    expected_membership = {'train': 'b62dac2234295fc54b6a75f018be819d49253eaf36dbc3921e2ab0b249f869ed', 'val': '26dd2edd32ac77f6069b1d0e4b59664f9ac2e9500ab5be3a7108c84aae4681ae'}
    if data['membership_size_sha256'] != expected_membership:
        raise RuntimeError('ImageNet file membership/size differs from the 4090 experiment')
    report = {'status': 'environment_and_data_checked', 'training_started': False,
              'python': platform.python_version(), 'versions': versions,
              'gpu': gpu.name, 'gpu_memory_bytes': gpu.total_memory,
              'cuda_runtime': torch.version.cuda,
              'remaining': ['model forward/backward and checkpoint roundtrip',
                            'external checkpoint retrieval and restore',
                            'effective remote stop and resource return']}
    (args.output_root / 'preflight.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
