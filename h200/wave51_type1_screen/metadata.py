"""Record the hardware and software used by the screening runner."""
import json
import platform
import subprocess

import torch
import triton

from lnet import general_cuda_scans

props = torch.cuda.get_device_properties(0)
print(json.dumps({  # noqa: T201
    "gpu": props.name,
    "sm_count": props.multi_processor_count,
    "vram_bytes": props.total_memory,
    "l2_bytes": general_cuda_scans._l2_size(0),  # noqa: SLF001
    "python": platform.python_version(),
    "pytorch": torch.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "triton": triton.__version__,
    "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),  # noqa: S607
}, indent=2))
