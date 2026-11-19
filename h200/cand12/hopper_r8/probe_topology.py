"""Record CUDA device properties before checking fixed-input mounts."""
import ctypes as c
import json
import os
from pathlib import Path


def main():
    cuda = c.CDLL("libcuda.so.1")

    def call(symbol, *args):
        code = getattr(cuda, symbol)(*args)
        if code:
            raise RuntimeError(f"{symbol} failed with CUDA driver code {code}")

    device = c.c_int()
    count = c.c_int()
    call("cuInit", 0)
    call("cuDeviceGetCount", c.byref(count))
    if count.value != 1:
        raise RuntimeError(f"Expected one visible CUDA device, got {count.value}")
    call("cuDeviceGet", c.byref(device), 0)
    name = c.create_string_buffer(256)
    total = c.c_size_t()
    call("cuDeviceGetName", name, len(name), device)
    call("cuDeviceTotalMem_v2", c.byref(total), device)

    def attr(index):
        value = c.c_int()
        call("cuDeviceGetAttribute", c.byref(value), index, device)
        return value.value

    # CUdevice_attribute enumerators for compute capability and SM count.
    gib = total.value / 2**30
    mig = gib < 80
    topology = dict(device_name=name.value.decode(), capability=[attr(75), attr(76)],
                    multi_processor_count=attr(16), total_memory_gib=gib,
                    is_mig=mig, pool_mode="mig_1g18gb" if mig else "whole_h200")
    path = Path(os.environ["RUN_ROOT"]) / "topology.json"
    path.write_text(json.dumps(topology))
    print("CUDA_DRIVER_TOPOLOGY", json.dumps(topology), flush=True)
    if topology["capability"] != [9, 0] or topology["pool_mode"] != os.environ["HOPPER_REQUESTED_POOL"]:
        raise RuntimeError("CUDA device capability and topology do not match the requested pool")


if __name__ == "__main__":
    main()
