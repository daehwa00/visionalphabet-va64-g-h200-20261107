"""Shared CUDA driver and NVRTC utilities for native LNet kernels."""

from __future__ import annotations

import ctypes as ct
import sys
from functools import cache, lru_cache
from importlib.util import find_spec
from pathlib import Path

_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8


def _load_cuda_driver() -> ct.CDLL:
    return ct.CDLL("libcuda.so.1")


def _check_nvrtc(nvrtc: ct.CDLL, status: int, prog: ct.c_void_p | None = None) -> None:
    if status == 0:
        return
    log_msg = ""
    if prog is not None:
        log_size = ct.c_size_t(0)
        if nvrtc.nvrtcGetProgramLogSize(prog, ct.byref(log_size)) == 0 and log_size.value > 1:
            buf = ct.create_string_buffer(log_size.value)
            nvrtc.nvrtcGetProgramLog(prog, buf)
            log_msg = "\n" + buf.value.decode("utf-8", errors="replace")
    raise RuntimeError(f"NVRTC failed with status {status}{log_msg}")


def _check_cu(status: int, name: str) -> None:
    if status != 0:
        raise RuntimeError(f"CUDA driver call {name} failed with status {status}")


def _package_dirs(name: str) -> tuple[Path, ...]:
    try:
        spec = find_spec(name)
    except ModuleNotFoundError:
        return ()
    return tuple(Path(p) for p in (spec.submodule_search_locations or ())) if spec else ()


@lru_cache
def _nvrtc() -> ct.CDLL:
    for root in (*_package_dirs("nvidia.cuda_nvrtc"), Path("/usr/local/cuda")):
        for directory in ("lib", "lib64"):
            for name in ("libnvrtc.so.12", "libnvrtc.so.13", "libnvrtc.so"):
                candidate = root / directory / name
                if candidate.is_file():
                    return ct.CDLL(str(candidate))
    return ct.CDLL("libnvrtc.so")


def _include_dirs() -> tuple[Path, ...]:
    roots = (*_package_dirs("nvidia.cuda_runtime"), Path("/usr/local/cuda"), Path(sys.prefix))
    return tuple(root / "include" for root in roots if (root / "include/cuda_fp16.h").is_file())


@cache
def _primary_context(device: int) -> ct.c_void_p:
    context = ct.c_void_p()
    driver = _load_cuda_driver()
    _check_cu(
        driver.cuDevicePrimaryCtxRetain(ct.byref(context), device),
        "cuDevicePrimaryCtxRetain",
    )
    return context


def _ensure_context(driver: ct.CDLL, device: int) -> None:
    # Autograd threads may only have a logical runtime device until their
    # first non-cached CUDA allocation. The driver needs a current context.
    current = ct.c_void_p()
    _check_cu(driver.cuCtxGetCurrent(ct.byref(current)), "cuCtxGetCurrent")
    if not current:
        _check_cu(driver.cuCtxSetCurrent(_primary_context(device)), "cuCtxSetCurrent")


class _CudaModule:
    def __init__(self, image: ct.Array, device: int) -> None:
        self.driver = _load_cuda_driver()
        _ensure_context(self.driver, device)
        self._module = ct.c_void_p()
        self.functions: dict[str, ct.c_void_p] = {}
        _check_cu(
            self.driver.cuModuleLoadData(ct.byref(self._module), image),
            "cuModuleLoadData",
        )

    def load_function(self, name: str, smem_bytes: int) -> None:
        func = ct.c_void_p()
        _check_cu(
            self.driver.cuModuleGetFunction(
                ct.byref(func), self._module, name.encode("utf-8")
            ),
            f"cuModuleGetFunction({name})",
        )
        if smem_bytes > 49152:
            _check_cu(
                self.driver.cuFuncSetAttribute(
                    func,
                    ct.c_int(_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES),
                    ct.c_int(smem_bytes),
                ),
                f"cuFuncSetAttribute({name}, {smem_bytes})",
            )
        self.functions[name] = func

    def launch(
        self,
        name: str,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        smem_bytes: int,
        stream: int,
        args: list[ct._SimpleCData],
    ) -> None:
        ptrs = (ct.c_void_p * len(args))(*(ct.addressof(a) for a in args))
        _check_cu(
            self.driver.cuLaunchKernel(
                self.functions[name],
                ct.c_uint(grid[0]),
                ct.c_uint(grid[1]),
                ct.c_uint(grid[2]),
                ct.c_uint(block[0]),
                ct.c_uint(block[1]),
                ct.c_uint(block[2]),
                ct.c_uint(smem_bytes),
                ct.c_void_p(stream),
                ptrs,
                ct.c_void_p(0),
            ),
            f"cuLaunchKernel({name})",
        )
