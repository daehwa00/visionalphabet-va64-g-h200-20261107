"""Native CUDA smoke cases mirrored from the two Reader regression test modules."""
# These cases intentionally mirror pytest assertions and probe private dispatch APIs.
# ruff: noqa: ANN001, ANN201, ANN202, BLE001, FBT001, S101, SLF001

import argparse
import json
import os
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from bench_reader_e2e_micro import SHAPES, make_reader

from lnet import memory_backbone
from lnet.factorized_complex_scan_reader import FactorizedComplexConv2dReader
from lnet.memory_backbone import FactorizedMemoryReader


class Overrides(ExitStack):
    def setattr(self, obj, name, value):
        self.enter_context(patch.object(obj, name, value))

    def setenv(self, key, value):
        self.enter_context(patch.dict(os.environ, {key: value}))

    def delenv(self, key):
        self.enter_context(patch.dict(os.environ))
        os.environ.pop(key, None)


def dispatch_case(shape, expected, monkeypatch) -> None:
    reader = make_reader(shape, torch.device("cpu"))
    reader.execution_mode = "dense"
    source = SimpleNamespace(
        is_cuda=True,
        device=torch.device("cuda:1"),
        dtype=torch.bfloat16,
        shape=(shape.batch, shape.height, shape.width, shape.channels),
    )

    sm = (9, 0, 132)

    def properties(index):
        assert index == 1
        return SimpleNamespace(major=sm[0], minor=sm[1], multi_processor_count=sm[2])

    monkeypatch.setenv("LNET_READER_EXEC_MODE", "auto_roofline")
    runtime = SimpleNamespace(**vars(source))
    runtime.device = torch.device("cuda", torch.cuda.current_device())
    assert reader._reader_execution_mode(runtime) == expected

    monkeypatch.setattr(memory_backbone, "_reader_cuda_device_properties", properties)
    monkeypatch.setenv("LNET_READER_EXEC_MODE", "auto_roofline")
    assert reader._reader_execution_mode(source) == expected

    def dispatch(value):
        return value + (1 if reader._reader_execution_mode(source) == "dense_nhwc" else 2)

    compiled_dispatch = torch.compile(dispatch, backend="aot_eager", fullgraph=True)
    assert compiled_dispatch(torch.zeros((), device="cuda")) == (
        1 if expected == "dense_nhwc" else 2
    )
    # The compiled graph specializes once. Eager dispatch must still read the
    # current properties after that specialization has populated Dynamo's cache.
    sm = (7, 0, 80)
    assert reader._reader_execution_mode(source) == "factorized_triton"
    torch._dynamo.reset()
    assert (
        torch.compile(dispatch, backend="aot_eager", fullgraph=True)(torch.zeros((), device="cuda"))
        == 2
    )
    monkeypatch.setenv("LNET_READER_EXEC_MODE", "off")
    assert reader._reader_execution_mode(source) == "dense"


def _check_gradients(output, reference, inputs, reference_inputs, *, rtol, atol) -> None:
    probe = torch.randn_like(output, dtype=torch.float32) / output.shape[-1]
    gradients = torch.autograd.grad(output.float(), inputs, probe)
    expected_gradients = torch.autograd.grad(reference.float(), reference_inputs, probe)
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual.float(), expected.float(), rtol=rtol, atol=atol)


def parity_case(
    compiled: bool,
    dtype: torch.dtype,
    fused: bool,
    monkeypatch: Overrides,
) -> None:
    torch.manual_seed(972)
    legacy = FactorizedComplexConv2dReader(
        13,
        48,
        rank=2,
        kernel_size=3,
        execution_mode="factorized_triton",
    )
    reader = FactorizedMemoryReader(legacy).cuda()
    reference = deepcopy(reader)
    reference.execution_mode = "factorized"
    with torch.no_grad():
        # Break the input factors' original complex symmetry.
        reader.point_weight.normal_(std=0.2)
        reader.spatial_weight.normal_(std=0.15)
    reference.load_state_dict(reader.state_dict())
    if fused:
        monkeypatch.delenv("LNET_FUSED_FACTORIZED_READER")
    else:
        monkeypatch.setenv("LNET_FUSED_FACTORIZED_READER", "0")
    run = torch.compile(reader, fullgraph=True) if compiled else reader
    x = torch.randn(2, 7, 11, 26, device="cuda", requires_grad=True)
    y = x.detach().clone().requires_grad_()

    def unexpected_synthesis() -> None:
        raise AssertionError("Fused real Reader called synthesized_kernel")

    def unexpected_fused_spatial(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Opted-out Reader used fused spatial convolution")

    if fused:
        monkeypatch.setattr(reader, "synthesized_kernel", unexpected_synthesis)
    else:
        monkeypatch.setattr(memory_backbone, "real_packed_spatial", unexpected_fused_spatial)
    autocast_dtype = torch.bfloat16 if dtype == torch.float32 else dtype
    with torch.autocast("cuda", dtype=autocast_dtype, enabled=dtype != torch.float32):
        expected = reference(y)
        output = run(x)
    atol, rtol = {
        torch.float32: (5e-6, 3e-5),
        torch.float16: (8e-4, 5e-3),
        torch.bfloat16: (4e-3, 3e-2),
    }[dtype]
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)
    _check_gradients(
        output,
        expected,
        (x, *reader.parameters()),
        (y, *reference.parameters()),
        rtol=max(rtol, 5e-5),
        atol=max(atol, 1e-5),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    cases = [
        ("auto_roofline_dense", dispatch_case, (SHAPES[0], "dense_nhwc")),
        ("auto_roofline_factorized", dispatch_case, (SHAPES[-1], "factorized_triton")),
        *(
            (f"inductor_{mode}_{dtype}", parity_case, (True, dtype, mode == "default"))
            for mode in ("default", "opt-out")
            for dtype in (torch.float32, torch.float16, torch.bfloat16)
        ),
    ]
    results = {}
    for name, case, params in cases:
        print(name, flush=True)  # noqa: T201
        torch._dynamo.reset()
        try:
            with Overrides() as overrides:
                case(*params, overrides)
        except Exception as error:
            results[name] = f"{type(error).__name__}: {error}"[:500]
        else:
            results[name] = "passed"
        finally:
            torch._dynamo.reset()
        passed = sum(value == "passed" for value in results.values())
        report = {"passed": passed, "failed": len(results) - passed, "cases": results}
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    return int(passed != len(cases))


if __name__ == "__main__":
    raise SystemExit(main())
