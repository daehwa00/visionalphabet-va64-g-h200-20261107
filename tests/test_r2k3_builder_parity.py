from __future__ import annotations

# pyright: reportPrivateUsage=false
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts import a2d_r2k3_runtime as runtime
from scripts import run_a2d_r2k3_same_resolution_factorial_imagenet100 as runner


def test_runtime_import_does_not_eagerly_load_legacy_training_graph() -> None:
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{root / 'src'}:{root / 'scripts'}"
    legacy = "run_a2d_pgv2_h96_rank2_polewise_k3_q4_nopg_short_pole_termmax20_rawq_imagenet100"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys, a2d_r2k3_runtime; assert {legacy!r} not in sys.modules",
        ],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_model_build_does_not_load_legacy_training_graph() -> None:
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{root / 'src'}:{root / 'scripts'}"
    legacy = "run_a2d_pgv2_h96_rank2_polewise_k3_q4_nopg_short_pole_termmax20_rawq_imagenet100"
    source = (
        "import sys, torch; "
        "import a2d_r2k3_runtime as runtime; "
        "import run_a2d_r2k3_same_resolution_factorial_imagenet100 as runner; "
        "torch.manual_seed(501); "
        "runner._build('K64-PF15K-Base', runtime.model_config()); "
        f"assert {legacy!r} not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_source_manifest_keeps_lazy_legacy_training_dependency() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = runtime.source_dependency_paths(root, ("a2d_r2k3_runtime",))
    names = {path.stem for path in paths}
    assert (
        "run_a2d_pgv2_h96_rank2_polewise_k3_q4_nopg_short_pole_termmax20_rawq_imagenet100" in names
    )


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize(
    ("variant", "parameters", "state_entries", "expected_digest"),
    [
        (
            "K64-PF15K-Base",
            408_868,
            124,
            "813f76bb23457987907b4396dbdafdd9127417eb9befccd733836cdc540d29f3",
        ),
        (
            "K96-PF15K-Base",
            837_028,
            124,
            "8b52678485240afcc831d9b061691004c0c66baa4df3748e28ed38590c6bc348",
        ),
        (
            "K128-PF15K-Base",
            1_414_692,
            124,
            "8a92e7b35a2ceff32d1aee9303e106527d8e9a4876b5bc8ae069f1193a47a198",
        ),
        (
            "K128-PF15K-SR56-28-14-7",
            2_772_516,
            248,
            "3395266491841aeabb49ac3603c6c1088b7dc0d46433c6e50fe61c3ed0d9d92c",
        ),
    ],
)
def test_seed501_builder_state_is_frozen(
    variant: str,
    parameters: int,
    state_entries: int,
    expected_digest: str,
) -> None:
    # CPU QR is allowed to choose different reduction order for different OMP
    # limits.  These historical digests were emitted by the canonical 32-thread
    # construction environment, so pin that test-only context and then restore
    # the suite's configured thread count.
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(32)
        torch.manual_seed(501)
        model = runner._build(variant, runtime.model_config())
        actual_parameters = sum(parameter.numel() for parameter in model.parameters())
        actual_state_entries = len(model.state_dict())
        actual_digest = _state_digest(model)
    finally:
        torch.set_num_threads(previous_threads)

    assert actual_parameters == parameters
    assert actual_state_entries == state_entries
    assert actual_digest == expected_digest
