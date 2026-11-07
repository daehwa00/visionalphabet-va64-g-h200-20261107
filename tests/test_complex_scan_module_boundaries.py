from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from lnet import complex_scan as facade
from lnet import complex_scan_config as config_module
from lnet import complex_scan_stage as bank
from lnet import complex_scan_transitions as transitions
from lnet import complex_scan_types as contracts
from lnet.complex_scan import ComplexScanBackbone, ComplexScanConfig

_PACKED_IMPORT_ORDER = (
    "pac_packed_postcarry_inference, complex_scan_transitions, complex_scan_stage, complex_scan"
)


def _schema_digest(config: ComplexScanConfig) -> str:
    model = ComplexScanBackbone(config)
    state_schema = [
        (name, tuple(value.shape), str(value.dtype)) for name, value in model.state_dict().items()
    ]
    parameter_order = [name for name, _ in model.named_parameters()]
    payload = json.dumps([state_schema, parameter_order], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (
            ComplexScanConfig(
                output_dim=5,
                stem_width=8,
                stem_strides=(1, 1),
                modes=(4, 4, 4),
                augmented_widths=(8, 8),
                carry_bases=("s2d", "s2d"),
                fusion_width=8,
            ),
            "87e019b14d45f3f1043d3a2cbc395f66f5577f132a0a4794f0f1ed0b2f2fe61b",
        ),
        (
            ComplexScanConfig(
                output_dim=5,
                stem_width=8,
                stem_strides=(1, 1),
                modes=(4, 4, 4),
                quadrant_path_mode_cffn_widths=(8, 8),
                quadrant_path_cffn_widths=(4, 4),
            ),
            "ae94188b7a595cbea6c6c84614c02bc7925ab5a8cc3a67011ceaa07612f8f8ac",
        ),
    ],
)
def test_refactor_preserves_state_schema_and_parameter_order(
    config: ComplexScanConfig,
    expected: str,
) -> None:
    assert _schema_digest(config) == expected


@pytest.mark.parametrize(
    ("module", "symbols"),
    [
        (
            contracts,
            ("ComplexField", "DirectionalState", "ComplexCarryBasis", "ComplexCarryMerge"),
        ),
        (
            transitions,
            (
                "ComplexRMSNorm",
                "ComplexModulatedTransition",
                "S2DPostCFFNCarryMainTransition",
                "ComplexResidualFFN",
            ),
        ),
        (
            bank,
            (
                "ComplexCCCNDownsample",
                "ComplexScanStage",
            ),
        ),
        (config_module, ("ComplexScanConfig",)),
    ],
)
def test_public_module_reexports_implementation_symbols(
    module: object,
    symbols: tuple[str, ...],
) -> None:
    for symbol in symbols:
        assert getattr(facade, symbol) is getattr(module, symbol)


def test_internal_modules_do_not_import_the_public_module() -> None:
    for module in (contracts, transitions, bank, config_module):
        module_path = module.__file__
        assert module_path is not None
        source = Path(module_path).read_text()
        assert "from .complex_scan import" not in source
        assert "import lnet.complex_scan" not in source


@pytest.mark.parametrize(
    "imports",
    [
        _PACKED_IMPORT_ORDER,
        "complex_scan_transitions, complex_scan, pac_packed_postcarry_inference",
        "complex_scan_stage, complex_scan, a2d_q_heads",
    ],
)
def test_supported_import_orders_start_in_a_fresh_process(imports: str) -> None:
    statements = "; ".join(f"import lnet.{name.strip()}" for name in imports.split(","))
    environment = os.environ | {"PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    subprocess.run(
        [sys.executable, "-c", statements],
        check=True,
        env=environment,
    )


def test_state_dict_strict_round_trip_preserves_output() -> None:
    active = ComplexScanConfig(
        output_dim=5,
        stem_width=8,
        stem_strides=(1, 1),
        modes=(4, 4, 4),
        augmented_widths=(8, 8),
        carry_bases=("s2d", "s2d"),
        fusion_width=8,
    )
    torch.manual_seed(991)
    source = ComplexScanBackbone(active).eval()
    torch.manual_seed(992)
    restored = ComplexScanBackbone(active).eval()
    restored.load_state_dict(source.state_dict(), strict=True)
    inputs = torch.randn(2, 3, 8, 8)
    with torch.inference_mode():
        torch.testing.assert_close(restored(inputs), source(inputs), rtol=0.0, atol=0.0)
