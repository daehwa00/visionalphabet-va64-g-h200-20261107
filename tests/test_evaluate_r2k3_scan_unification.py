from __future__ import annotations

# pyright: reportPrivateUsage=false
import evaluate_r2k3_scan_unification as evaluator


def _benchmark(median_ms: float, peak_bytes: int) -> evaluator.BenchmarkResult:
    return {
        "median_ms": median_ms,
        "samples_ms": [median_ms],
        "peak_bytes": peak_bytes,
    }


def test_matrix_declares_only_odd_coarse_as_not_applicable() -> None:
    specs = evaluator._matrix_specs()

    assert len(specs) == 12
    assert [spec for spec in specs if not spec[2]] == [(7, "coarse", False)]
    assert sum(supported for _length, _epilogue, supported in specs) == 11


def test_matrix_gate_requires_every_valid_cell_and_exact_na_record() -> None:
    complete: list[dict[str, object]] = [
        {
            "length": length,
            "epilogue": epilogue,
            "supported": supported,
            "status": "passed" if supported else "not_applicable",
            "passed": supported,
        }
        for length, epilogue, supported in evaluator._matrix_specs()
    ]

    assert evaluator._matrix_passed(complete)
    assert not evaluator._matrix_passed(complete[:-1])
    complete[0]["status"] = "passed"
    assert not evaluator._matrix_passed(complete)


def test_policy_selection_gates_the_selected_policy_not_any_policy() -> None:
    benchmark = {
        "legacy": _benchmark(10.0, 100),
        "recompute": _benchmark(10.1, 100),
        "retain": _benchmark(9.0, 101),
    }
    ratios = evaluator._policy_ratios(benchmark)
    parity = {
        "recompute": {"output_relative_l2": 0.01, "gradient_relative_l2": 0.01},
        "retain": {"output_relative_l2": 0.01, "gradient_relative_l2": 0.01},
    }

    assert evaluator._select_policy(benchmark, ratios, parity) == "recompute"
    assert ratios["retain"]["memory"] > evaluator.MEMORY_LIMIT


def test_provenance_hashes_evaluator_and_remote_gate() -> None:
    provenance = evaluator._source_provenance()
    files = provenance["files"]

    assert isinstance(files, dict)
    assert "scripts/evaluate_r2k3_scan_unification.py" in files
    assert "scripts/run_r2k3_scan_unification_gate.sh" in files
    assert "src/lnet/pac_product_scan_ops.py" in files
    assert "src/lnet/pac_triton_product_scan_kernels.py" in files
    assert "tests/test_pac_product_scan_ops.py" in files
    assert "tests/test_pac_product_scan_module_boundaries.py" in files
    assert len(str(provenance["bundle_sha256"])) == 64
