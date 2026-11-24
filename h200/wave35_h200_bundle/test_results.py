"""Offline checks for Wave 35 selection, classification and transport."""

# ruff: noqa: S101
import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path

from audit import PREFIXES, SUBGROUPS, classify
from results import ARMS, BASE, OFF, select, transport


def test_classification() -> None:
    for name, group in PREFIXES.items():
        assert classify(name) == (SUBGROUPS[group], group)
    assert classify("void reader_dweight<half, 4>") == ("factorized_reader", "reader_dweight")
    assert classify("reduce_hscan_partials") == (None, None)
    assert classify("path_mlp_backward") == ("general_vscan", None)
    assert classify("unrelated") == (None, None)


def row(ms: float, allocated: float = 130, **kwargs: object) -> dict:
    return {
        "status": "ok", "median_ms": ms, "peak_allocated_gib": allocated,
        "finite_losses": True, "finite_gradients": True, **kwargs,
    }


def test_selection_all_partial_and_none() -> None:
    all_win = dict.fromkeys(ARMS, row(10))
    all_win[BASE] = row(11)
    all_win["w35a_decoupled_simt"] = row(9)
    assert select(all_win)["presets"] == [
        "vscan_decoupled_simt,hscan_vec2", "dweight_rows4,fwd_dinput_vec4",
        "stem_v2,swiglu_v3,rmsnorm_bwd_inductor,inductor_tune",
    ]
    partial = dict.fromkeys(ARMS, row(10))
    partial["w35a_coop_bwd"] = row(9.85)
    partial["w35a_decoupled_simt"] = row(8, allocated=130.01)
    partial["w35b_dweight_rt4"] = row(9.85)
    for name, reader_ms in ((BASE, 4), ("w35b_fwd_dinput_vec4", 3)):
        partial[name] = row(10, profiler={"major_kernel_ms": {"reader_forward": reader_ms}})
    selection = select(partial)
    assert selection["presets"] == ["vscan_coop_bwd", "off", "off"]
    assert selection["reader_forward_won"]
    assert selection["reader_forward_delta_ms"] == -1
    assert tuple(select(dict.fromkeys(ARMS, row(10)))["presets"]) == OFF


def test_transport_roundtrip(tmp_path: Path) -> None:
    data = tmp_path / "input"
    data.mkdir()
    source = {"status": "ok", "profiler": {"top_kernels": [{"name": "test", "count": 3}]}}
    (data / "arm.json").write_text(json.dumps(source))
    transport(data, tmp_path)
    archive = (tmp_path / "task128.tar.xz").read_bytes()
    encoded = base64.b64encode(archive)
    assert len(encoded) < 45_000
    decoded = base64.b64decode(encoded, validate=True)
    assert hashlib.sha256(decoded).digest() == hashlib.sha256(archive).digest()
    with tarfile.open(fileobj=io.BytesIO(decoded), mode="r:xz") as tar:
        assert json.load(tar.extractfile("arm.json")) == source
