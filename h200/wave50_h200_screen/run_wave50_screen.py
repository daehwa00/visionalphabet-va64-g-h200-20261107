"""Zero-bootstrap Whole H200 screen and compiled lock-in template."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tarfile
from pathlib import Path
from time import monotonic

ROOT = Path(__file__).resolve().parents[2]
MODULE = "h200.wave50_h200_screen.run_wave50_screen"
BENCH_ENV = {
    "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
    "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
    "TORCHINDUCTOR_AUTOGRAD_CACHE_ALLOW_CUSTOM_AUTOGRAD": "1",
    "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE": "0",
    "TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE": "0",
    "LNET_CUDNN_BENCHMARK": "1",
}


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def stage(
    out: Path,
    name: str,
    argv: list[str],
    timeout: int,
    env: dict[str, str],
    *,
    json_flag: str | None = "--out",
) -> dict:
    target = out / f"{name}.json"
    target.unlink(missing_ok=True)
    cmd = [sys.executable, *argv]
    if json_flag:
        cmd.extend((json_flag, str(target)))
    started = monotonic()
    with (
        (out / f"{name}.log").open("wb") as log,
        subprocess.Popen(  # noqa: S603
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        ) as process,
    ):
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code = 124
    detail = f"exit code {code}"
    if json_flag and code == 0:
        try:
            raw = json.loads(target.read_text())
            if not isinstance(raw, dict) or raw.get("status") == "error":
                code, detail = 1, "invalid result or benchmark error"
        except (OSError, ValueError):
            target.unlink(missing_ok=True)
            code, detail = 1, "missing or malformed JSON result"
    return {
        "status": "ok" if code == 0 else "error",
        "detail": detail,
        "exit_code": code,
        "wall_s": monotonic() - started,
    }


def screen(out: Path, env: dict[str, str]) -> dict:
    scan_flags = {
        "LNET_SCAN_SGS": "strict_ro",
        "LNET_SCAN_VFWD": "batch4_w2",
        "LNET_SCAN_HSCAN": "bwd_opt",
    }
    results = {
        "vscan_e2e": stage(
            out,
            "vscan_e2e",
            [
                "scripts/bench_scan_micro.py",
                "--ops",
                "vscan_e2e",
                "--modes",
                "off,saved_fused,saved_split,auto_vram",
                "--batch",
                "1024",
                "--dtype",
                "bf16",
                "--warmup",
                "1",
                "--iters",
                "3",
                "--flush-l2",
                "--parity",
            ],
            150,
            {**env, **scan_flags},
        ),
        "reader_compiled": stage(
            out,
            "reader_compiled",
            [
                "scripts/bench_reader_e2e_micro.py",
                "--compiled",
                "--mode",
                "graph",
                "--modes",
                "off_bench,dense_nhwc_inductor_bench,dense_nhwc_bench,auto_roofline_bench",
                "--dtypes",
                "bf16,fp16",
                "--timing-dtype",
                "bf16",
                "--model-shapes-only",
                "--warmup",
                "1",
                "--iterations",
                "3",
            ],
            150,
            env,
            json_flag="--json-out",
        ),
    }
    for name, script, modes in (
        ("stem_stage0", "scripts/bench_stem_micro.py", "off,cuda_tiled"),
        ("rmsnorm_fwd", "scripts/bench_norm_micro.py", "off,fast_fwd"),
    ):
        results[name] = stage(
            out,
            name,
            [script, "--modes", modes, "--flush-l2", "--parity"],
            60,
            env,
        )
    name = "pytest_reader_fix"
    results[name] = stage(
        out,
        name,
        [
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            "-rA",
            "tests/test_reader_dense_nhwc.py",
            "tests/test_factorized_reader_kernels.py",
            "-k",
            "test_auto_roofline_dispatch_uses_runtime_properties"
            " or test_free_real_memory_reader_matches_grouped_cuda",
        ],
        90,
        env,
        json_flag=None,
    )
    log = (out / f"{name}.log").read_text(errors="replace")
    counts = dict.fromkeys(("passed", "failed", "skipped", "deselected", "error"), 0)
    for count, label in re.findall(r"(\d+) (passed|failed|skipped|deselected|error)", log):
        counts[label] = int(count)
    write_json(out / f"{name}.json", {**results[name], **counts, "log_tail": log[-4000:]})
    for name, result in results.items():
        path = out / f"{name}.json"
        if not path.exists():
            log = (out / f"{name}.log").read_text(errors="replace")
            write_json(path, {**result, "log_tail": log[-4000:]})
    return results


def compiled_arm(out: Path) -> int:
    import torch  # noqa: PLC0415
    from torch._dynamo.utils import counters  # noqa: PLC0415

    from h200.liverms_v2_opt.bench_h200 import train_arm  # noqa: PLC0415
    from h200.wave41_h200_bundle.bench_wave41_h200 import VARIANT  # noqa: PLC0415

    if not (
        torch._inductor.config.fx_graph_cache  # noqa: SLF001
        and torch._functorch.config.enable_autograd_cache  # noqa: SLF001
        and torch._functorch.config.autograd_cache_allow_custom_autograd_functions  # noqa: SLF001
    ):
        raise RuntimeError("FX and AOT Autograd caches must be enabled before importing PyTorch")
    result = train_arm(argparse.Namespace(variant=VARIANT, batch=1024, profile=False, out=out))
    result["cache"] = {
        "aot_autograd": dict(counters["aot_autograd"]),
        "inductor": {key: value for key, value in counters["inductor"].items() if "cache" in key},
    }
    aot = counters["aot_autograd"]
    replays = result.get("cudagraph_replays_by_phase", {})
    result["status"] = (
        "ok"
        if (
            result["finite_losses"]
            and result["finite_grad_count"] == result["grad_tensor_count"] == 193
            and aot["autograd_cache_hit"] + aot["autograd_cache_miss"] > 0
            and not aot["autograd_cache_bypass"]
            and replays.get("forward", 0) > 0
            and replays.get("backward", 0) > 0
        )
        else "error"
    )
    write_json(out, result)
    return int(result["status"] != "ok")


def lockin(out: Path, env: dict[str, str]) -> dict:
    from h200.wave49_h200_lockin.run_wave49_h200 import BENCH_TRUE, MODES  # noqa: PLC0415

    shared = {
        **env,
        **MODES[BENCH_TRUE],
        "LNET_MODEL_W36B": "cuda_graphs,tma_gemm,max_autotune_pointwise",
        "TORCH_LOGS": "aot",
    }
    for key, name in (
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("TRITON_CACHE_DIR", "triton"),
        ("CUDA_CACHE_PATH", "cuda"),
    ):
        cache_dir = Path(env.get(key, str(out / "cache" / name))).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        shared[key] = str(cache_dir)
    shared.pop("TORCHINDUCTOR_FORCE_DISABLE_CACHES", None)
    results = {
        "correctness": stage(
            out,
            "correctness",
            [
                "-m",
                "pytest",
                "-q",
                "tests/test_stem_stage0_cuda_tiled.py",
                "tests/test_model_w29c.py::test_row_pitch",
                "-k",
                "wave50_cuda or row_pitch",
            ],
            75,
            shared,
            json_flag=None,
        ),
    }
    for name, stem, norm, budget in (
        ("reference", "off", "off", 480),
        ("candidate", "cuda_tiled", "fast_fwd", 270),
    ):
        if results["correctness"]["status"] != "ok":
            break
        results[name] = stage(
            out,
            name,
            ["-m", MODULE, "--arm"],
            budget,
            {**shared, "LNET_STEM_STAGE0_MODE": stem, "LNET_NORM_W50_MODE": norm},
        )
    return results


def archive(out: Path, results: dict) -> None:
    files = [
        out / f"{name}.{suffix}"
        for name in results
        for suffix in ("json", "log")
        if (out / f"{name}.{suffix}").is_file() and results[name]["status"] != "pending"
    ]
    if (out / "meta.json").is_file():
        files = [out / "meta.json", *(file for file in files if file.suffix == ".json")]
    manifest = out / "manifest.json"
    write_json(
        manifest,
        {
            "stages": results,
            "json_sha256": {
                file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                for file in files
                if file.suffix == ".json"
            },
        },
    )
    bundle = out / "wave50.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for file in [manifest, *files]:
            tar.add(file, arcname=file.name)
    raw = bundle.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    print(f"VA_T177_SHA256 {hashlib.sha256(raw).hexdigest()}")  # noqa: T201
    for file in [manifest, *files]:
        if file.suffix == ".json":
            print(  # noqa: T201
                f"VA_T177_JSON_SHA256 {file.name} {hashlib.sha256(file.read_bytes()).hexdigest()}"
            )
    print("VA_T177_BEGIN")  # noqa: T201
    for start in range(0, len(encoded), 240):
        print(f"VA_T177_GZ {encoded[start : start + 240]}")  # noqa: T201
    print("VA_T177_END", flush=True)  # noqa: T201


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("wave50_results"))
    parser.add_argument("--run", choices=("screen", "lockin"), default="screen")
    parser.add_argument("--arm", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.arm:
        return compiled_arm(out)

    import torch  # noqa: PLC0415

    if (
        torch.cuda.device_count() != 1
        or "H200" not in torch.cuda.get_device_name(0)
        or torch.cuda.get_device_properties(0).multi_processor_count != 132
    ):
        parser.error("one whole H200 is required")
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **BENCH_ENV}
    started = monotonic()
    print(  # noqa: T201
        sys.version,
        torch.__version__,
        torch.version.cuda,
        torch.backends.cudnn.version(),
        flush=True,
    )
    results = (screen if args.run == "screen" else lockin)(out, env)
    props = torch.cuda.get_device_properties(0)
    write_json(
        out / "meta.json",
        {
            "python": sys.version,
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": props.name,
            "multiprocessors": props.multi_processor_count,
            "total_memory_bytes": props.total_memory,
            "bootstrap_s": 0,
            "screen_wall_s": monotonic() - started,
        },
    )
    archive(out, results)
    print(json.dumps(results, indent=2))  # noqa: T201
    return int(any(row["status"] == "error" for row in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
