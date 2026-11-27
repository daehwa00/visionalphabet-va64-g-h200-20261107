"""Tests for the H200 reporter and loopback receiver."""

import importlib.util
import io
import json
import sys
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reporter = load_module("h200_telemetry", "h200/pico/h200_telemetry.py")
receiver = load_module("h200_telemetry_server", "scripts/h200_telemetry_server.py")


@pytest.fixture
def endpoint(tmp_path):
    token = "x" * 32
    with receiver.create_server(tmp_path, token, port=0) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            yield base, f"{base}/report/{token}"
        finally:
            server.shutdown()
            thread.join()


def post(url, data):
    with urlopen(Request(url, data=data), timeout=2) as response:
        return json.load(response)


def test_receiver(endpoint, tmp_path):
    base, url = endpoint
    with urlopen(f"{base}/health", timeout=2) as response:
        assert json.load(response) == {"ok": True}
    report = b'{"run_id":"pico.1-test_0","event":"epoch"}'
    assert post(url, report) == {"ok": True, "stop": False}
    for flag in ("STOP_pico.1-test_0", "STOP_H200"):
        path = tmp_path / flag
        path.touch()
        assert post(url, report) == {"ok": True, "stop": True}
        path.unlink()
    lines = (tmp_path / "pico.1-test_0.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == [json.loads(report)] * 3


@pytest.mark.parametrize(
    "body,status",
    [
        (b"[]", 400),
        (b'{"run_id":"../out"}', 400),
        (b'{"run_id":"test","loss":NaN}', 400),
        (b'{"run_id":"test","loss":1e999}', 400),
        (b"x" * 8193, 413),
    ],
)
def test_receiver_rejects_invalid_body(endpoint, tmp_path, body, status):
    with pytest.raises(HTTPError) as error:
        post(endpoint[1], body)
    assert error.value.code == status
    assert list(tmp_path.iterdir()) == []


def test_receiver_rejects_wrong_token(endpoint, tmp_path):
    with pytest.raises(HTTPError) as error:
        post(endpoint[0] + "/report/wrong", b'{"run_id":"test"}')
    assert error.value.code == 404
    assert list(tmp_path.iterdir()) == []


def test_payload_and_primary_stop(endpoint, tmp_path, monkeypatch):
    monkeypatch.setenv("H200_TELEMETRY_URL", endpoint[1])
    monkeypatch.delenv("H200_KILL_SWITCH_URL", raising=False)
    telemetry = reporter.Telemetry("pico", 600, "H200")
    row = {
        "train_loss": 1.25,
        "train_mixed_accuracy": 0.5,
        "raw_validation_accuracy": 0.6,
        "validation_accuracy": 0.7,
        "raw_validation_cross_entropy": 1.1,
        "validation_cross_entropy": 0.9,
        "learning_rate": 0.0025,
        "training_seconds": 77.0,
        "finite_gradients": True,
    }
    assert not telemetry.should_stop(1, 1251, row)
    (tmp_path / "STOP_pico").touch()
    assert telemetry.should_stop(2, 2502, row)
    report = json.loads((tmp_path / "pico.jsonl").read_text().splitlines()[0])
    assert report.pop("utc_time").endswith("+00:00")
    assert report == dict(
        run_id="pico", event="epoch", epoch=1, total_epochs=600, global_step=1251,
        train_loss=1.25, train_acc=0.5, val_raw_top1=0.6, val_ema_top1=0.7,
        val_raw_ce=1.1, val_ema_ce=0.9, lr=0.0025, epoch_sec=77.0,
        grad_finite=True, gpu_name="H200",
    )


@pytest.mark.parametrize("backup,stop", [(b"# STOP\nrunning\n", False), (b"# STOP\n STOP\n", True)])
def test_backup_after_post_failure(monkeypatch, backup, stop):
    monkeypatch.setenv("H200_TELEMETRY_URL", "https://example.com/report/test")
    monkeypatch.setenv("H200_KILL_SWITCH_URL", "https://example.com/STOP")
    monkeypatch.setattr(reporter.time, "time", lambda: 123)

    def request(req, timeout):
        assert timeout == 5
        if req.get_method() == "POST":
            raise TimeoutError
        assert req.full_url == "https://example.com/STOP?t=123"
        return io.BytesIO(backup)

    monkeypatch.setattr(reporter, "urlopen", request)
    assert reporter.Telemetry("pico", 600, "H200").should_stop(1, 10, {}) is stop


def test_all_network_errors_are_nonfatal(monkeypatch):
    monkeypatch.setenv("H200_TELEMETRY_URL", "bad-url")
    monkeypatch.setenv("H200_KILL_SWITCH_URL", "bad-url")
    assert not reporter.Telemetry("pico", 600, "H200").should_stop(1, 10, {})


@pytest.mark.parametrize("source", ["primary", "backup"])
def test_remote_stop_forces_checkpoint_before_return(endpoint, tmp_path, monkeypatch, source):
    torch = pytest.importorskip("torch")
    for location in ("src", "scripts", "h200/pico"):
        monkeypatch.syspath_prepend(str(ROOT / location))
    monkeypatch.setitem(sys.modules, "h200_telemetry", reporter)
    import va_imagenet100_runtime as runtime

    monkeypatch.setenv("H200_TELEMETRY_URL", endpoint[1])
    monkeypatch.delenv("H200_KILL_SWITCH_URL", raising=False)
    monkeypatch.delenv("H200_RUN_ID", raising=False)
    loader = torch.utils.data.DataLoader([0])
    monkeypatch.setattr(runtime.data_runtime, "loaders", lambda *args, **kwargs: (loader, loader))
    monkeypatch.setattr(runtime, "_initialize_wandb_run", lambda *args, **kwargs: None)
    checkpoint = tmp_path / "checkpoints" / "pico__seed501.pt"
    history = tmp_path / "history.jsonl"
    events = []
    original_report = reporter.Telemetry.report

    def report(self, event, *args):
        events.append(event)
        if event in {"epoch", "stopped_by_kill_switch"}:
            assert json.loads(history.read_text())["epoch"] == 1
        if event == "stopped_by_kill_switch":
            state = torch.load(checkpoint, weights_only=False)
            assert state["epoch"] == state["global_step"] == 1
            assert state["history"][0]["epoch"] == 1
        return original_report(self, event, *args)

    monkeypatch.setattr(reporter.Telemetry, "report", report)
    if source == "primary":
        (tmp_path / "STOP_H200").touch()
    else:
        backup_url = "https://example.com/STOP"
        monkeypatch.setenv("H200_KILL_SWITCH_URL", backup_url)

        def request(req, timeout):
            if req.full_url.startswith(backup_url + "?t="):
                return io.BytesIO(b"STOP\n")
            return urlopen(req, timeout=timeout)

        monkeypatch.setattr(reporter, "urlopen", request)

    def train_epoch(model, runtime_model, loader, optimizer, **kwargs):
        optimizer.zero_grad()
        model(torch.ones(1, 1)).sum().backward()
        optimizer.step()
        return {"loss": 1.0, "mixed_accuracy": 0.5}

    recipe = dict(
        epochs=3, batch_size=2, validation_loader_persistent_workers=False,
        loader_prefetch_factor=2, fast_gpu_augment=False, loader_persistent_workers=False,
        progress_every=1, mixup_alpha=0.2, label_smoothing=0.05, precision="float32",
        eval_every=2, eval_dense_from=3, dual_eval=False, checkpoint_every=3,
    )
    bindings = runtime.RunnerBindings(
        variants=("pico",), seeds=(501,), model_config=lambda **kwargs: None,
        build_model=lambda *args: torch.nn.Linear(1, 1),
        contract=lambda _: {}, prepare_model=lambda model, recipe: model,
        build_optimizer=lambda model, recipe: torch.optim.AdamW(model.parameters()),
        train_epoch=train_epoch,
        evaluate=lambda *args, **kwargs: {"accuracy": 0.5, "cross_entropy": 1.0},
        wandb_model_metrics=lambda _: {}, summarize=lambda *args: None,
    )
    assert runtime._run_job(
        tmp_path, {"recipe": recipe, "model": {}}, variant="pico", seed=501,
        data_root=tmp_path, workers=0, device=torch.device("cpu"), bindings=bindings,
    ) is True
    assert events == ["started", "epoch", "stopped_by_kill_switch"]
