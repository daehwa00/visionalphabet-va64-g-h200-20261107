"""Results-only telemetry must never request an external model checkpoint."""
import json
from types import SimpleNamespace

import pytest

from scripts import va64_h200_observer as observer

JOB = "va64-g-seed0"
CAMPAIGN = "results-only"


def snapshot(*, ended=False, event_id=1):
    status = {
        "current": {"job": JOB, "campaign_id": CAMPAIGN, "stage": "completed" if ended else "training",
                    "checkpoint_policy": "local_only"},
        "checkpoint": {"epoch": 14, "global_step": 70070, "validation": {"accuracy": .42, "loss": 2.3}},
        "result": {"status": "completed", "completed_epochs": 100, "final_validation": {"accuracy": .71}} if ended else {},
        "summary": {"cost": {"train_seconds": 1000}, "status": "completed"} if ended else {},
        "ended": ended, "exit_code": 0 if ended else None,
    }
    return {"session": {"id": "session", "code": "b" * 40, "ended": ended},
            "events": [{"id": event_id, "value": {"status": status}}]}


class Relay:
    token = "private-owner-value"

    def __init__(self, views, heartbeat_fails=False):
        self.views = iter(views)
        self.commands = []
        self.heartbeat_fails = heartbeat_fails

    def call(self, path, value=None):
        if path.startswith("/snapshot"):
            view = next(self.views)
            if isinstance(view, Exception):
                raise view
            return view
        assert path == "/command", "no artifact endpoint should ever be used"
        self.commands.append(value)
        if self.heartbeat_fails:
            raise ConnectionError("relay unavailable")
        return {}


class Mirror:
    enabled = True

    def __init__(self, *args, **kwargs):
        self.consumed = []
        self.saved = []

    def _run(self, campaign, job):
        return True

    def consume(self, campaign, job, status):
        self.consumed.append(status)

    def save_logs(self, campaign, job):
        self.saved.append(job)

    def finish_all(self):
        pass


def install(monkeypatch, relay, mirror):
    monkeypatch.setattr(observer, "owner_api_from_secrets", lambda _: relay)
    monkeypatch.setattr(observer, "_WandbMirror", lambda *a, **k: mirror)
    def forbidden(*args, **kwargs):
        raise AssertionError("results-only observer attempted a model backup")
    monkeypatch.setattr(observer, "_backup_latest", forbidden)


def test_results_only_skips_checkpoint_and_persists_final_result(tmp_path, monkeypatch):
    relay = Relay([snapshot(ended=True)])
    mirror = Mirror()
    install(monkeypatch, relay, mirror)
    observer.observe(tmp_path, "unused", once=True)
    directory = observer._result_directory(tmp_path, CAMPAIGN, JOB)
    assert json.loads((directory / "result.json").read_text())["completed_epochs"] == 100
    assert json.loads((directory / "final-summary.json").read_text())["cost"]["train_seconds"] == 1000
    assert json.loads((directory / "epoch-14.json").read_text())["validation"]["accuracy"] == .42
    assert len((directory / "epochs.jsonl").read_text().splitlines()) == 1
    assert (tmp_path / observer.EVENTS_NAME).is_file()
    state = json.loads((tmp_path / observer.STATE_NAME).read_text())
    assert state["completed_session_id"] == "session"
    assert mirror.saved == [JOB]
    assert not any(command.get("action") == "ready" for command in relay.commands)


def test_wandb_failure_cannot_prevent_raw_results_or_kill_observer(tmp_path, monkeypatch):
    class FailingMirror(Mirror):
        def _run(self, *args):
            assert (tmp_path / observer.EVENTS_NAME).is_file(), "raw data must be saved first"
            raise ConnectionError("W&B unavailable")
    install(monkeypatch, Relay([snapshot(ended=True)], heartbeat_fails=True), FailingMirror())
    observer.observe(tmp_path, "unused", once=True)
    state = json.loads((tmp_path / observer.STATE_NAME).read_text())
    assert state["pending_status"][JOB]["result"]["completed_epochs"] == 100
    assert not state["completed_session_id"]
    assert (observer._result_directory(tmp_path, CAMPAIGN, JOB) / "result.json").is_file()


def test_transient_snapshot_and_logs_failures_are_retried(tmp_path, monkeypatch):
    first = snapshot(ended=True)
    next_view = {**first, "events": []}
    relay = Relay([ConnectionError("unavailable"), first, next_view])
    class RetryingMirror(Mirror):
        attempts = 0
        def save_logs(self, campaign, job):
            self.attempts += 1
            if self.attempts == 1:
                raise ConnectionError("upload unavailable")
            super().save_logs(campaign, job)
    mirror = RetryingMirror()
    install(monkeypatch, relay, mirror)
    class Done(Exception):
        pass
    def sleep(_):
        state = observer._read_json(tmp_path / observer.STATE_NAME)
        if state.get("completed_session_id") == "session":
            raise Done
    monkeypatch.setattr(observer.time, "sleep", sleep)
    with pytest.raises(Done):
        observer.observe(tmp_path, "unused", poll_seconds=.01)
    assert mirror.attempts == 2
    assert mirror.saved == [JOB]


def test_logs_artifact_includes_small_results_and_never_model(tmp_path):
    status = snapshot(ended=True)["events"][0]["value"]["status"]
    observer._persist_results(tmp_path, CAMPAIGN, JOB, status)
    directory = observer._result_directory(tmp_path, CAMPAIGN, JOB)
    (directory / "checkpoint.pt").write_bytes(b"must not upload")
    files = []
    class Artifact:
        def __init__(self, **kwargs):
            assert kwargs["type"] == "logs"
        def add_file(self, path, name):
            files.append(name)
    class Run:
        def log_artifact(self, artifact, aliases):
            return SimpleNamespace(wait=lambda **kwargs: None)
    mirror = observer._WandbMirror(tmp_path, enabled=False)
    mirror.module = SimpleNamespace(Artifact=Artifact)
    mirror._run = lambda campaign, job: Run()
    mirror.save_logs(CAMPAIGN, JOB)
    assert {"results/result.json", "results/final-summary.json", "results/epochs.jsonl"} <= set(files)
    assert all(not name.endswith(".pt") for name in files)


def test_wandb_init_failure_remains_retryable(tmp_path):
    mirror = observer._WandbMirror(tmp_path, enabled=False)
    mirror.enabled = True
    attempts = []
    def init(**kwargs):
        attempts.append(kwargs)
        raise ConnectionError("temporary")
    mirror.module = SimpleNamespace(init=init, Settings=lambda **kwargs: None)
    assert mirror._run(CAMPAIGN, JOB) is None
    assert mirror.enabled
    assert mirror._run(CAMPAIGN, JOB) is None
    assert len(attempts) == 2


def test_wandb_log_failure_is_retryable_and_does_not_disable_mirror(tmp_path):
    mirror = observer._WandbMirror(tmp_path, enabled=False)
    mirror.enabled = True
    class Run:
        def log(self, payload):
            raise ConnectionError("temporary W&B log error")
        def finish(self):
            pass
    mirror.runs[JOB] = Run()
    mirror.campaign_by_job[JOB] = CAMPAIGN
    mirror.module = SimpleNamespace()
    status = snapshot()["events"][0]["value"]["status"]
    with pytest.raises(ConnectionError):
        mirror.consume(CAMPAIGN, JOB, status)
    assert mirror.enabled
    assert JOB not in mirror.runs


def test_epoch_rows_are_deduplicated_and_legacy_checkpoint_policy_still_backs_up(tmp_path, monkeypatch):
    view = snapshot()
    status = view["events"][0]["value"]["status"]
    status["current"].pop("checkpoint_policy")
    observer._persist_results(tmp_path, CAMPAIGN, JOB, status)
    observer._persist_results(tmp_path, CAMPAIGN, JOB, status)
    rows = observer._result_directory(tmp_path, CAMPAIGN, JOB) / "epochs.jsonl"
    assert len(rows.read_text().splitlines()) == 1
    mirror = Mirror()
    install(monkeypatch, Relay([view]), mirror)
    calls = []
    monkeypatch.setattr(observer, "_backup_latest", lambda *args: calls.append(args[2]))
    observer.observe(tmp_path, "unused", once=True)
    assert calls == [JOB]


def test_ended_session_drains_all_pages_before_finalizing_results(tmp_path, monkeypatch):
    first = snapshot()
    first["session"]["ended"] = True
    first["events"] = [{"id": index, "value": first["events"][0]["value"]} for index in range(1, 101)]
    final = snapshot(ended=True, event_id=101)
    relay = Relay([first, final])
    class CheckingMirror(Mirror):
        def save_logs(self, campaign, job):
            state = observer._read_json(tmp_path / observer.STATE_NAME)
            assert state["cursor"] == 101
            assert state["terminal_event_seen"]
            directory = observer._result_directory(tmp_path, campaign, job)
            assert json.loads((directory / "result.json").read_text())["completed_epochs"] == 100
            super().save_logs(campaign, job)
    mirror = CheckingMirror()
    install(monkeypatch, relay, mirror)
    observer.observe(tmp_path, "unused", once=True)
    state = observer._read_json(tmp_path / observer.STATE_NAME)
    assert state["completed_session_id"] == "session"
    assert len((tmp_path / observer.EVENTS_NAME).read_text().splitlines()) == 101
    assert mirror.saved == [JOB]


def test_new_session_resets_terminal_event_seen(tmp_path, monkeypatch):
    observer.atomic_json(tmp_path / observer.STATE_NAME, {
        "session_id": "old", "terminal_event_seen": True, "completed_session_id": "old", "cursor": 100,
    })
    relay = Relay([snapshot(event_id=101)])
    install(monkeypatch, relay, Mirror())
    observer.observe(tmp_path, "unused", once=True)
    state = observer._read_json(tmp_path / observer.STATE_NAME)
    assert state["session_id"] == "session"
    assert state["terminal_event_seen"] is False


def test_validation_loss_uses_runtime_cross_entropy():
    metrics = observer._metrics({"checkpoint": {"validation": {"cross_entropy": 2.3}}})
    assert metrics["validation/loss"] == 2.3
