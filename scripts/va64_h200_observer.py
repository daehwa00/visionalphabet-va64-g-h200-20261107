"""Owner-side observer for VA64 H200 logs, metrics, and durable backups."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from va64_h200_transport import (
    API,
    JOBS,
    atomic_json,
    download_artifact,
    owner_api_from_secrets,
    validate_metadata,
)

STATE_NAME = "observer-state.json"
EVENTS_NAME = "events.jsonl"
BACKUP_ROOT = "backups"
WANDB_PROJECT_ENV = "VA64_G_WANDB_PROJECT"
_SENSITIVE = re.compile(
    r"(?i)([\"']?(?:authorization|token|access_token|api[_-]?key|password|secret|x-session-id)"
    r"[\"']?\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;<>]+)"
)


def _redact(value: Any, secrets: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        return {str(key): _redact(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    if not isinstance(value, str):
        return value
    text = value
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"(?i)\bBearer\s+[^\s,;<>]+", "Bearer [REDACTED]", text)
    text = _SENSITIVE.sub(r"\1[REDACTED]", text)
    text = re.sub(r"\b[A-Za-z0-9_+/=-]{32,}\b", "[REDACTED]", text)
    text = re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?[REDACTED]", text)
    return text[:200_000]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _append_event(path: Path, event: dict[str, Any], owner_token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = _redact(event, (owner_token,))
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(safe, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _metrics(status: dict[str, Any]) -> dict[str, float | int]:
    progress = status.get("progress") or {}
    checkpoint = status.get("checkpoint") or {}
    train = checkpoint.get("train") or {}
    validation = checkpoint.get("validation") or {}
    steps = [
        value
        for value in (progress.get("global_step"), checkpoint.get("global_step"))
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
    ]
    epochs = [
        value
        for value in (progress.get("epoch"), checkpoint.get("epoch"))
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
    ]
    step = max(steps) if steps else None
    epoch = max(epochs) if epochs else None
    result: dict[str, float | int] = {}
    if isinstance(step, int):
        result["optimizer_updates"] = step
    if isinstance(epoch, int):
        result["epoch"] = epoch
    loss = progress.get("loss")
    if not isinstance(loss, (int, float)):
        loss = train.get("loss")
    if isinstance(loss, (int, float)):
        result["train/loss"] = float(loss)
    for source, target in (
        ("images_per_second", "train/images_per_second"),
        ("loader_wait_seconds", "timing/loader_wait_seconds"),
        ("gpu_step_span_seconds", "timing/gpu_step_span_seconds"),
    ):
        value = train.get(source)
        if isinstance(value, (int, float)):
            result[target] = float(value)
    for source, target in (
        ("accuracy", "validation/top1_percent"),
        ("top5_accuracy", "validation/top5_percent"),
    ):
        value = validation.get(source)
        if isinstance(value, (int, float)):
            result[target] = 100.0 * float(value)
    return result


class _WandbMirror:
    def __init__(self, root: Path, *, enabled: bool = True) -> None:
        self.root = root
        self.enabled = enabled
        self.module = None
        self.runs: dict[str, Any] = {}
        self.campaign_by_job: dict[str, str] = {}
        self.completed_jobs: set[str] = set()
        self.project = os.environ.get(WANDB_PROJECT_ENV, "visionalphabet-rebuttal-va64")
        self.entity = os.environ.get("VA64_G_WANDB_ENTITY", "daehwa")
        if not enabled:
            return
        try:
            import wandb

            self.module = wandb
        except Exception:  # noqa: BLE001 - W&B is an optional best-effort dependency.
            self.enabled = False

    def _run(self, campaign: str, job: str) -> Any | None:
        if not self.enabled or self.module is None:
            return None
        if job in self.runs and self.campaign_by_job.get(job) == campaign:
            return self.runs[job]
        self.finish(job)
        run_id = hashlib.sha256((campaign + ":" + job).encode("utf-8")).hexdigest()[:16]
        try:
            run = self.module.init(
                entity=self.entity,
                project=self.project,
                group=campaign,
                id=run_id,
                resume="allow",
                reinit="create_new",
                mode="online",
                name="VA64-" + job,
                dir=str(self.root / "wandb"),
                config={
                    "campaign_id": campaign,
                    "job": job,
                    "seed": 0,
                    "dataset": "ImageNet1K",
                    "epochs": 0 if job.endswith("preflight") else 100,
                    "synthetic_input": job.endswith("preflight"),
                },
                settings=self.module.Settings(
                    console="off",
                    disable_code=True,
                    disable_git=True,
                    x_disable_stats=True,
                    init_timeout=45,
                ),
            )
            run.define_metric("optimizer_updates")
            run.define_metric("train/*", step_metric="optimizer_updates")
            run.define_metric("validation/*", step_metric="optimizer_updates")
            run.define_metric("timing/*", step_metric="optimizer_updates")
        except Exception:  # noqa: BLE001 - W&B initialization must not block backups.
            self.enabled = False
            return None
        self.runs[job] = run
        self.campaign_by_job[job] = campaign
        return run

    def consume(self, campaign: str, job: str, status: dict[str, Any]) -> None:
        if job in self.completed_jobs:
            return
        if (status.get("current") or {}).get("stage") not in {"training", "completed", "preflight", "bootstrap", "waiting_wandb", "wait_backup"}:
            return
        run = self._run(campaign, job)
        if run is None:
            return
        payload = _metrics(status)
        if payload:
            try:
                run.log(payload)
            except Exception:  # noqa: BLE001 - W&B is an optional best-effort mirror.
                self.enabled = False
                self.finish_all()
                return
        result = status.get("result") or {}
        if result.get("status") == "completed":
            run.summary["completed_epochs"] = result.get("completed_epochs")
            run.summary["global_step"] = result.get("global_step")
            final = (result.get("final_validation") or {}).get("accuracy")
            if isinstance(final, (int, float)):
                run.summary["final/top1_percent"] = 100.0 * float(final)
            # Finish only after the final external checkpoint is confirmed.

    def save_backup(self, campaign: str, job: str, metadata: dict, archive: Path) -> dict:
        run = self._run(campaign, job)
        if run is None:
            raise RuntimeError("W&B is not connected; checkpoint ACK withheld")
        artifact = self.module.Artifact(
            name=job + "-checkpoints", type="model",
            metadata={"sha256": metadata["sha"], "bytes": metadata["bytes"],
                      "epoch": metadata["epoch"], "global_step": metadata["global_step"],
                      "code_sha": metadata["code_sha"], "synthetic_input": job.endswith("preflight")},
        )
        artifact.add_file(str(archive), name="checkpoint-bundle.tar.gz")
        logged = run.log_artifact(artifact, aliases=["latest", "epoch-" + str(metadata["epoch"])])
        logged.wait(timeout=180)
        receipt = {"qualified_name": logged.qualified_name, "digest": logged.digest,
                   "sha256": metadata["sha"], "uploaded": True}
        run.summary["backup/latest_sha256"] = metadata["sha"]
        run.summary["backup/latest_epoch"] = metadata["epoch"]
        return receipt

    def save_logs(self, campaign: str, job: str) -> None:
        run = self._run(campaign, job)
        if run is None:
            raise RuntimeError("W&B log preservation unavailable")
        artifact = self.module.Artifact(name=job + "-logs", type="logs")
        events = self.root / EVENTS_NAME
        if events.is_file(): artifact.add_file(str(events), name="events.jsonl")
        logged = run.log_artifact(artifact, aliases=["latest"])
        logged.wait(timeout=180)

    def finish(self, job: str) -> None:
        run = self.runs.pop(job, None)
        self.campaign_by_job.pop(job, None)
        if run is not None:
            try:
                run.finish()
            except Exception:  # noqa: BLE001 - W&B shutdown failure must not affect backups.
                self.enabled = False

    def finish_all(self) -> None:
        for job in list(self.runs):
            self.finish(job)


def _backup_path(root: Path, job: str, metadata: dict[str, Any]) -> Path:
    directory = root / BACKUP_ROOT / job
    filename = (
        "epoch-"
        + str(metadata["epoch"]).zfill(4)
        + "-step-"
        + str(metadata["global_step"]).zfill(10)
        + "-"
        + metadata["sha"][:12]
        + ".tar.gz"
    )
    return directory / filename


def _persist_receipt(job: str, metadata: dict[str, Any], archive_path: Path) -> None:
    receipt = {
        "observer_managed": True,
        "job": job,
        "sha": metadata["sha"],
        "bytes": metadata["bytes"],
        "epoch": metadata["epoch"],
        "global_step": metadata["global_step"],
        "code_sha": metadata["code_sha"],
        "archive": str(archive_path),
        "verified_at": time.time(),
    }
    atomic_json(archive_path.with_suffix(".receipt.json"), receipt)


def _cleanup_managed_archives(root: Path, job: str, keep_latest: int = 2) -> None:
    """Keep recent verified archives; remove only files with our own receipts."""
    if job not in JOBS or keep_latest < 1:
        raise ValueError("invalid VA64 archive retention policy")
    directory = root / BACKUP_ROOT / job
    if directory.is_symlink() or not directory.is_dir():
        return
    trusted_directory = directory.resolve()
    candidates: list[tuple[tuple[int, int, float, str], Path]] = []
    for receipt_path in directory.glob("*.receipt.json"):
        if receipt_path.is_symlink() or not receipt_path.is_file():
            continue
        receipt = _read_json(receipt_path)
        if receipt.get("observer_managed") is not True or receipt.get("job") != job:
            continue
        archive_path = Path(str(receipt.get("archive", "")))
        if (
            archive_path.parent.resolve() != trusted_directory
            or archive_path.with_suffix(".receipt.json") != receipt_path
            or not archive_path.name.endswith(".tar.gz")
            or archive_path.is_symlink()
            or not archive_path.is_file()
        ):
            continue
        epoch = receipt.get("epoch")
        step = receipt.get("global_step")
        verified_at = receipt.get("verified_at")
        sha = receipt.get("sha")
        if (
            not isinstance(epoch, int)
            or isinstance(epoch, bool)
            or not isinstance(step, int)
            or isinstance(step, bool)
            or not isinstance(verified_at, (int, float))
            or not isinstance(sha, str)
        ):
            continue
        candidates.append(((epoch, step, float(verified_at), sha), archive_path))
    candidates.sort(key=lambda item: item[0], reverse=True)
    retained = {path for _, path in candidates[:keep_latest]}
    retained.update(path for version, path in candidates if version[0] == 100)
    for _, archive_path in candidates:
        if archive_path not in retained:
            archive_path.unlink(missing_ok=True)


def validate_checkpoint_bundle(archive: Path) -> None:
    import io, tarfile, torch
    import run_va64_retrain_imagenet1k as experiment
    with tarfile.open(archive, "r:gz") as stream:
        member = stream.extractfile("checkpoint.pt")
        if member is None: raise ValueError("checkpoint missing")
        payload = torch.load(io.BytesIO(member.read()), map_location="cpu", weights_only=True)
        if not isinstance(payload.get("model"), dict) or not isinstance(payload.get("optimizer"), dict) or not isinstance(payload.get("rng"), dict):
            raise ValueError("checkpoint model/optimizer missing")
        model = experiment.build_model("G", prepared=False)
        model.load_state_dict(payload["model"], strict=True)
        if payload.get("variant") == "G":
            if payload.get("definition") != experiment.definition_digest("G"): raise ValueError("G definition mismatch")
        else:
            contracts = [m for m in stream.getmembers() if m.name.startswith("contracts/") and m.name.endswith(".json")]
            matches = []
            for record in contracts:
                value = json.load(stream.extractfile(record))
                if value.get("schema") == experiment.runtime.SCHEMA and value.get("model", {}).get("key") == "va_k64_retrain_G":
                    matches.append(value)
            if len(matches) != 1: raise ValueError("G checkpoint contract missing or ambiguous")
            experiment.runtime._validate_binding(payload, matches[0])


def _backup_latest(
    api: API,
    root: Path,
    job: str,
    expected_code_sha: str | None,
    previous_sha: str | None,
    mirror: _WandbMirror,
    campaign: str,
) -> str | None:
    metadata = api.call("/artifact/" + job + "/latest")
    if metadata is None:
        return None
    metadata = validate_metadata(metadata)
    if expected_code_sha and metadata.get("code_sha") != expected_code_sha:
        raise ValueError("checkpoint code revision does not match the active session")
    target = _backup_path(root, job, metadata)
    receipt = target.with_suffix(".receipt.json")
    if previous_sha == metadata["sha"] and target.is_file() and receipt.is_file():
        saved = _read_json(receipt)
        if saved.get("wandb", {}).get("uploaded"):
            api.call("/artifact/" + job + "/" + metadata["sha"] + "/ack", {})
            _cleanup_managed_archives(root, job)
            return metadata["sha"]
    download_artifact(api, job, metadata, target)
    validate_checkpoint_bundle(target)
    wandb_receipt = mirror.save_backup(campaign, job, metadata, target)
    _persist_receipt(job, metadata, target)
    record = _read_json(receipt); record["wandb"] = wandb_receipt; atomic_json(receipt, record)
    api.call("/artifact/" + job + "/" + metadata["sha"] + "/ack", {})
    _cleanup_managed_archives(root, job)
    return metadata["sha"]


def _is_json_object(value: Any) -> bool:
    return isinstance(value, dict)


def observe(
    root: str | Path,
    secrets_path: str | Path,
    *,
    poll_seconds: float = 30.0,
    once: bool = False,
    wandb_enabled: bool = True,
) -> None:
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    output = Path(root)
    output.mkdir(parents=True, exist_ok=True)
    api = owner_api_from_secrets(secrets_path)
    state_path = output / STATE_NAME
    state = _read_json(state_path)
    cursor = int(state.get("cursor", 0))
    latest_sha = dict(state.get("latest_sha", {}))
    mirror = _WandbMirror(output, enabled=wandb_enabled)
    if not mirror.enabled: raise RuntimeError("online W&B is required for the G observer")
    active_jobs = set(state.get("active_jobs", []))
    campaigns = dict(state.get("campaigns", {}))
    child_failed = bool(state.get("child_failed", False))
    last_heartbeat = 0.0
    pending_warned: set[str] = set()

    while True:
        view = api.call("/snapshot?after=" + str(cursor))
        if not _is_json_object(view):
            raise ValueError("relay returned an invalid snapshot")
        session = view.get("session") or {}
        code_sha = session.get("code")
        for event in view.get("events", []):
            if not isinstance(event, dict):
                continue
            try:
                value = event.get("value")
                payload = json.loads(value) if isinstance(value, str) else value
                if not isinstance(payload, dict):
                    continue
                status = payload.get("status") or {}
                current = status.get("current") or {}
                job = current.get("job")
                campaign = current.get("campaign_id")
                if job in JOBS and isinstance(campaign, str) and isinstance(status, dict):
                    active_jobs.add(job); campaigns[job] = campaign
                    child_failed = bool(status.get("ended") and status.get("exit_code") not in (None, 0))
                    if mirror._run(campaign, job) is None: raise RuntimeError("W&B initialization failed")
                    if current.get("stage") == "waiting_wandb":
                        api.call("/command", {"action": "ready", "job": job})
                    mirror.consume(campaign, job, status)
                event_id = event.get("id")
                if isinstance(event_id, int):
                    cursor = max(cursor, event_id)
                _append_event(output / EVENTS_NAME, payload, api.token)
            except (ValueError, TypeError):
                continue

        if time.monotonic() - last_heartbeat >= 60:
            api.call("/command", {"action": "heartbeat"})
            last_heartbeat = time.monotonic()

        backups_ready = True
        for job in active_jobs:
            try:
                observed_sha = _backup_latest(
                    api,
                    output,
                    job,
                    code_sha if isinstance(code_sha, str) else None,
                    latest_sha.get(job), mirror, campaigns[job],
                )
                if observed_sha is not None:
                    latest_sha[job] = observed_sha
                elif session.get("ended") and not child_failed:
                    backups_ready = False
                    if job not in pending_warned:
                        print("VA64_OBSERVER_BACKUP_PENDING job=" + job, flush=True)
                        pending_warned.add(job)
            except Exception as error:  # noqa: BLE001 - retry transient relay failures next poll.
                backups_ready = False
                print(
                    "VA64_OBSERVER_BACKUP_RETRY job=" + job + " error=" + type(error).__name__,
                    flush=True,
                )

        atomic_json(
            state_path,
            {
                "child_failed": child_failed,
                "active_jobs": sorted(active_jobs),
                "campaigns": campaigns,
                "cursor": cursor,
                "latest_sha": latest_sha,
                "session_ended": bool(session.get("ended")),
                "heartbeat": time.time(),
            },
        )
        ended = bool(session.get("ended"))
        if ended and backups_ready:
            for job in active_jobs: mirror.save_logs(campaigns[job], job)
            mirror.finish_all()
        if once or (ended and backups_ready):
            return
        time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--secrets", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    observe(
        args.root,
        args.secrets,
        poll_seconds=args.poll_seconds,
        once=args.once,
        wandb_enabled=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
