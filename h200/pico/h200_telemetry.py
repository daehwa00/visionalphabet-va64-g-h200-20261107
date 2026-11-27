"""Best-effort epoch telemetry and remote stop checks."""

import json
import os
import time
from datetime import datetime, timezone
from urllib.request import Request, urlopen


class Telemetry:
    def __init__(self, run_id, total_epochs, gpu_name):
        self.url = os.environ.get("H200_TELEMETRY_URL", "")
        self.kill_url = os.environ.get("H200_KILL_SWITCH_URL", "")
        self.run_id = os.environ.get("H200_RUN_ID") or run_id
        self.total_epochs = total_epochs
        self.gpu_name = gpu_name

    def report(self, event, epoch, global_step, row=None):
        if not self.url:
            return False
        try:
            row = row or {}
            payload = {
                "run_id": self.run_id,
                "event": event,
                "epoch": epoch,
                "total_epochs": self.total_epochs,
                "global_step": global_step,
                "train_loss": row.get("train_loss"),
                "train_acc": row.get("train_mixed_accuracy"),
                "val_raw_top1": row.get("raw_validation_accuracy"),
                "val_ema_top1": row.get("validation_accuracy"),
                "val_raw_ce": row.get("raw_validation_cross_entropy"),
                "val_ema_ce": row.get("validation_cross_entropy"),
                "lr": row.get("learning_rate"),
                "epoch_sec": row.get("training_seconds"),
                "grad_finite": row.get("finite_gradients"),
                "gpu_name": self.gpu_name,
                "utc_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            request = Request(
                self.url,
                data=json.dumps(payload, separators=(",", ":"), allow_nan=False).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=5) as response:
                reply = json.loads(response.read(8193))
            return isinstance(reply, dict) and reply.get("stop") is True
        except Exception as exc:
            print(f"H200 telemetry {event} unavailable ({type(exc).__name__})", flush=True)
            return False

    def should_stop(self, epoch, global_step, row):
        if self.report("epoch", epoch, global_step, row):
            return True
        if not self.kill_url:
            return False
        try:
            separator = "&" if "?" in self.kill_url else "?"
            request = Request(
                f"{self.kill_url}{separator}t={int(time.time())}",
                headers={"Cache-Control": "no-cache"},
            )
            with urlopen(request, timeout=5) as response:
                body = response.read(8193)
            if len(body) > 8192:
                return False
            return any(line.strip() == "STOP" for line in body.decode("utf-8").splitlines())
        except Exception as exc:
            print(f"H200 backup stop check unavailable ({type(exc).__name__})", flush=True)
            return False
