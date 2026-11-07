from __future__ import annotations

from scripts import remote_run_control as control


def test_stop_requested_error_preserves_the_control_record() -> None:
    record = {
        "schema": control.SCHEMA,
        "campaign_id": "campaign",
        "generation": 2,
        "action": "stop",
    }

    error = control.StopRequestedError(record)

    assert str(error) == "repository owner requested a graceful stop"
    assert error.record is record


def test_remote_control_exports_only_the_used_contract() -> None:
    assert control.__all__ == ["SCHEMA", "StopRequestedError"]
