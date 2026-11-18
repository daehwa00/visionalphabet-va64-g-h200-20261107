#!/usr/bin/env python3
"""Fail-closed validator for the K128 D2262 no-training checkpoint study.

The validator deliberately uses only the Python standard library so that it can
run in a clean environment after the experiments have finished.  It treats the
manifest as untrusted input: registered files are re-hashed, JSON is parsed in
strict mode, and every study-local path is checked for escapes and symlinks.
"""

from __future__ import annotations

# pyright: reportArgumentType=false, reportExplicitAny=false
# pyright: reportImplicitStringConcatenation=false, reportOperatorIssue=false
# pyright: reportOptionalOperand=false
import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA = "lnet.a2d.r2k3.k128_d2262_no_training_study.v1"
VARIANT = "K128-P160-160-160-128-D2262"
MANIFEST_NAME = "study.json"

FAMILY_IDS = (
    "F01_pole_atlas",
    "F02_synthetic_identification",
    "F03_recurrence_causality",
    "F04_pole_targeted",
    "F05_d4_paths",
    "F06_transition_decomposition",
    "F07_rank_capacity",
    "F08_terminal_q",
    "F09_representation_geometry",
    "F10_erf_attribution",
    "F11_robustness_invariance",
    "F12_calibration_error_statistics",
)
FIGURE_IDS = (
    "FIG01_alpha_half_life",
    "FIG02_omega_preferred_frequency",
    "FIG03_path_orientation",
    "FIG04_intervention_forest",
    "FIG05_full_memory_carry_erf",
    "FIG06_depth_rank_cka_knn",
)
POLE_MODULE_IDS = (
    "stage1",
    "stage2",
    "stage3",
    "terminal",
    "same_resolution.56",
    "same_resolution.28",
    "same_resolution.14",
    "same_resolution.7",
    "extra14.1",
    "extra14.2",
    "extra14.3",
    "extra14.4",
)
POSTFUSION_LABELS = (
    "main.stage1",
    "main.stage2",
    "main.stage3",
    "same_resolution.56",
    "same_resolution.28",
    "same_resolution.14",
    "same_resolution.7",
    "extra14.1",
    "extra14.2",
    "extra14.3",
    "extra14.4",
)
F04_TARGETS = tuple(
    f"{kind}_{band}_off"
    for kind in ("memory", "frequency")
    for band in ("low", "mid", "high")
)
F11_BASE_CONDITIONS = (
    *(f"noise_{value}" for value in (0.05, 0.1, 0.2)),
    *(f"blur_{value}" for value in (1.0, 2.0, 4.0)),
    *(f"contrast_{value}" for value in (0.75, 0.5, 0.25)),
    *(f"occlusion_{value}" for value in (0.1, 0.2, 0.3, 0.4)),
    *(f"resolution_{value}" for value in (192, 256, 288)),
    "lowpass_2",
    "lowpass_4",
    "highpass_2",
    "highpass_4",
    "bandpass",
    "rotate90",
    "hflip",
    "vflip",
    "translate_16",
    "crop_scale_0.75",
    "patch_shuffle_4x4",
    "grayscale",
)
F11_CROSS_BASES = (
    "noise_0.1",
    "blur_2.0",
    "occlusion_0.2",
    "lowpass_4",
    "highpass_4",
    "rotate90",
    "grayscale",
)
EXPECTED_CONDITION_IDS: dict[str, tuple[str, ...]] = {
    "F01_pole_atlas": POLE_MODULE_IDS,
    "F02_synthetic_identification": POLE_MODULE_IDS,
    "F03_recurrence_causality": (
        "recurrence_off",
        *(f"effective_half_life_cap_{value}" for value in (2, 4, 8, 16)),
        *(f"periodic_reset_{value}" for value in (1, 2, 4, 8, 16)),
        "distant_region_reset",
    ),
    "F04_pole_targeted": (
        *(
            condition_id
            for target in F04_TARGETS
            for condition_id in (target, *(f"{target}_random_{seed:02d}" for seed in range(20)))
        ),
        "omega_zero",
        "omega_sign_flip",
        "omega_shuffle",
        "xy_swap",
        "alpha_shuffle",
        "terminal_all_poles_off",
        "extra14_long_memory_off",
    ),
    "F05_d4_paths": (
        *(f"path_{value}_off" for value in range(4)),
        "paths_0_1_off",
        "paths_2_3_off",
        "paths_0_2_off",
        "paths_1_3_off",
        "all_paths_off",
        "path_reverse",
        "path_rotate",
        "path_swap_x",
        "path_swap_y",
        "rotation_equivariance",
    ),
    "F06_transition_decomposition": (
        "memory_zero",
        "carry_zero",
        "memory_and_carry_zero",
        *(
            f"postfusion_scale_{str(value).replace('.', '_')}"
            for value in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
        ),
        *(f"postfusion_off_{label.replace('.', '_')}" for label in POSTFUSION_LABELS),
        "same_class_branch_swap",
        "different_class_branch_swap",
    ),
    "F07_rank_capacity": (
        "weight_space",
        "reader_component2_main_off",
        "reader_component2_extra14_off",
        "reader_component2_all_off",
        "postfusion_all_activation_rank_75",
        "postfusion_all_activation_rank_50",
        "postfusion_all_activation_rank_25",
        "postfusion_main_activation_rank_50",
        "postfusion_base_activation_rank_50",
        "postfusion_extra14_activation_rank_50",
        "postfusion_main_stage1_activation_rank_50",
        "postfusion_main_stage2_activation_rank_50",
        "postfusion_main_stage3_activation_rank_50",
    ),
    "F08_terminal_q": ("terminal_q_contribution",),
    "F09_representation_geometry": (
        "r56/base/E_out",
        "r28/base/E_out",
        "r14/base/E_out",
        "r7/base/E_out",
        *(f"r14/extra{value}/E_out" for value in range(1, 5)),
        "main/stage1/E_out",
        "main/stage2/E_out",
        "main/stage3/E_out",
    ),
    "F10_erf_attribution": (
        "gradient_full",
        "gradient_memory_zero",
        "gradient_carry_zero",
        "gradient_long_memory_off",
        "gradient_path0_off",
        "integrated_gradients_full",
        "occlusion_full",
    ),
    "F11_robustness_invariance": (
        *F11_BASE_CONDITIONS,
        *(f"half_life_cap_2_x_{condition_id}" for condition_id in F11_CROSS_BASES),
    ),
    "F12_calibration_error_statistics": (
        "baseline_calibration",
        "robustness_calibration",
        "robustness_error_transitions",
    ),
}
ALLOWED_UNSUPPORTED: dict[str, frozenset[str]] = {
    "F03_recurrence_causality": frozenset(
        (*((f"periodic_reset_{value}") for value in (1, 2, 4, 8, 16)), "distant_region_reset")
    ),
    "F05_d4_paths": frozenset(("rotation_equivariance",)),
    "F06_transition_decomposition": frozenset(
        ("same_class_branch_swap", "different_class_branch_swap")
    ),
}

_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
_HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
_HISTORICAL_WORDS = ("역사", "기록", "최고", "best", "historical", "epoch 99", "에폭 99")
_EPOCH100_WORDS = ("epoch 100", "에폭 100", "100 epoch", "100에폭", "체크포인트")


class StrictJSONError(ValueError):
    """Raised when JSON uses duplicate keys or non-standard numeric values."""


def _reject_constant(value: str) -> None:
    raise StrictJSONError(f"non-finite JSON number {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJSONError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    _assert_finite(value)
    return value


def _assert_finite(value: Any, pointer: str = "$") -> None:
    """Reject non-finite floats recursively, including programmatic inline data."""

    if isinstance(value, float) and not math.isfinite(value):
        raise StrictJSONError(f"non-finite number at {pointer}")
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, f"{pointer}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, f"{pointer}[{index}]")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class StudyValidator:
    def __init__(self, root: Path, expected_checkpoint_sha256: str) -> None:
        self.root_argument = root
        self.root = root.resolve()
        self.expected_checkpoint_sha256 = expected_checkpoint_sha256.lower()
        self.errors: list[str] = []
        self.checked_artifacts = 0
        self.checked_families = 0
        self.checked_figures = 0
        self._strict_json_paths: set[Path] = set()

    def error(self, message: str) -> None:
        self.errors.append(message)

    def require_mapping(self, value: Any, where: str) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            self.error(f"{where} must be an object")
            return None
        return value

    def local_file(self, raw_path: Any, where: str) -> Path | None:  # noqa: PLR0911
        """Resolve a study-local regular file without following symlink entries."""

        if not isinstance(raw_path, str) or not raw_path:
            self.error(f"{where} must be a non-empty relative path")
            return None
        try:
            relative = Path(raw_path)
        except (OSError, ValueError) as exc:
            self.error(f"{where} is not a valid path: {raw_path!r} ({exc})")
            return None
        if relative.is_absolute() or ".." in relative.parts:
            self.error(f"{where} must remain under the study root: {raw_path!r}")
            return None

        candidate = self.root / relative
        current = self.root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                self.error(f"{where} contains a symlink: {raw_path!r}")
                return None

        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError, ValueError) as exc:
            self.error(f"{where} does not exist: {raw_path!r} ({exc})")
            return None
        try:
            resolved.relative_to(self.root)
        except ValueError:
            self.error(f"{where} escapes the study root: {raw_path!r}")
            return None
        if not resolved.is_file():
            self.error(f"{where} is not a regular file: {raw_path!r}")
            return None
        return resolved

    def external_checkpoint_file(self, raw_path: Any) -> Path | None:
        if not isinstance(raw_path, str) or not raw_path:
            self.error("provenance.checkpoint.path must be a non-empty path")
            return None
        try:
            candidate = Path(raw_path)
        except (OSError, ValueError) as exc:
            self.error(f"provenance.checkpoint.path is not valid: {raw_path!r} ({exc})")
            return None
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError, ValueError) as exc:
            self.error(f"provenance.checkpoint.path does not exist: {raw_path!r} ({exc})")
            return None
        if not resolved.is_file():
            self.error(f"provenance.checkpoint.path is not a regular file: {raw_path!r}")
            return None
        return resolved

    def strict_json_file(self, path: Path, where: str) -> Any | None:
        try:
            value = _load_json(path)
        except (OSError, UnicodeError, json.JSONDecodeError, StrictJSONError) as exc:
            self.error(f"{where} is not strict finite JSON: {exc}")
            return None
        self._strict_json_paths.add(path)
        return value

    def check_file_record(self, record: Any, where: str) -> Path | None:
        mapping = self.require_mapping(record, where)
        if mapping is None:
            return None
        path = self.local_file(mapping.get("path"), f"{where}.path")
        expected_hash = mapping.get("sha256")
        expected_bytes = mapping.get("bytes")
        if not isinstance(expected_hash, str) or not _SHA256_RE.fullmatch(expected_hash):
            self.error(f"{where}.sha256 must be a 64-character hexadecimal digest")
        if not _is_int(expected_bytes) or expected_bytes < 0:
            self.error(f"{where}.bytes must be a non-negative integer")
        if path is not None:
            try:
                actual_bytes = path.stat().st_size
                actual_hash = _sha256(path)
            except OSError as exc:
                self.error(f"{where} could not be hashed: {exc}")
                return None
            if _is_int(expected_bytes) and actual_bytes != expected_bytes:
                self.error(
                    f"{where}.bytes mismatch: recorded {expected_bytes}, actual {actual_bytes}"
                )
            if (
                isinstance(expected_hash, str)
                and _SHA256_RE.fullmatch(expected_hash)
                and actual_hash != expected_hash.lower()
            ):
                self.error(
                    f"{where}.sha256 mismatch: recorded {expected_hash}, actual {actual_hash}"
                )
        return path

    def validate_manifest_identity(self, manifest: dict[str, Any]) -> None:
        if manifest.get("schema") != SCHEMA:
            self.error(f"schema must equal {SCHEMA!r}")
        study = self.require_mapping(manifest.get("study"), "study")
        if study is None:
            return
        if study.get("variant") != VARIANT:
            self.error(f"study.variant must equal {VARIANT!r}")

    def validate_checkpoint(self, manifest: dict[str, Any]) -> None:
        provenance = self.require_mapping(manifest.get("provenance"), "provenance")
        if provenance is None:
            return
        checkpoint = self.require_mapping(provenance.get("checkpoint"), "provenance.checkpoint")
        guard = self.require_mapping(
            provenance.get("no_training_guard"), "provenance.no_training_guard"
        )

        if checkpoint is not None:
            recorded_hash = checkpoint.get("sha256")
            if (
                not isinstance(recorded_hash, str)
                or recorded_hash.lower() != self.expected_checkpoint_sha256
            ):
                self.error(
                    "provenance.checkpoint.sha256 does not equal the expected checkpoint SHA256"
                )
            checkpoint_path = self.external_checkpoint_file(checkpoint.get("path"))
            if checkpoint_path is not None:
                try:
                    actual_hash = _sha256(checkpoint_path)
                    actual_bytes = checkpoint_path.stat().st_size
                except OSError as exc:
                    self.error(f"checkpoint file could not be hashed: {exc}")
                    actual_hash = ""
                    actual_bytes = -1
                if actual_hash != self.expected_checkpoint_sha256:
                    self.error(
                        "checkpoint file SHA256 does not equal the expected checkpoint SHA256: "
                        f"{actual_hash}"
                    )
                recorded_bytes = checkpoint.get("bytes")
                if not _is_int(recorded_bytes) or recorded_bytes != actual_bytes:
                    self.error("provenance.checkpoint.bytes does not match the checkpoint file")

            payload = self.require_mapping(
                checkpoint.get("payload"), "provenance.checkpoint.payload"
            )
            if payload is not None:
                if payload.get("variant") != VARIANT:
                    self.error(f"checkpoint payload variant must equal {VARIANT!r}")
                if not _is_int(payload.get("seed")) or payload.get("seed") != 501:
                    self.error("checkpoint payload seed must equal 501")
                if not _is_int(payload.get("epoch")) or payload.get("epoch") != 100:
                    self.error("checkpoint payload epoch must equal 100")

            inventory_before = checkpoint.get("parent_inventory_sha256_before")
            inventory_after = checkpoint.get("parent_inventory_sha256_after")
            if not isinstance(inventory_before, str) or not _SHA256_RE.fullmatch(inventory_before):
                self.error("checkpoint parent_inventory_sha256_before must be a SHA256 digest")
            if not isinstance(inventory_after, str) or not _SHA256_RE.fullmatch(inventory_after):
                self.error("checkpoint parent_inventory_sha256_after must be a SHA256 digest")
            if (
                isinstance(inventory_before, str)
                and isinstance(inventory_after, str)
                and inventory_before.lower() != inventory_after.lower()
            ):
                self.error("checkpoint parent inventory changed during the study")

        if guard is not None:
            for key in ("optimizer_step_calls", "parameter_update_calls", "linear_probe_runs"):
                value = guard.get(key)
                if not _is_int(value) or value != 0:
                    self.error(f"provenance.no_training_guard.{key} must equal integer zero")
            for key, value in guard.items():
                if (
                    _is_number(value)
                    and key.endswith(("_calls", "_runs", "_updates", "_steps"))
                    and value != 0
                ):
                    self.error(f"provenance.no_training_guard.{key} records training activity")

            before = guard.get("checkpoint_sha256_before")
            after = guard.get("checkpoint_sha256_after")
            if not isinstance(before, str) or before.lower() != self.expected_checkpoint_sha256:
                self.error(
                    "no-training checkpoint SHA256 before does not equal the expected digest"
                )
            if not isinstance(after, str) or after.lower() != self.expected_checkpoint_sha256:
                self.error("no-training checkpoint SHA256 after does not equal the expected digest")

        training_record = self.require_mapping(
            provenance.get("training_record"), "provenance.training_record"
        )
        if training_record is not None:
            expected = {
                "best_accuracy": 0.8834,
                "best_epoch": 99,
                "final_accuracy": 0.8830,
                "final_epoch": 100,
            }
            for key, expected_value in expected.items():
                value = training_record.get(key)
                if not _is_number(value) or not math.isclose(
                    float(value), float(expected_value), rel_tol=0.0, abs_tol=1e-6
                ):
                    self.error(f"provenance.training_record.{key} must equal {expected_value!r}")

    def validate_baseline(self, manifest: dict[str, Any]) -> None:
        baseline = self.require_mapping(manifest.get("baseline"), "baseline")
        if baseline is None:
            return
        raw_metrics = baseline.get("metrics", baseline)
        metrics = self.require_mapping(raw_metrics, "baseline.metrics")
        if metrics is None:
            return
        samples = metrics.get("samples")
        correct = metrics.get("correct")
        accuracy = metrics.get("accuracy")
        if not _is_int(samples) or samples != 5000:
            self.error("baseline.metrics.samples must equal 5000")
        if not _is_int(correct) or not (0 <= correct <= 5000):
            self.error("baseline.metrics.correct must be an integer from 0 through 5000")
        if not _is_number(accuracy) or not math.isfinite(float(accuracy)):
            self.error("baseline.metrics.accuracy must be finite")
        elif not (0.0 <= float(accuracy) <= 1.0):
            self.error("baseline.metrics.accuracy must be in [0, 1]")
        if _is_int(samples) and samples > 0 and _is_int(correct) and _is_number(accuracy):
            expected_accuracy = correct / samples
            if not math.isclose(float(accuracy), expected_accuracy, rel_tol=0.0, abs_tol=1e-12):
                self.error(
                    "baseline.metrics.accuracy must equal correct / samples "
                    f"({expected_accuracy!r})"
                )

    def _load_family(self, value: Any, family_id: str) -> dict[str, Any] | None:
        where = f"families.{family_id}"
        if isinstance(value, str):
            path = self.local_file(value, where)
            if path is None:
                return None
            loaded = self.strict_json_file(path, where)
            return self.require_mapping(loaded, where) if loaded is not None else None
        if isinstance(value, dict) and "family_id" not in value and "path" in value:
            path = self.check_file_record(value, where)
            if path is None:
                return None
            loaded = self.strict_json_file(path, where)
            return self.require_mapping(loaded, where) if loaded is not None else None
        family = self.require_mapping(value, where)
        if family is not None:
            try:
                _assert_finite(family, f"$.families.{family_id}")
            except StrictJSONError as exc:
                self.error(str(exc))
        return family

    def validate_families(self, manifest: dict[str, Any]) -> None:
        families = self.require_mapping(manifest.get("families"), "families")
        if families is None:
            return
        observed_family_ids = set(families)
        required_family_ids = set(FAMILY_IDS)
        missing = sorted(required_family_ids - observed_family_ids)
        extra = sorted(observed_family_ids - required_family_ids)
        if missing:
            self.error(f"families missing required IDs: {missing}")
        if extra:
            self.error(f"families contains unexpected IDs: {extra}")

        for family_id in FAMILY_IDS:
            if family_id not in families:
                continue
            family = self._load_family(families[family_id], family_id)
            if family is None:
                continue
            self.checked_families += 1
            where = f"families.{family_id}"
            if family.get("family_id") != family_id:
                self.error(f"{where}.family_id must equal {family_id!r}")
            if family.get("status") != "complete":
                self.error(f"{where}.status must equal 'complete'")

            required = family.get("required_condition_ids")
            observed = family.get("observed_condition_ids")
            conditions = family.get("conditions")
            for name, ids in (
                ("required_condition_ids", required),
                ("observed_condition_ids", observed),
            ):
                if (
                    not isinstance(ids, list)
                    or not ids
                    or not all(isinstance(item, str) and item for item in ids)
                ):
                    self.error(f"{where}.{name} must be a non-empty list of strings")
                elif len(ids) != len(set(ids)):
                    self.error(f"{where}.{name} contains duplicate IDs")
            if not isinstance(conditions, list) or not conditions:
                self.error(f"{where}.conditions must be a non-empty list")
                condition_ids: list[str] = []
                condition_by_id: dict[str, dict[str, Any]] = {}
            else:
                condition_ids = []
                condition_by_id = {}
                for index, condition in enumerate(conditions):
                    if not isinstance(condition, dict):
                        self.error(f"{where}.conditions[{index}] must be an object")
                        continue
                    condition_id = condition.get("condition_id")
                    if not isinstance(condition_id, str) or not condition_id:
                        self.error(
                            f"{where}.conditions[{index}].condition_id must be a non-empty string"
                        )
                    else:
                        condition_ids.append(condition_id)
                        condition_by_id[condition_id] = condition
                    status = condition.get("status")
                    if status not in {"complete", "unsupported"}:
                        self.error(
                            f"{where}.conditions[{index}].status must be complete or unsupported"
                        )
                    if status == "complete":
                        metrics = self.require_mapping(
                            condition.get("metrics"),
                            f"{where}.conditions[{index}].metrics",
                        )
                        if metrics is not None and not metrics:
                            self.error(f"{where}.conditions[{index}].metrics must not be empty")
                    if status == "unsupported":
                        reason = condition.get("reason")
                        if not isinstance(reason, str) or not reason.strip():
                            self.error(
                                f"{where}.conditions[{index}] unsupported condition needs a reason"
                            )
                        if condition_id not in ALLOWED_UNSUPPORTED.get(family_id, frozenset()):
                            self.error(
                                f"{where}.conditions[{index}] is not an allowed unsupported condition"
                            )
                    per_example_path = condition.get("per_example_path")
                    if per_example_path is not None:
                        path = self.local_file(
                            per_example_path,
                            f"{where}.conditions[{index}].per_example_path",
                        )
                        expected_hash = condition.get("per_example_sha256")
                        expected_bytes = condition.get("per_example_bytes")
                        if not isinstance(expected_hash, str) or not _SHA256_RE.fullmatch(
                            expected_hash
                        ):
                            self.error(
                                f"{where}.conditions[{index}].per_example_sha256 must be a digest"
                            )
                        if not _is_int(expected_bytes) or expected_bytes < 0:
                            self.error(
                                f"{where}.conditions[{index}].per_example_bytes must be non-negative"
                            )
                        if (
                            path is not None
                            and isinstance(expected_hash, str)
                            and _SHA256_RE.fullmatch(expected_hash)
                            and _sha256(path) != expected_hash.lower()
                        ):
                            self.error(
                                f"{where}.conditions[{index}] per-example SHA256 mismatch"
                            )
                        if (
                            path is not None
                            and _is_int(expected_bytes)
                            and path.stat().st_size != expected_bytes
                        ):
                            self.error(
                                f"{where}.conditions[{index}] per-example byte size mismatch"
                            )
                    if status == "complete" and "paired_effect" in condition:
                        metrics = self.require_mapping(
                            condition.get("metrics"),
                            f"{where}.conditions[{index}].metrics",
                        )
                        if metrics is not None and metrics.get("samples") != 5000:
                            self.error(
                                f"{where}.conditions[{index}].metrics.samples must equal 5000"
                            )
                        if per_example_path is None:
                            self.error(
                                f"{where}.conditions[{index}] paired effect lacks per-example artifact"
                            )
                    matched = condition.get("matched_control")
                    if isinstance(matched, dict) and matched:
                        repeats = matched.get("repeats")
                        control_ids = matched.get("condition_ids")
                        if not _is_int(repeats) or repeats < 20:
                            self.error(
                                f"{where}.conditions[{index}] matched control requires at least 20 repeats"
                            )
                        if not isinstance(control_ids, list) or len(control_ids) < 20:
                            self.error(
                                f"{where}.conditions[{index}] matched control needs 20 condition IDs"
                            )
                if len(condition_ids) != len(set(condition_ids)):
                    self.error(f"{where}.conditions contains duplicate condition IDs")

            if isinstance(required, list) and isinstance(observed, list):
                if required != observed:
                    self.error(
                        f"{where} required_condition_ids must exactly equal observed_condition_ids"
                    )
                if set(required) != set(condition_ids) or len(required) != len(condition_ids):
                    self.error(f"{where}.conditions IDs must exactly cover required_condition_ids")
                expected_ids = set(EXPECTED_CONDITION_IDS[family_id])
                if set(required) != expected_ids or len(required) != len(expected_ids):
                    self.error(
                        f"{where}.required_condition_ids do not match the independent protocol contract"
                    )

            if family_id == "F04_pole_targeted":
                for target_id in F04_TARGETS:
                    target = condition_by_id.get(target_id)
                    matched = target.get("matched_control") if isinstance(target, dict) else None
                    if not isinstance(matched, dict):
                        self.error(f"{where}.{target_id} lacks matched_control")
                contrasts = family.get("matched_control_contrasts")
                if not isinstance(contrasts, list):
                    self.error(f"{where}.matched_control_contrasts must be a list")
                else:
                    contrast_ids = {
                        item.get("target_condition_id")
                        for item in contrasts
                        if isinstance(item, dict)
                    }
                    if contrast_ids != set(F04_TARGETS):
                        self.error(
                            f"{where}.matched_control_contrasts must cover all targeted bands"
                        )
                    for index, contrast in enumerate(contrasts):
                        if not isinstance(contrast, dict):
                            continue
                        for key in (
                            "accuracy_delta_pp_vs_random_mean",
                            "accuracy_delta_pp_vs_random_mean_ci95",
                            "empirical_p_more_harmful",
                            "empirical_q_more_harmful_bh",
                        ):
                            if key not in contrast:
                                self.error(
                                    f"{where}.matched_control_contrasts[{index}] lacks {key}"
                                )

            if family_id == "F11_robustness_invariance":
                cross = self.require_mapping(
                    family.get("intervention_cross"),
                    f"{where}.intervention_cross",
                )
                if cross is not None:
                    if cross.get("status") != "complete":
                        self.error(f"{where}.intervention_cross.status must be complete")
                    if set(cross.get("condition_ids", [])) != {
                        f"half_life_cap_2_x_{condition_id}"
                        for condition_id in F11_CROSS_BASES
                    }:
                        self.error(
                            f"{where}.intervention_cross.condition_ids do not match protocol"
                        )

    def validate_artifacts(self, manifest: dict[str, Any]) -> None:
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list):
            self.error("artifacts must be a list")
            return
        artifact_ids: set[str] = set()
        artifact_paths: set[str] = set()
        for index, artifact in enumerate(artifacts):
            where = f"artifacts[{index}]"
            mapping = self.require_mapping(artifact, where)
            if mapping is None:
                continue
            artifact_id = mapping.get("artifact_id")
            if not isinstance(artifact_id, str) or not artifact_id:
                self.error(f"{where}.artifact_id must be a non-empty string")
            elif artifact_id in artifact_ids:
                self.error(f"duplicate artifact_id {artifact_id!r}")
            else:
                artifact_ids.add(artifact_id)
            raw_path = mapping.get("path")
            if isinstance(raw_path, str) and raw_path in artifact_paths:
                self.error(f"duplicate artifact path {raw_path!r}")
            elif isinstance(raw_path, str):
                artifact_paths.add(raw_path)
            path = self.check_file_record(mapping, where)
            if path is None:
                continue
            self.checked_artifacts += 1
            media_type = mapping.get("media_type")
            if (
                path.suffix.lower() == ".json" or media_type == "application/json"
            ) and path not in self._strict_json_paths:
                self.strict_json_file(path, where)

    def _check_figure_file(self, figure: dict[str, Any], prefix: str, where: str) -> Path | None:
        record = {
            "path": figure.get(f"{prefix}path"),
            "sha256": figure.get(f"{prefix}sha256"),
            "bytes": figure.get(f"{prefix}bytes"),
        }
        return self.check_file_record(record, where)

    def validate_figures(self, manifest: dict[str, Any]) -> None:
        figures = manifest.get("figures")
        if not isinstance(figures, list):
            self.error("figures must be a list")
            return
        found: dict[str, dict[str, Any]] = {}
        for index, value in enumerate(figures):
            figure = self.require_mapping(value, f"figures[{index}]")
            if figure is None:
                continue
            figure_id = figure.get("figure_id")
            if not isinstance(figure_id, str) or not figure_id:
                self.error(f"figures[{index}].figure_id must be a non-empty string")
                continue
            if figure_id in found:
                self.error(f"duplicate figure_id {figure_id!r}")
                continue
            found[figure_id] = figure

        missing = [figure_id for figure_id in FIGURE_IDS if figure_id not in found]
        if missing:
            self.error(f"figures missing required IDs: {missing}")
        for figure_id in FIGURE_IDS:
            figure = found.get(figure_id)
            if figure is None:
                continue
            where = f"figures.{figure_id}"
            image_path = self._check_figure_file(figure, "", f"{where}.file")
            sidecar_path = self._check_figure_file(figure, "sidecar_", f"{where}.sidecar")
            if image_path is not None and image_path.suffix.lower() not in {".png", ".pdf", ".svg"}:
                self.error(f"{where}.path must be a PNG, PDF, or SVG file")
            if sidecar_path is not None:
                sidecar = self.strict_json_file(sidecar_path, f"{where}.sidecar")
                if not isinstance(sidecar, dict):
                    self.error(f"{where}.sidecar must contain a JSON object")
                elif sidecar.get("figure_id") not in (None, figure_id):
                    self.error(f"{where}.sidecar figure_id does not match")
            if image_path is not None and sidecar_path is not None:
                self.checked_figures += 1

    def validate_report(self, manifest: dict[str, Any]) -> None:
        report = self.require_mapping(manifest.get("report"), "report")
        if report is None:
            return
        path = self.local_file(report.get("path"), "report.path")
        if path is None:
            return
        if "sha256" in report or "bytes" in report:
            self.check_file_record(report, "report")
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            self.error(f"report cannot be read as UTF-8: {exc}")
            return
        if not _HANGUL_RE.search(text):
            self.error("report must be written in Korean")
        lowered = " ".join(text.lower().split())
        if "88.34" not in lowered or "88.30" not in lowered:
            self.error(
                "report must mention both the historical 88.34% and epoch-100 88.30% records"
            )
            return
        historical_index = lowered.find("88.34")
        epoch100_index = lowered.find("88.30")
        historical_context = lowered[max(0, historical_index - 100) : historical_index + 100]
        epoch100_context = lowered[max(0, epoch100_index - 100) : epoch100_index + 100]
        if not any(word in historical_context for word in _HISTORICAL_WORDS):
            self.error("report must identify 88.34% as a historical/best record (epoch 99)")
        if not any(word in epoch100_context for word in _EPOCH100_WORDS):
            self.error("report must identify 88.30% with the epoch-100 checkpoint")

    def run(self) -> dict[str, Any]:
        if not _SHA256_RE.fullmatch(self.expected_checkpoint_sha256):
            self.error("--checkpoint-sha256 must be a 64-character hexadecimal digest")
        if not self.root.exists() or not self.root.is_dir():
            self.error(f"study root is not a directory: {os.fspath(self.root_argument)!r}")
            return self.summary()

        manifest_path = self.local_file(MANIFEST_NAME, MANIFEST_NAME)
        if manifest_path is None:
            return self.summary()
        manifest_value = self.strict_json_file(manifest_path, MANIFEST_NAME)
        manifest = self.require_mapping(manifest_value, MANIFEST_NAME)
        if manifest is None:
            return self.summary()

        self.validate_manifest_identity(manifest)
        study = self.require_mapping(manifest.get("study"), "study")
        validation = self.require_mapping(manifest.get("validation"), "validation")
        if study is not None:
            status = study.get("status")
            if status not in {"complete_pending_validation", "complete_validated"}:
                self.error(
                    "study.status must be complete_pending_validation or complete_validated"
                )
            if status == "complete_validated" and (
                validation is None
                or validation.get("ok") is not True
                or validation.get("expected_checkpoint_sha256")
                != self.expected_checkpoint_sha256
            ):
                self.error("validated study status requires matching successful validation evidence")
        self.validate_checkpoint(manifest)
        self.validate_baseline(manifest)
        self.validate_families(manifest)
        self.validate_artifacts(manifest)
        self.validate_figures(manifest)
        self.validate_report(manifest)
        return self.summary()

    def summary(self) -> dict[str, Any]:
        return {
            "ok": not self.errors,
            "root": os.fspath(self.root),
            "manifest": MANIFEST_NAME,
            "expected_checkpoint_sha256": self.expected_checkpoint_sha256,
            "checked_families": self.checked_families,
            "checked_artifacts": self.checked_artifacts,
            "checked_figures": self.checked_figures,
            "errors": self.errors,
        }


def validate_study(root: str | os.PathLike[str], checkpoint_sha256: str) -> dict[str, Any]:
    """Validate a study root and return the same summary emitted by the CLI."""

    return StudyValidator(Path(root), checkpoint_sha256).run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="study output root containing study.json"
    )
    parser.add_argument(
        "--checkpoint-sha256",
        required=True,
        help="expected immutable checkpoint SHA256",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="record a successful validation in study.json and mark the study complete",
    )
    return parser


def _atomic_manifest(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = validate_study(args.root, args.checkpoint_sha256)
    if args.finalize and summary["ok"]:
        manifest_path = args.root / MANIFEST_NAME
        manifest = _load_json(manifest_path)
        manifest["study"]["status"] = "complete_validated"
        manifest["validation"] = {
            **summary,
            "validated_utc": dt.datetime.now(dt.UTC).isoformat(),
        }
        _atomic_manifest(manifest_path, manifest)
        summary = validate_study(args.root, args.checkpoint_sha256)
    json.dump(summary, sys.stdout, ensure_ascii=False, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
