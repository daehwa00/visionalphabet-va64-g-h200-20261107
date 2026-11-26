"""Run one scan lock-in ABBA arm using the Wave 51 full-model measurement protocol."""

import argparse
import os
from pathlib import Path

from h200.wave49_h200_lockin.run_wave49_h200 import measure_arm
from h200.wave51_h200_lockin.run_wave51_h200 import REF, WIN, arm_env


def lockin_arm_env(name: str) -> dict[str, str]:
    mode = name.removesuffix("_rev")
    if mode == REF:
        return {**arm_env(REF), "LNET_SCAN_W52A": "off", "LNET_SCAN_W53A": "off",
                "LNET_FUSED_STEM_CONV": "auto"}
    if mode == "w57_lockin":
        return {**lockin_arm_env("w55_lockin"), "LNET_SCAN_W53A": mode,
                "LNET_MODEL_W56B": "swiglu_vec8_tanh", "LNET_MODEL_W57B_RMS": "cuda_sp_vec8_4cta"}
    if mode == "w55_lockin":
        return {**lockin_arm_env("w53_lockin"), "LNET_SCAN_W53A": mode,
                "LNET_SCAN_HSCAN": "h55_lockin"}
    if mode != "w53_lockin":
        raise ValueError(f"Unknown arm {name!r}")
    return {**arm_env(WIN), "LNET_SCAN_W52A": "off", "LNET_SCAN_W53A": "w53_lockin",
            "LNET_SCAN_HSCAN": "h53_lockin", "LNET_FUSED_STEM_CONV": "on",
            "LNET_STEM_STAGE0_MODE": "triton_v3_m64_w4"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    arms = (REF, "w53_lockin", "w55_lockin", "w57_lockin")
    parser.add_argument("--arm", required=True, choices=(*arms, *(arm + "_rev" for arm in arms)))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    env = lockin_arm_env(args.arm)
    os.environ.update(env)
    return measure_arm(args.out, profile=args.profile, arm_env_keys=tuple(env))


if __name__ == "__main__":
    raise SystemExit(main())
