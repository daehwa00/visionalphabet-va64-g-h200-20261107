"""Run existing parity tests with an experimental scan plan."""

import argparse
import os

import pytest
from variants import BACKWARD, FORWARD, select


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward", choices=("baseline", *FORWARD), default="baseline")
    parser.add_argument("--backward", choices=("baseline", *BACKWARD), default="baseline")
    args, pytest_args = parser.parse_known_args()
    os.environ.pop("LNET_VSCAN_DECOUPLED", None)
    select((args.forward, args.backward))
    raise SystemExit(pytest.main(pytest_args))


if __name__ == "__main__":
    main()
