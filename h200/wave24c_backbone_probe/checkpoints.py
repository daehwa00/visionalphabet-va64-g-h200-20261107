"""Check strict checkpoint interoperability against the isolated baseline."""

# ruff: noqa: ANN201, S101, T201
from run_va_eight_ablations import build_variant

VARIANTS = (
    "17-p1-nr4-1024d-v3",
    "17-live-complex-rms-s03",
    "17-e2-multiscale-s3s4-live-rms-v2",
)


def main():
    for variant in VARIANTS:
        reference = build_variant(variant)
        candidate = build_variant(variant)
        original = reference.state_dict()
        updated = candidate.state_dict()
        assert tuple(original) == tuple(updated)
        candidate.load_state_dict(original, strict=True)
        reference.load_state_dict(candidate.state_dict(), strict=True)
        print(variant, len(original), "strict both directions passed")


if __name__ == "__main__":
    main()
