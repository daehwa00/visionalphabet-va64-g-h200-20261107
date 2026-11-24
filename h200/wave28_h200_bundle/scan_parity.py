"""Check the selected Hopper scan preset against off and FP32."""

import json
import sys
from itertools import product
from pathlib import Path
from unittest.mock import patch

import validate


def main() -> None:
    output, preset = sys.argv[1:]
    # CUDA coarse requires even axes. Use the next even size for the 7x7 stage.
    cases = (
        *((size, size, 160, coarse) for size, coarse in product((56, 28, 14), (False, True))),
        (7, 7, 128, False),
        (8, 8, 128, True),
    )
    results = []
    with patch.object(validate, "PRESETS", ("off", preset)):
        for case, dtype, interleaved in product(cases, validate.DTYPES, (False, True)):
            try:
                rows = list(validate.run(case, dtype, interleaved=interleaved, forced=False))
                results.extend(row for row in rows if row["preset"] == preset)
            except (ValueError, RuntimeError) as exc:
                results.append({
                    "case": case, "dtype": str(dtype), "interleaved": interleaved,
                    "error": str(exc),
                })
            passed = all("error" not in row for row in results)
            temporary = Path(output).with_suffix(".tmp")
            temporary.write_text(json.dumps({
                "preset": preset, "expected_cases": 32,
                "passed": len(results) == 32 and passed, "cases": results,
            }, separators=(",", ":")))
            temporary.replace(output)
    raise SystemExit(int(not passed))


if __name__ == "__main__":
    main()
