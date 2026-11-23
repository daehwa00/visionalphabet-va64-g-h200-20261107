"""Recover the portable H200 bundle from the provisioning issue transcript."""

import argparse
import base64
import json
import lzma
import re
from pathlib import Path

MARKER = "H200_LIVERMS_V2_OPT_XZ"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("comments", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.comments.read_text())
    comments = payload["comments"] if isinstance(payload, dict) else payload
    summaries = [
        comment["body"] for comment in comments
        if "결과 보고서" in comment["body"] or MARKER in comment["body"]
    ]
    if not summaries:
        raise SystemExit("No completed H200 job found")
    output = args.comments.with_name(args.comments.name.removesuffix("-comments.json"))
    output.with_suffix(".log").write_text("\n\n".join(summaries) + "\n")
    encoded = re.findall(rf"\b{MARKER}\s+([A-Za-z0-9+/=]+)", "\n".join(summaries))
    if not encoded:
        raise SystemExit("The report does not contain the portable H200 bundle")
    data = json.loads(lzma.decompress(base64.b64decode(encoded[-1], validate=True)))
    output.with_name(output.name + "-result.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
