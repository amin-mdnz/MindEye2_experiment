"""Validate local Subject-8 data and create an experiment manifest.

No dataset is downloaded automatically. This avoids silently changing the
experimental dataset and keeps large NSD artifacts out of Git.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BUDGETS = {"60m": 750, "30m": 375, "15m": 188, "7_5m": 94}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", default="experiments/subject08_data_efficiency/subj08_manifest.json")
    args = parser.parse_args()

    data_path = Path(args.data_path).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Subject-8 data path does not exist: {data_path}")

    manifest = {
        "subject": 8,
        "data_path": str(data_path),
        "budgets": BUDGETS,
        "note": "Verify sample-to-minute mapping against the local NSD/MindEye2 dataset before training.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
