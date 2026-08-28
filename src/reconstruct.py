"""Minimal reconstruction entry point.

This module keeps reconstruction separate from training. It accepts a saved
prediction tensor and writes it to an experiment-specific artifact. The
actual MindEye2 SDXL reconstruction should be connected here after the
verified environment/checkpoint test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Saved model prediction (.pt)")
    parser.add_argument("--output", required=True, help="Output artifact (.pt)")
    args = parser.parse_args()

    source = Path(args.input)
    if not source.exists():
        raise FileNotFoundError(source)

    prediction = torch.load(source, map_location="cpu")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prediction, destination)
    print(f"Saved reconstruction artifact to {destination}")


if __name__ == "__main__":
    main()
