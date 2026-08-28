"""Evaluation utilities for the Subject-8 data-efficiency experiment.

The evaluator is intentionally independent of the MindEye2 training loop so
that predictions can be evaluated against the same fixed Subject-8 test set.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F


def retrieval_metrics(predicted: torch.Tensor, target: torch.Tensor, topk=(1, 5, 10)):
    """Compute image/brain retrieval accuracy from paired feature matrices.

    Rows are paired: predicted[i] corresponds to target[i]. Returns top-k
    diagonal retrieval accuracy in both directions.
    """
    predicted = F.normalize(predicted.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    sim = predicted @ target.T
    labels = torch.arange(sim.shape[0], device=sim.device)

    result = {}
    for k in topk:
        k_eff = min(k, sim.shape[1])
        pred_top = sim.topk(k_eff, dim=1).indices
        result[f"forward_top{k}"] = (pred_top == labels[:, None]).any(dim=1).float().mean().item()

        rev_top = sim.T.topk(k_eff, dim=1).indices
        result[f"backward_top{k}"] = (rev_top == labels[:, None]).any(dim=1).float().mean().item()
    return result


def reconstruction_metrics(predicted: torch.Tensor, target: torch.Tensor):
    """Return MSE, Pearson correlation, and cosine similarity for reconstructions."""
    predicted = predicted.float()
    target = target.float()
    p = predicted.flatten(1)
    t = target.flatten(1)
    p_center = p - p.mean(dim=1, keepdim=True)
    t_center = t - t.mean(dim=1, keepdim=True)
    pearson = (p_center * t_center).sum(1) / (
        p_center.norm(dim=1) * t_center.norm(dim=1) + 1e-8
    )
    cosine = F.cosine_similarity(p, t, dim=1)
    return {
        "mse": F.mse_loss(predicted, target).item(),
        "pearson": pearson.mean().item(),
        "cosine": cosine.mean().item(),
    }


def load_tensor(path: str):
    value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        for key in ("predicted", "prediction", "features", "embeddings"):
            if key in value:
                return value[key]
        raise KeyError(f"No prediction tensor found in {path}; keys={list(value)}")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help=".pt file containing predicted features/images")
    parser.add_argument("--targets", default=None, help=".pt file containing paired target features/images")
    parser.add_argument("--output", default="experiments/subject08_data_efficiency/results.csv")
    parser.add_argument("--condition", default="unknown")
    parser.add_argument("--budget", default="unknown")
    args = parser.parse_args()

    predicted = load_tensor(args.predictions)
    if args.targets is None:
        raise SystemExit("--targets is required for evaluation")
    target = load_tensor(args.targets)

    if predicted.shape != target.shape:
        raise ValueError(f"Prediction/target shapes differ: {predicted.shape} vs {target.shape}")

    metrics = {}
    if predicted.ndim >= 2:
        metrics.update(retrieval_metrics(predicted, target))
    metrics.update(reconstruction_metrics(predicted, target))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    row = {"condition": args.condition, "budget": args.budget, **metrics}

    existing = []
    if output.exists() and output.stat().st_size:
        with output.open(newline="") as f:
            existing = list(csv.DictReader(f))
    existing.append(row)
    fields = list(row.keys())
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing)

    print(row)


if __name__ == "__main__":
    main()
