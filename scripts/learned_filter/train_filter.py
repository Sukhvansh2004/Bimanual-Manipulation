#!/usr/bin/env python3
"""Train the learned handover filter (feasibility + cost) from handover logs.

Each ``--train`` / ``--test`` entry is a handover-log NPZ written by
``segment_assignment.py --handover-log`` (one per planning instance).
Splitting by file keeps evaluation on unseen grasp sets / seeds.

Usage::

    python scripts/learned_filter/train_filter.py \
        --train results/campaign/*/s100[1-3]/handover_log_*.npz \
        --test  results/campaign/*/s100[4-5]/handover_log_*.npz \
        --output models/handover_filter.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from bimanual_manipulation.planning.assignment.learned_filter import (  # noqa: E402
    ARCHITECTURES,
    HandoverFeasibilityNet,
)

THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
DEFAULT_WIDTH = {"mlp": 256, "transformer": 64}


def load_logs(paths: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and concatenate handover logs.

    Args:
        paths: Handover-log NPZ files; empty logs are skipped.

    Returns:
        ``(features (M, N_FEATURES), feasible (M,) float32, cost (M,))``.
    """
    X, y, c = [], [], []
    for p in paths:
        d = np.load(p)
        if len(d["feasible"]) == 0:
            continue
        X.append(d["features"])
        y.append(d["feasible"])
        c.append(d["cost"])
    return (np.concatenate(X), np.concatenate(y).astype(np.float32),
            np.concatenate(c).astype(np.float32))


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Area under the ROC curve via the rank-sum statistic.

    Args:
        y: ``(M,)`` binary labels.
        s: ``(M,)`` scores, higher = more likely positive.

    Returns:
        AUROC in ``[0, 1]``, or ``nan`` if only one class is present.
    """
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def threshold_report(y: np.ndarray, p: np.ndarray) -> List[Dict[str, float]]:
    """Skip rate and recall of the feasibility filter at each threshold.

    Args:
        y: ``(M,)`` feasibility labels.
        p: ``(M,)`` predicted ``P(feasible)``.

    Returns:
        One dict per threshold in :data:`THRESHOLDS` with the fraction of
        candidates skipped, feasible candidates kept (recall), and
        infeasible candidates skipped.
    """
    rows = []
    for t in THRESHOLDS:
        keep = p >= t
        rows.append({
            "threshold": t,
            "skip_rate": float(1 - keep.mean()),
            "feasible_recall": float(keep[y == 1].mean()) if y.sum() else float("nan"),
            "infeasible_skipped": float((~keep)[y == 0].mean()) if (1 - y).sum() else float("nan"),
        })
    return rows


def cost_report(y: np.ndarray, cost: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Accuracy of the cost head on feasible handovers.

    Args:
        y: ``(M,)`` feasibility labels.
        cost: ``(M,)`` true handover costs.
        pred: ``(M,)`` predicted handover costs.

    Returns:
        Median / 10th / 90th percentile of ``pred / cost``, mean absolute
        log error and Spearman rank correlation; empty if no feasible
        samples.
    """
    m = (y == 1) & np.isfinite(cost) & (cost > 0)
    if not m.any():
        return {}
    rel = pred[m] / cost[m]
    return {
        "median_ratio": float(np.median(rel)),
        "p10_ratio": float(np.percentile(rel, 10)),
        "p90_ratio": float(np.percentile(rel, 90)),
        "log_mae": float(np.mean(np.abs(np.log(pred[m]) - np.log(cost[m])))),
        "spearman": float(np.corrcoef(np.argsort(np.argsort(pred[m])),
                                      np.argsort(np.argsort(cost[m])))[0, 1]),
    }


def single_query_latency_ms(model: HandoverFeasibilityNet, X: np.ndarray, n: int = 500) -> float:
    """Median latency of one single-candidate prediction, as used in the DP.

    Args:
        model: Trained model (on CPU, in eval mode).
        X: Feature rows to sample queries from.
        n: Number of timed queries.

    Returns:
        Median latency in milliseconds.
    """
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        model.predict(X[i % len(X)])
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1e3)


def main() -> None:
    """Train the model, report held-out metrics and save the checkpoint."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--arch", choices=ARCHITECTURES, default="mlp")
    ap.add_argument("--hidden", type=int, default=None,
                    help="MLP width / transformer d_model (default: 256 / 64).")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=3.0,
                    help="BCE weight on feasible samples (>1 favours feasible recall).")
    ap.add_argument("--cost-weight", type=float, default=1.0,
                    help="Weight of the log-cost regression loss.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Xtr, ytr, ctr = load_logs(args.train)
    Xte, yte, cte = load_logs(args.test)
    print(f"train: {len(ytr)} samples ({ytr.mean():.1%} feasible) from {len(args.train)} logs")
    print(f"test:  {len(yte)} samples ({yte.mean():.1%} feasible) from {len(args.test)} logs")

    hidden = args.hidden or DEFAULT_WIDTH[args.arch]
    model = HandoverFeasibilityNet(Xtr.mean(0), Xtr.std(0) + 1e-6, hidden=hidden, arch=args.arch)
    print(f"arch={args.arch}  width={hidden}  parameters={model.n_parameters():,}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.net.to(device)

    feasible_cost = (ytr == 1) & np.isfinite(ctr) & (ctr > 0)
    log_cost = np.where(feasible_cost, np.log(np.where(feasible_cost, ctr, 1.0)), 0.0)
    Xtr_t = torch.as_tensor(Xtr, device=device)
    ytr_t = torch.as_tensor(ytr, device=device)
    ctr_t = torch.as_tensor(log_cost, dtype=torch.float32, device=device)
    mtr_t = torch.as_tensor(feasible_cost, dtype=torch.float32, device=device)
    pos_weight = torch.tensor(args.pos_weight, device=device)

    opt = torch.optim.AdamW(model.net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    batch = 512
    for epoch in range(args.epochs):
        model.net.train()
        perm = torch.randperm(len(ytr_t), device=device)
        total = 0.0
        for k in range(0, len(perm), batch):
            idx = perm[k:k + batch]
            logit, pred_log_cost = model.forward(Xtr_t[idx])
            loss_feas = F.binary_cross_entropy_with_logits(
                logit, ytr_t[idx], pos_weight=pos_weight,
            )
            m = mtr_t[idx]
            loss_cost = (F.smooth_l1_loss(pred_log_cost, ctr_t[idx], reduction="none") * m
                         ).sum() / m.sum().clamp(min=1.0)
            loss = loss_feas + args.cost_weight * loss_cost
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        sched.step()
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1:3d}  loss {total / len(perm):.4f}")

    model.net.eval().cpu()
    p_tr, _ = model.predict(Xtr)
    p_te, c_te = model.predict(Xte)
    report = {
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "train_feasible_frac": float(ytr.mean()), "test_feasible_frac": float(yte.mean()),
        "train_auroc": auroc(ytr, p_tr), "test_auroc": auroc(yte, p_te),
        "test_thresholds": threshold_report(yte, p_te),
        "test_cost": cost_report(yte, cte, c_te),
        "arch": args.arch, "width": hidden, "n_parameters": model.n_parameters(),
        "latency_ms": single_query_latency_ms(model, Xte),
        "args": vars(args),
    }
    print(f"AUROC train {report['train_auroc']:.3f}  test {report['test_auroc']:.3f}  "
          f"single-query latency {report['latency_ms']:.2f} ms")
    for r in report["test_thresholds"]:
        print("  τ={threshold:.2f}  skip={skip_rate:.1%}  feasible recall={feasible_recall:.1%}  "
              "infeasible skipped={infeasible_skipped:.1%}".format(**r))
    if report["test_cost"]:
        print("  cost head: median pred/true={median_ratio:.2f} (p10 {p10_ratio:.2f}, "
              "p90 {p90_ratio:.2f}), log-MAE {log_mae:.3f}, Spearman {spearman:.3f}"
              .format(**report["test_cost"]))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    with open(Path(args.output).with_suffix(".json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
