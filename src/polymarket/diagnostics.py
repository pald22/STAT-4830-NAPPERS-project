"""Plots for mispricing pipeline runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_run(result: dict[str, Any], out_dir: Path | str | None = None) -> None:
    out_dir = Path(out_dir) if out_dir is not None else None
    metrics = result.get("metrics", [])
    if not metrics:
        return
    folds = [m["fold"] for m in metrics]
    obj = [m["objective"] for m in metrics]
    mse = [m["mse_p_vs_implied"] for m in metrics]
    turn = [m["turnover_l1"] for m in metrics]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes[0, 0]
    ax.bar(folds, obj, color="steelblue")
    ax.set_title("Objective by fold")
    ax.set_xlabel("fold")

    ax = axes[0, 1]
    ax.bar(folds, mse, color="coral")
    ax.set_title("MSE(p_hat vs implied)")
    ax.set_xlabel("fold")

    ax = axes[1, 0]
    ax.bar(folds, turn, color="darkgreen")
    ax.set_title("Turnover ||w - w_prev||_1")
    ax.set_xlabel("fold")

    ax = axes[1, 1]
    wdf = result.get("weights")
    if isinstance(wdf, pd.DataFrame) and len(wdf) and "edge" in wdf.columns and "weight" in wdf.columns:
        ax.scatter(wdf["edge"], wdf["weight"], s=6, alpha=0.35)
        ax.set_xlabel("edge (p_hat - c_exec)")
        ax.set_ylabel("weight")
        ax.set_title("Weights vs edge")
    else:
        ax.set_visible(False)

    plt.tight_layout()
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "diagnostics.png", dpi=120)
    plt.close(fig)
