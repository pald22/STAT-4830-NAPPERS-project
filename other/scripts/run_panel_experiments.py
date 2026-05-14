"""Run all panel experiments sequentially with reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from polymarket.panel_experiments import (  # noqa: E402
    add_features,
    load_panel,
    run_exp_01,
    run_exp_02,
    run_exp_03,
    run_exp_04,
    run_exp_05,
    to_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Polymarket panel experiments (1-5).")
    parser.add_argument("--csv", type=Path, default=ROOT / "polymarket_daily_panel_60plus.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "polymarket_panel_experiments")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    report_dir = args.out / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    df = add_features(load_panel(args.csv))

    exp_runs = [
        ("Experiment 1: Persistent Predictability", run_exp_01),
        ("Experiment 2: Liquidity and Forecastability", run_exp_02),
        ("Experiment 3: Cross-Sectional Factors", run_exp_03),
        ("Experiment 4: Unresolved Market Risk", run_exp_04),
        ("Experiment 5: Validation Design", run_exp_05),
    ]

    summary_lines = ["# Panel Experiment Master Results", "", "## Experiment index", ""]
    for i, (name, fn) in enumerate(exp_runs, start=1):
        out_i = args.out / f"exp_{i:02d}"
        out_i.mkdir(parents=True, exist_ok=True)
        result = fn(df, out_i)
        report_path = report_dir / f"exp_{i:02d}_report.md"
        to_markdown(name, result, report_path)
        summary_lines.append(f"- [{name}](./reports/{report_path.name})")

        failed = [k for k, v in result["checks"].items() if not v]
        if failed:
            summary_lines.append(f"  - Status: FAIL ({len(failed)} sanity checks)")
            summary_lines.append(f"  - Failed checks: {', '.join(failed)}")
        else:
            summary_lines.append("  - Status: PASS")

    summary_lines += [
        "",
        "## Notes",
        "",
        "- Split policy enforces non-overlap within each experiment.",
        "- Overlap across different experiments is allowed by design.",
        "- Hyperparameter/model selection is done with validation only in each experiment runner.",
    ]
    (args.out / "master_report.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Wrote master report to: {args.out / 'master_report.md'}")


if __name__ == "__main__":
    main()

