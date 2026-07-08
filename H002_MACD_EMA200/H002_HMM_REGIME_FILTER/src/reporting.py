from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_markdown_report(
    path: str | Path,
    regime_summary: pd.DataFrame,
    significance: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sections = [
        "# H002 HMM Regime Filter Report",
        "## Regime Trade Summary",
        regime_summary.to_markdown(index=False) if not regime_summary.empty else "No labeled trades.",
        "## Statistical Tests",
        significance.to_markdown(index=False) if not significance.empty else "No statistical table.",
        "## Baseline vs Filtered",
        comparison.to_markdown(index=False) if not comparison.empty else "No comparison table.",
    ]
    Path(path).write_text("\n\n".join(sections) + "\n", encoding="utf-8")
