from __future__ import annotations

import numpy as np
import pandas as pd

from regime_analysis import max_drawdown, profit_factor, trade_sharpe


def select_profitable_regimes(training_summary: pd.DataFrame, cfg: dict) -> list[int]:
    filt = cfg["filter"]
    candidates = training_summary[
        (training_summary["trades"] >= cfg["hmm"]["min_trades_per_state"])
        & (training_summary["expectancy_r"] >= filt["min_training_expectancy_r"])
        & (training_summary["profit_factor"] >= filt["min_training_profit_factor"])
    ]
    return [int(value) for value in candidates["regime"].tolist()]


def apply_regime_filter(labeled_trades: pd.DataFrame, allowed_regimes: list[int], cfg: dict) -> pd.DataFrame:
    min_probability = cfg["filter"]["min_entry_probability"]
    allowed = labeled_trades["entry_regime"].isin(allowed_regimes)
    confident = labeled_trades["entry_regime_probability"] >= min_probability
    return labeled_trades[allowed & confident].copy()


def strategy_metrics(trades: pd.DataFrame, r_col: str = "r_multiple") -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate": np.nan,
            "expectancy_r": np.nan,
            "profit_factor": np.nan,
            "trade_sharpe": np.nan,
            "max_drawdown_r": np.nan,
            "total_r": 0.0,
        }
    r = trades[r_col].astype(float)
    return {
        "trades": int(len(trades)),
        "win_rate": float((r > 0).mean()),
        "expectancy_r": float(r.mean()),
        "profit_factor": profit_factor(r),
        "trade_sharpe": trade_sharpe(r),
        "max_drawdown_r": max_drawdown(r),
        "total_r": float(r.sum()),
    }


def compare_baseline_filtered(baseline_trades: pd.DataFrame, filtered_trades: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"strategy": "baseline", **strategy_metrics(baseline_trades)},
            {"strategy": "hmm_filtered", **strategy_metrics(filtered_trades)},
        ]
    )
