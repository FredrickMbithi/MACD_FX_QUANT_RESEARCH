from __future__ import annotations

import numpy as np
import pandas as pd


def _net_profit(trades: pd.DataFrame) -> pd.Series:
    return trades["money_profit"] + trades["commission"] + trades["swap"]


def add_trade_r_multiples(trades: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    result = trades.copy()
    base = cfg["baseline"]
    risk_price = base["stop_loss_pips"] * base["pip_size"]
    result["r_multiple"] = result["profit"] / risk_price
    result["net_profit"] = _net_profit(result)
    result["holding_bars"] = np.nan
    return result


def add_mae_mfe(
    trades: pd.DataFrame,
    bars: pd.DataFrame,
    cfg: dict,
    timestamp_col: str = "time_utc",
    high_col: str = "high",
    low_col: str = "low",
) -> pd.DataFrame:
    result = trades.copy()
    base = cfg["baseline"]
    risk_price = base["stop_loss_pips"] * base["pip_size"]
    bars_work = bars.copy()
    bars_work[timestamp_col] = pd.to_datetime(bars_work[timestamp_col])

    maes = []
    mfes = []
    holding_bars = []
    for trade in result.itertuples(index=False):
        entry_time = pd.Timestamp(trade.entry_time)
        exit_time = pd.Timestamp(trade.exit_time)
        path = bars_work[(bars_work[timestamp_col] >= entry_time) & (bars_work[timestamp_col] <= exit_time)]
        holding_bars.append(len(path))
        if path.empty:
            maes.append(np.nan)
            mfes.append(np.nan)
            continue
        if trade.direction == "LONG":
            adverse = (trade.entry_price - path[low_col].min()) / risk_price
            favorable = (path[high_col].max() - trade.entry_price) / risk_price
        else:
            adverse = (path[high_col].max() - trade.entry_price) / risk_price
            favorable = (trade.entry_price - path[low_col].min()) / risk_price
        maes.append(float(adverse))
        mfes.append(float(favorable))

    result["mae_r"] = maes
    result["mfe_r"] = mfes
    result["holding_bars"] = holding_bars
    return result


def label_trades_by_entry_regime(
    trades: pd.DataFrame,
    regimes: pd.DataFrame,
    timestamp_col: str = "time_utc",
    regime_col: str = "confirmed_regime",
) -> pd.DataFrame:
    trade_work = trades.copy()
    trade_work["entry_time"] = pd.to_datetime(trade_work["entry_time"])

    regime_work = regimes.copy().sort_values(timestamp_col)
    regime_work[timestamp_col] = pd.to_datetime(regime_work[timestamp_col])

    labeled = pd.merge_asof(
        trade_work.sort_values("entry_time"),
        regime_work,
        left_on="entry_time",
        right_on=timestamp_col,
        direction="backward",
    )
    labeled = labeled.rename(
        columns={
            regime_col: "entry_regime",
            "regime_probability": "entry_regime_probability",
        }
    )
    return labeled


def profit_factor(r_values: pd.Series) -> float:
    gains = r_values[r_values > 0].sum()
    losses = r_values[r_values < 0].sum()
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return float(gains / abs(losses))


def max_drawdown(r_values: pd.Series) -> float:
    equity = r_values.cumsum()
    drawdown = equity - equity.cummax()
    return float(drawdown.min()) if len(drawdown) else np.nan


def trade_sharpe(r_values: pd.Series) -> float:
    std = r_values.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return float(r_values.mean() / std * np.sqrt(len(r_values)))


def summarize_by_regime(labeled_trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, group in labeled_trades.dropna(subset=["entry_regime"]).groupby("entry_regime"):
        r = group["r_multiple"].astype(float)
        rows.append(
            {
                "regime": int(regime),
                "trades": int(len(group)),
                "win_rate": float((r > 0).mean()),
                "expectancy_r": float(r.mean()),
                "average_r": float(r.mean()),
                "profit_factor": profit_factor(r),
                "trade_sharpe": trade_sharpe(r),
                "max_drawdown_r": max_drawdown(r),
                "average_holding_bars": float(group["holding_bars"].mean()),
                "median_holding_bars": float(group["holding_bars"].median()),
                "average_mae_r": float(group["mae_r"].mean()) if "mae_r" in group else np.nan,
                "average_mfe_r": float(group["mfe_r"].mean()) if "mfe_r" in group else np.nan,
                "average_entry_probability": float(group["entry_regime_probability"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)
