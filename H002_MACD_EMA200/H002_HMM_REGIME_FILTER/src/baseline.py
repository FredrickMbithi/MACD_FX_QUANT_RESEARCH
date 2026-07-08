from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


H001_SRC = Path(__file__).resolve().parents[2] / "src"
if str(H001_SRC) not in sys.path:
    sys.path.insert(0, str(H001_SRC))

import execution
import indicators
import signals


def build_baseline_trades(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = cfg["baseline"]
    indicators_df = indicators.add_macd_indicators(
        raw,
        fast_span=base["macd_fast_span"],
        slow_span=base["macd_slow_span"],
        signal_span=base["macd_signal_span"],
    )
    signals_df = signals.generate_signals(indicators_df)
    trades = execution.run_backtest(
        signals_df,
        lot_size=base["lot_size"],
        commission_per_lot=base["commission_per_lot"],
        swap_long_per_night=base["swap_long_per_night"],
        swap_short_per_night=base["swap_short_per_night"],
        stop_loss_pips=base["stop_loss_pips"],
        take_profit_pips=base["take_profit_pips"],
        pip_size=base["pip_size"],
    )
    return trades, signals_df
