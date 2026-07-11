"""
indicators.py
=============

Purpose
-------
Calculate the technical indicators this strategy needs - EMA and MACD -
from a validated OHLC DataFrame.

This file implements the *math*, not the *decisions*. It answers
"what is the MACD value on this bar?", never "should I buy on this
bar?" - that question belongs to signals.py.

Responsibilities
-----------------
This file is ONLY responsible for turning a 'close' price column into
indicator columns. It deliberately does NOT:
- validate the data (that already happened in validate.py - this file
  trusts its input completely, which is only safe because validate.py
  ran first)
- generate trading signals (that's signals.py)
- know anything about strategy rules, thresholds, or trade direction

Inputs
------
A pandas DataFrame that has already passed validate.py's checks, sorted
oldest-to-newest, with at least a 'close' price column.

Outputs
-------
The same DataFrame with five new columns appended:
    macd_fast_ema, macd_slow_ema, macd, macd_signal, macd_histogram

Assumptions
-----------
- Input data is already clean (no NaNs, no duplicate/out-of-order
  timestamps) - see validate.py. This file does not re-check that.
- Rows are in chronological order (oldest first). Every formula below
  is recursive and directional - running it on unsorted data produces
  numbers that look valid but are meaningless.
- 'close' is the price series used for all indicators here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """
    Calculate the Exponential Moving Average (EMA) of a series.

    The formula
    -----------
        EMA[t] = price[t] * alpha + EMA[t-1] * (1 - alpha)
        alpha = 2 / (span + 1)
    """
    return series.ewm(span=span, adjust=False).mean()


def calculate_macd_line(close: pd.Series, fast_period: int = 12, slow_period: int = 26) -> pd.Series:
    """
    Calculate the MACD line: the difference between a fast and slow EMA.

    The formula
    -----------
        Fast EMA = EMA(close, fast_period)
        Slow EMA = EMA(close, slow_period)
        MACD Line = Fast EMA - Slow EMA
    """
    fast_ema = calculate_ema(close, span=fast_period)
    slow_ema = calculate_ema(close, span=slow_period)
    return fast_ema - slow_ema


def calculate_macd_signal_line(macd_line: pd.Series, signal_period: int = 9) -> pd.Series:
    """
    Calculate the MACD signal line as an EMA of the MACD series.
    """
    return calculate_ema(macd_line, span=signal_period)


def add_macd_indicators(
    df: pd.DataFrame,
    price_col: str = "close",
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    Add MACD indicator columns to a copy of `df`.

    The added columns are:
    - macd_fast_ema
    - macd_slow_ema
    - macd
    - macd_signal
    - macd_histogram
    """
    result = df.copy()
    close = result[price_col]

    result["macd_fast_ema"] = calculate_ema(close, span=fast_period)
    result["macd_slow_ema"] = calculate_ema(close, span=slow_period)
    
    result["macd"] = result["macd_fast_ema"] - result["macd_slow_ema"]
    result["macd_signal"] = calculate_macd_signal_line(result["macd"], signal_period=signal_period)
    result["macd_histogram"] = result["macd"] - result["macd_signal"]

    return result


if __name__ == "__main__":
    # A tiny, hand-checkable example: a flat (constant) price series.
    # If price never changes, the fast and slow EMAs converge, and the 
    # MACD line should settle at exactly zero.
    flat_prices = pd.DataFrame({"close": [1.1000] * 40})
    flat_result = add_macd_indicators(flat_prices)
    print("Flat price sanity check (last row should be exactly 0 for macd/signal/histogram):")
    print(flat_result.tail(1)[["close", "macd_fast_ema", "macd_slow_ema", "macd", "macd_signal", "macd_histogram"]])

    # A small trending example, so you can see MACD respond as
    # momentum builds, and confirm the shapes/columns look right.
    trending_prices = pd.DataFrame({"close": [1.1000 + i * 0.0010 for i in range(40)]})
    trending_result = add_macd_indicators(trending_prices)
    print("\nTrending price example (last 3 rows):")
    print(trending_result.tail(3)[["close", "macd", "macd_signal", "macd_histogram"]])