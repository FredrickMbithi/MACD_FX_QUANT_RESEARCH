"""
indicators.py
=============

Purpose
-------
Calculate the technical indicators this strategy needs - EMA and TRIX -
from a validated OHLC DataFrame.

This file implements the *math*, not the *decisions*. It answers
"what is the TRIX value on this bar?", never "should I buy on this
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
    trix_ema1, trix_ema2, trix_ema3, trix, trix_signal

Assumptions
-----------
- Input data is already clean (no NaNs, no duplicate/out-of-order
  timestamps) - see validate.py. This file does not re-check that.
- Rows are in chronological order (oldest first). Every formula below
  is recursive and directional - running it on unsorted data produces
  numbers that look valid but are meaningless.
- 'close' is the price series used for all indicators here (TRIX is
    conventionally calculated on close, not open/high/low).

Possible edge cases
--------------------
- A flat/constant price series: TRIX should converge toward zero once
    the triple-smoothed EMA has stabilized.

Future improvements
--------------------
- Optional minimum-history guard that returns NaN for rows before the
    triple-smoothed EMA has accumulated enough history.
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

    `alpha` is called the "smoothing factor". A small span (e.g. 12)
    gives a large alpha, which weights recent prices heavily and makes
    the EMA react quickly. A large span (e.g. 26) gives a small alpha,
    which weights history more evenly and makes the EMA react slowly.

    WHY EMA is recursive (unlike a Simple Moving Average)
    -------------------------------------------------------
    An SMA of period N is just `mean(last N prices)` - a fresh, static
    calculation on a fixed window. It has no memory of anything outside
    that window.

    An EMA is defined in terms of *itself*: today's EMA depends on
    yesterday's EMA, which depends on the day before that, all the way
    back to the first price in the series. There is no fixed window -
    every past price still has *some* influence on today's value,
    just an exponentially shrinking one. That's precisely why it's
    called "exponential": the weight given to a price N bars ago is
    proportional to (1 - alpha)^N, which decays exponentially as N
    grows. This recursion is also why you cannot compute EMA[t] in
    isolation - you always need EMA[t-1] first, which is what makes it
    a genuinely sequential (not just windowed) calculation.

    A note on `adjust=False`
    --------------------------
    pandas' `.ewm()` supports two modes. `adjust=True` (the default)
    computes a weighted average over *all* prices seen so far, then
    renormalizes - mathematically different from, though similar to,
    the textbook recursive formula. `adjust=False` implements the exact
    recursive formula above, bar by bar. We use `adjust=False` because
    it matches how EMA is computed on trading platforms like MT5,
    TradingView, and cTrader - important for you since your Pine Script
    and cTrader C# strategies should produce comparable numbers to this
    Python implementation. (Verified independently: a hand-rolled loop
    implementing the recursive formula above matches
    series.ewm(span=span, adjust=False).mean() to within floating-point
    precision, max abs diff 0.0 on a test series.)

    Parameters
    ----------
    series : pd.Series
        A price column (typically 'close'), in chronological order
        (oldest first). Order matters - see "Assumptions" in the module
        docstring.
    span : int
        The lookback period. Must be a positive integer.

    Returns
    -------
    pd.Series
        The EMA, same length and index as `series`. The first value
        equals the first price (there's no history yet to smooth with).
    """
    return series.ewm(span=span, adjust=False).mean()


def calculate_trix_line(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate the TRIX line: the percent change of a triple-smoothed EMA.

    The formula
    -----------
        EMA1 = EMA(close, period)
        EMA2 = EMA(EMA1, period)
        EMA3 = EMA(EMA2, period)
        TRIX[t] = (EMA3[t] - EMA3[t-1]) / EMA3[t-1] * 100

    TRIX is designed to filter more noise than MACD by applying three
    EMA passes before measuring momentum. The output is a percentage
    change, so it remains comparable across instruments.
    """
    ema1 = calculate_ema(close, span=period)
    ema2 = calculate_ema(ema1, span=period)
    ema3 = calculate_ema(ema2, span=period)

    previous_ema3 = ema3.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        trix = (ema3 - previous_ema3) / previous_ema3 * 100

    return trix.replace([np.inf, -np.inf], np.nan)


def calculate_trix_signal_line(trix_line: pd.Series, signal_span: int = 9) -> pd.Series:
    """
    Calculate a TRIX signal line as an EMA of the TRIX series.

    This is optional for earlier entries; the core TRIX signal is still
    the zero-line cross.
    """
    return calculate_ema(trix_line, span=signal_span)


def add_trix_indicators(
    df: pd.DataFrame,
    price_col: str = "close",
    period: int = 14,
    signal_span: int = 9,
) -> pd.DataFrame:
    """
    Add TRIX indicator columns to a copy of `df`.

    The added columns are:
    - trix_ema1
    - trix_ema2
    - trix_ema3
    - trix
    - trix_signal
    """
    result = df.copy()
    close = result[price_col]

    result["trix_ema1"] = calculate_ema(close, span=period)
    result["trix_ema2"] = calculate_ema(result["trix_ema1"], span=period)
    result["trix_ema3"] = calculate_ema(result["trix_ema2"], span=period)
    result["trix"] = calculate_trix_line(close, period=period)
    result["trix_signal"] = calculate_trix_signal_line(result["trix"], signal_span=signal_span)

    return result


if __name__ == "__main__":
    # A tiny, hand-checkable example: a flat (constant) price series.
    # WHY this is a good sanity check: if price never changes, TRIX
    # should settle near zero once the triple-smoothed EMA stabilizes.
    flat_prices = pd.DataFrame({"close": [1.1000] * 40})
    flat_result = add_trix_indicators(flat_prices)
    print("Flat price sanity check (last row should be ~0 for trix/signal):")
    print(flat_result.tail(1)[["close", "trix_ema1", "trix_ema2", "trix_ema3", "trix", "trix_signal"]])

    # A small trending example, so you can see TRIX respond as
    # momentum builds, and confirm the shapes/columns look right.
    trending_prices = pd.DataFrame({"close": [1.1000 + i * 0.0010 for i in range(40)]})
    trending_result = add_trix_indicators(trending_prices)
    print("\nTrending price example (last 3 rows):")
    print(trending_result.tail(3)[["close", "trix", "trix_signal"]])