"""
export.py
=========

Purpose
-------
Convert the completed-trade log produced by execution.py into a CSV
that YOUR specific Quant Analyzer Free 4.9.4 installation actually
reads correctly - matching its auto-detected "General CSV Format #3"
positional template exactly, field for field.

Why this file targets a positional template instead of a named schema
-----------------------------------------------------------------------
Two earlier versions of this file targeted named column schemas (a
custom General CSV mapping, then an MT4-report-shaped file) and both
were confirmed, empirically, to load into Quant Analyzer with every
money field reading as $0. Inspecting the loaded report's own Settings
tab revealed why: Quant Analyzer's auto-detection skips row 1
(SkipRow=1) and applies a FIXED, built-in 15-field template
("Format #3") by POSITION, regardless of what our header text says.
Our header calling column 13 "Swap" doesn't matter - Quant Analyzer
still reads whatever is physically sitting in column 13 and treats it
as "PL". Fighting that with better header names would never have
worked; matching the positions themselves does.

Format #3's exact template (confirmed from the loaded report's own
Settings tab):
    1 Ticket, 2 OpenTime, 3 Action, 4 Size, 5 Symbol, 6 OpenPrice,
    7 Unused, 8 Unused, 9 CloseTime, 10 ClosePrice, 11 Unused,
    12 Unused, 13 PL, 14 Unused, 15 Unused

Responsibilities
-----------------
This file is ONLY responsible for formatting and writing. It
deliberately does NOT calculate win rate, Sharpe ratio, drawdown, or
any other statistic - that's Quant Analyzer's job, once it can
actually read the numbers.

Inputs
------
The DataFrame returned by execution.run_backtest():
    entry_time, exit_time, entry_price, exit_price, direction,
    lot_size, profit, money_profit, commission, swap, exit_reason

Outputs
-------
A 15-column CSV where columns 1,2,3,4,5,6,9,10,13 carry real data at
the EXACT positions Format #3 reads, and columns 7,8,11,12,14,15 (all
"Unused" to Quant Analyzer) still carry real, human-readable values -
nothing is thrown away, it's just not in a position Quant Analyzer's
statistics engine will pick up.

WHY position 13 (PL) holds NET profit, not raw money_profit
-----------------------------------------------------------------
Format #3 has no dedicated slot for Commission or Swap at all - both
map to "Unused" positions. Writing money_profit alone at position 13
would make Quant Analyzer's statistics silently ignore real trading
costs, overstating performance. Folding commission + swap into the
number that Quant Analyzer actually reads as PL means its stats
reflect the true net result, even though Commission/Swap can't be
broken out as separate tracked columns under this template. The raw,
pre-cost money_profit and the swap figure are still written to columns
14 and 15 respectively, so nothing is lost from the file itself - a
human opening it directly can still see the breakdown.

Assumptions
-----------
- Quant Analyzer's auto-detection continues to select "Format #3" for
  a file shaped like this one - confirmed for the 14-column version of
  this export; expected, but not yet independently reconfirmed, for
  this exact 15-column version. Verify this by loading the new export
  and checking the Settings tab still reports Format #3's mapping
  before trusting the results.
- `direction` values are exactly LONG_SIGNAL/SHORT_SIGNAL from
  signals.py, mapped to "buy"/"sell".
- `entry_time`/`exit_time` are strings already in
  "YYYY-MM-DD HH:MM:SS" format, matching the raw data's time_utc
  column.

Possible edge cases
--------------------
- Zero completed trades: still writes a valid CSV containing only the
  header row.
- Floating point noise in prices/profit: rounded before writing (see
  export_trades_to_csv's price_decimals/money_decimals parameters).

Future improvements
--------------------
- If Quant Analyzer's Tools menu turns out to expose a proper CSV
  import-format editor (rather than requiring a hand-edited .ini),
  switch back to a named-column export and fix Format #3's mapping
  directly instead of conforming to its quirk - more transparent long
  term, since Commission/Swap could then be tracked as their own
  columns again.
"""

from __future__ import annotations

import pandas as pd

# WHY a flat import, not a relative one: this file also runs standalone
# via its own __main__ block, and a relative import breaks the moment a
# file is run directly as a script. main.py adds src/ to sys.path
# before importing anything, which makes this flat import resolve
# correctly in both contexts.
from signals import LONG_SIGNAL, SHORT_SIGNAL

# The exact 15 column labels, in the exact order Quant Analyzer's
# Format #3 expects them BY POSITION. The text itself is never read by
# Quant Analyzer (row 1 is skipped), but using Format #3's own field
# names at the positions it actually reads (Ticket, OpenTime, Action,
# Size, Symbol, OpenPrice, CloseTime, ClosePrice, PL) makes the file
# self-documenting for any human who opens it - and harmless "Unused"
# positions instead carry real, useful labels for the data we've
# preserved there.
QA_FORMAT3_COLUMNS = [
    "Ticket",
    "Open Time",
    "Action",
    "Size",
    "Symbol",
    "Open Price",
    "Stop Loss",
    "Take Profit",
    "Close Time",
    "Close Price",
    "Commission",
    "Taxes",
    "PL",
    "Raw Money Profit",
    "Swap",
]

# WHY "buy"/"sell": confirmed correct in the previous export - Quant
# Analyzer's trade list displayed these as "Buy"/"Sell" correctly, so
# whatever field this lands in (Action) clearly does recognize this
# vocabulary.
TYPE_LABELS = {
    LONG_SIGNAL: "buy",
    SHORT_SIGNAL: "sell",
}

REQUIRED_TRADE_COLUMNS = [
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "direction",
    "lot_size",
    "money_profit",
    "commission",
    "swap",
]


def format_timestamp(value) -> str:
    """
    Normalize one timestamp value to "YYYY-MM-DD HH:MM:SS".

    WHY this function exists at all
    -----------------------------------
    `entry_time`/`exit_time` might already be plain strings (true today
    - nothing upstream parses time_utc into a real datetime type), or
    they might be datetime objects if a future pipeline change starts
    parsing them. Formatting explicitly here means the output is
    identical either way.

    Parameters
    ----------
    value : str or datetime-like

    Returns
    -------
    str
        Formatted as "YYYY-MM-DD HH:MM:SS".
    """
    if isinstance(value, str):
        return value
    return value.strftime("%Y-%m-%d %H:%M:%S")


def build_qa_format3_dataframe(trades: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Reshape a completed-trade log into Quant Analyzer Format #3's exact
    15-column positional layout.

    Parameters
    ----------
    trades : pd.DataFrame
        Output of execution.run_backtest().
    symbol : str
        The traded instrument, e.g. "EURUSD".

    Returns
    -------
    pd.DataFrame
        Columns exactly matching QA_FORMAT3_COLUMNS, in that order.

    Raises
    ------
    ValueError
        If `trades` is missing any expected column.
    """
    missing = [c for c in REQUIRED_TRADE_COLUMNS if c not in trades.columns]
    if missing:
        raise ValueError(
            f"build_qa_format3_dataframe() is missing column(s) {missing} - "
            "did you run execution.run_backtest() first?"
        )

    output = pd.DataFrame()

    output["Ticket"] = range(1, len(trades) + 1)
    output["Open Time"] = trades["entry_time"].apply(format_timestamp)
    output["Action"] = trades["direction"].map(TYPE_LABELS)
    output["Size"] = trades["lot_size"]
    output["Symbol"] = symbol
    output["Open Price"] = trades["entry_price"]
    # WHY 0.0: H001 has no fixed stop-loss/take-profit (see
    # execution.py's module docstring) - these two positions are
    # "Unused" to Quant Analyzer regardless of their content, so 0.0
    # here is accurate, not a placeholder standing in for something
    # missing.
    output["Stop Loss"] = 0.0
    output["Take Profit"] = 0.0
    output["Close Time"] = trades["exit_time"].apply(format_timestamp)
    output["Close Price"] = trades["exit_price"]
    output["Commission"] = trades["commission"]
    output["Taxes"] = 0.0
    # WHY this is money_profit + commission + swap, not money_profit
    # alone: position 13 is the ONE field Quant Analyzer's Format #3
    # actually reads as PL - see the module docstring's explanation of
    # why costs are folded in here rather than lost.
    output["PL"] = trades["money_profit"] + trades["commission"] + trades["swap"]
    output["Raw Money Profit"] = trades["money_profit"]
    output["Swap"] = trades["swap"]

    return output[QA_FORMAT3_COLUMNS]


def export_trades_to_csv(
    trades: pd.DataFrame,
    output_path: str,
    symbol: str = "EURUSD",
    price_decimals: int = 5,
    money_decimals: int = 2,
) -> pd.DataFrame:
    """
    Write a completed-trade log to a CSV matching Quant Analyzer's
    Format #3 positional template.

    Parameters
    ----------
    trades : pd.DataFrame
        Output of execution.run_backtest().
    output_path : str
        Where to write the CSV.
    symbol : str, default "EURUSD"
    price_decimals : int, default 5
        Decimal places for price-like columns (Open/Close/Stop/Target).
    money_decimals : int, default 2
        Decimal places for money-like columns (PL, Raw Money Profit,
        Commission, Swap, Taxes) - kept separate from price_decimals
        for the same reason explained in execution.calculate_money_profit()'s
        docstring: a price and a dollar amount don't share one natural
        rounding precision.

    Returns
    -------
    pd.DataFrame
        The exact data written to `output_path`.
    """
    formatted = build_qa_format3_dataframe(trades, symbol=symbol)

    price_columns = ["Open Price", "Close Price", "Stop Loss", "Take Profit"]
    formatted[price_columns] = formatted[price_columns].round(price_decimals)
    money_columns = ["Commission", "Taxes", "PL", "Raw Money Profit", "Swap"]
    formatted[money_columns] = formatted[money_columns].round(money_decimals)

    # WHY index=False: pandas' default row-number index would be
    # written as an extra, unlabeled first column, shifting every
    # other column one position to the right - which, given this whole
    # file exists to get positions exactly right, would be an
    # especially self-defeating mistake to make here.
    formatted.to_csv(output_path, index=False, lineterminator="\r\n")

    return formatted


def verify_column_count(output_path: str, expected_count: int = 15) -> bool:
    """
    Check that an exported CSV's header has exactly the expected number
    of columns.

    WHY this check, and not a header-text comparison against a
    reference file (as an earlier version of this file did)
    ------------------------------------------------------------------
    We now know Quant Analyzer's auto-detection doesn't read header
    text at all - it applies a fixed template by column position. The
    thing that can actually break silently is the COLUMN COUNT drifting
    away from what that template expects (e.g. someone removes a column
    from QA_FORMAT3_COLUMNS without realizing position 13 would then
    shift). Checking the count directly tests the assumption that
    actually matters now.

    Parameters
    ----------
    output_path : str
        Path to a CSV written by export_trades_to_csv().
    expected_count : int, default 15
        The column count Quant Analyzer's Format #3 expects.

    Returns
    -------
    bool
        True if the header has exactly `expected_count` columns.
    """
    with open(output_path, "r", encoding="utf-8") as output_file:
        header = output_file.readline().strip()
    return len(header.split(",")) == expected_count


if __name__ == "__main__":
    # The same two trades produced by execution.py's own stop-and-
    # reverse demo, so you can see exactly how a completed trade log
    # becomes a Format #3 row - PL at position 13 now correctly holds
    # the real net result (money_profit + commission + swap).
    example_trades = pd.DataFrame(
        {
            "entry_time": ["2024-01-01 02:00:00", "2024-01-01 04:00:00"],
            "exit_time": ["2024-01-01 04:00:00", "2024-01-01 06:00:00"],
            "entry_price": [1.1000, 1.1050],
            "exit_price": [1.1050, 1.1020],
            "direction": [LONG_SIGNAL, SHORT_SIGNAL],
            "lot_size": [1.0, 1.0],
            "profit": [0.0050, 0.0030],
            "money_profit": [500.0, 300.0],
            "commission": [-7.0, -7.0],
            "swap": [0.0, 0.0],
            "exit_reason": ["reversal", "reversal"],
        }
    )

    result = export_trades_to_csv(example_trades, "quant_analyzer_format3_demo.csv")
    print(result.to_string(index=False))
    print("\nColumn count check:", verify_column_count("quant_analyzer_format3_demo.csv"))