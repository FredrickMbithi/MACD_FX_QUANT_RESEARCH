"""
execution.py
=============

Purpose
-------
Simulate how the signals produced by signals.py would actually have
been traded historically, using a FIXED stop-loss/take-profit system:
each signal opens one position, which is closed the instant price
touches its stop-loss or take-profit level - whichever comes first -
with no reaction to any signal that arrives while that position is
still open.

This is a reversion. The previous version of this file replaced fixed
exits with a "stop-and-reverse" system (close only on the opposite
signal, no price-level exit at all). That version was useful for
testing H001's raw signal quality in isolation, but the actual system
being deployed uses fixed exits - so this file reverts to that shape.
The one thing that changes from the ORIGINAL fixed-exit version (before
the stop-and-reverse detour) is the exit distances themselves: 30 pip
stop / 75 pip target (2.5R), replacing the old 20/40 pip placeholder
values, now that a parameter sweep has identified 30/75 as the best
point within the smooth, trustworthy region of the results surface (by
contrast, 30/60 - exactly 2.0R at that same stop - was only breakeven).

Responsibilities
-----------------
This file is ONLY responsible for turning a stream of LONG/SHORT
signals plus OHLC price action into a log of completed trades. It
deliberately does NOT:
- decide entry signal logic (that's signals.py's crossover detection -
  this file only reacts to it)
- calculate statistics like win rate or Sharpe ratio (that's Quant
  Analyzer's job, once export.py hands it the trade log)

Inputs
------
A DataFrame that already has a 'signal' column (from
signals.generate_signals()), plus open/high/low/timestamp columns, in
chronological order.

WHY high/low ARE needed here (unlike the stop-and-reverse version)
--------------------------------------------------------------------
The stop-and-reverse version only ever reacted to signals, so it never
needed to check price between bars - it only ever read each bar's
open. A fixed stop-loss/take-profit needs the OPPOSITE: every bar a
position is open, its high and low must be checked for an intrabar
touch of either exit level, since the exit can happen on ANY bar, not
just one where a new signal shows up.

Outputs
-------
A DataFrame with one row per COMPLETED trade:
    entry_time, exit_time, entry_price, exit_price, direction,
    lot_size, profit, money_profit, commission, swap, exit_reason
`profit` is the raw price difference (currency-agnostic); `money_profit`
converts that into account currency - see calculate_money_profit()'s
docstring. `commission`/`swap` are separate cost/credit line items,
kept apart from `money_profit` rather than netted into it, matching
how real broker reports separate these. This shape, and the
money_profit/commission/swap calculation logic behind it, is unchanged
from the stop-and-reverse version - only the exit mechanism (STEP 2 in
run_backtest) actually changed.

Assumptions
-----------
- The system is NOT "always in the market" - after an exit, the
  account sits flat until the next fresh signal arrives, exactly like
  any fixed-SL/TP system waiting for a new setup.
- A signal arriving while a position is already open is IGNORED
  entirely (not queued, not remembered) - only the stop or the target
  can end a trade once one is open. This is the one behavioral
  difference from the stop-and-reverse version, where an opposite
  signal was the only exit trigger and always acted on immediately.
- Entries fill at the exact open price of the bar after the signal
  bar - no slippage, no spread. Same next-bar-execution reasoning as
  before: a crossover signal on bar i is only knowable once bar i has
  fully closed.
- Stop-loss and take-profit levels are set once, at entry, from a
  fixed pip distance (DEFAULT_STOP_LOSS_PIPS / DEFAULT_TAKE_PROFIT_PIPS)
  and never move afterward - no breakeven-stop or trailing-stop logic.
- If a single bar's high/low range touches BOTH the stop and the
  target (a wide bar, or a gap), the STOP is assumed to trigger first.
  This is a deliberate conservative assumption, not a claim about what
  actually happened intrabar - real tick-by-tick order is unknowable
  from OHLC bars alone, and assuming the worse outcome avoids
  overstating performance.
- PIP_SIZE (0.0001) assumes a non-JPY pair quoted to 4/5 decimal places,
  matching calculate_money_profit()'s existing EURUSD assumption. A
  JPY-quoted pair would need PIP_SIZE = 0.01 instead.
- money_profit assumes the account currency matches the pair's quote
  currency (true for a USD account trading EURUSD) - see
  calculate_money_profit()'s docstring.
- commission_per_lot and swap rates default to illustrative,
  realistic-shaped placeholder values (see DEFAULT_COMMISSION_PER_LOT,
  DEFAULT_SWAP_LONG_PER_NIGHT, DEFAULT_SWAP_SHORT_PER_NIGHT), not a
  real broker's rate sheet.
- Swap is approximated using calendar days crossed between entry and
  exit, not a broker's exact rollover cutoff time or the MT4
  triple-swap-Wednesday convention.

Possible edge cases
--------------------
- A signal on the very last bar of the dataset can never be acted on -
  there's no next bar to enter at. Silently dropped; a real limit of
  finite historical data, not a bug.
- A position still open when the data runs out (never touched its
  stop or target) is NOT included in the returned trade log - it has
  no exit yet.
- A signal that arrives on the exact same bar a position's stop/target
  is hit: the exit is processed first (STEP 2), then the signal is
  free to be queued for entry at the NEXT bar's open (STEP 3), since
  the account is flat again by that point in the same bar.

Future improvements
--------------------
- Model spread and slippage instead of assuming perfect fills.
- Support variable or compounding lot sizing instead of one fixed
  `lot_size` for the whole backtest.
- Support JPY-quoted pairs and cross-currency accounts in
  calculate_money_profit() and via a configurable PIP_SIZE.
- Model swap using a broker's actual rollover cutoff time and the MT4
  triple-swap-Wednesday convention, instead of a calendar-day
  approximation.
- Add breakeven-stop or trailing-stop logic once the fixed-distance
  version has a solid, trusted baseline to compare against.
"""

from __future__ import annotations

import pandas as pd

# WHY import these from signals.py instead of redefining "LONG"/"SHORT"
# strings here: these two constants are created in signals.py and read
# here. Importing from one shared source means if signals.py ever
# renamed LONG_SIGNAL's value, this file could not silently fall out of
# sync with it - redefining the same string in two files could not
# guarantee that.
#
# WHY a flat import ("from signals import"), not a relative one: this
# file also runs standalone via its own __main__ block below, and a
# relative import breaks the moment a file is run directly as a script
# rather than imported as part of a package. main.py adds src/ to
# sys.path before importing anything, which makes this flat import
# resolve correctly in both contexts.
from signals import LONG_SIGNAL, SHORT_SIGNAL

# WHY these are constants: execution.py generates them, and export.py
# will need to read them later - one shared spelling, used everywhere.
EXIT_REASON_STOP_LOSS = "stop_loss"
EXIT_REASON_TAKE_PROFIT = "take_profit"

# WHY this specific number: a "standard lot" in forex is, by market
# convention, 100,000 units of the base currency (EUR, in EURUSD). This
# constant is what turns a raw price move into a real money amount -
# see calculate_money_profit() below for the full explanation and its
# assumptions.
STANDARD_LOT_UNITS = 100_000

# WHY 0.0001: the size of one pip for a non-JPY pair quoted to 4/5
# decimal places (e.g. EURUSD). A JPY-quoted pair (e.g. USDJPY, quoted
# to 2/3 decimals) would need 0.01 instead - see the module docstring's
# assumptions and "Future improvements".
PIP_SIZE = 0.0001

# WHY 30/75, not the original 20/40 placeholders
# ------------------------------------------------------
# A parameter sweep over stop/target combinations identified 30 pip
# stop / 75 pip target (2.5R) as the best-performing point within the
# smooth, trustworthy region of the results surface - the region where
# neighboring parameter combinations behaved consistently rather than
# spiking on noise. At that same 30 pip stop, 2.0R (a 60 pip target)
# was only breakeven, which is why 2.5R was chosen over the rounder
# 2.0R figure. These are still just the current best estimate from
# historical data, not a guarantee of future performance - rerun the
# sweep if the underlying data or instrument changes.
DEFAULT_STOP_LOSS_PIPS = 30
DEFAULT_TAKE_PROFIT_PIPS = 75

# WHY these are defaults, not "the truth": every broker sets its own
# commission and swap rates, and swap rates change over time with
# central bank interest rates. These numbers are realistic *shapes* -
# not real numbers to trust for live trading. Override them from
# config.yaml once you have your actual broker's rate sheet.
DEFAULT_COMMISSION_PER_LOT = 7.0  # USD, round-turn, per standard lot
DEFAULT_SWAP_LONG_PER_NIGHT = -2.5  # USD, per standard lot, per night held
DEFAULT_SWAP_SHORT_PER_NIGHT = 0.3  # USD, per standard lot, per night held

TRADE_LOG_COLUMNS = [
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "direction",
    "lot_size",
    "profit",
    "money_profit",
    "commission",
    "swap",
    "exit_reason",
]


def calculate_money_profit(
    price_profit: float,
    lot_size: float,
    contract_size: float = STANDARD_LOT_UNITS,
) -> float:
    """
    Convert a raw price-difference profit into a real account-currency
    amount.

    The formula
    -----------
        money_profit = price_profit * lot_size * contract_size

    WHY this formula works for EURUSD specifically
    ---------------------------------------------------
    One standard lot of EURUSD is a position of 100,000 EUR. If price
    moves by `price_profit` (e.g. 0.0050), the change in value of that
    100,000 EUR position is 0.0050 * 100,000 = 500 - and because
    EURUSD's quote currency is USD, that 500 is already in USD. This is
    the same "$10 per pip per standard lot" convention quoted
    everywhere in FX (0.0001 price move * 100,000 units = $10).

    Two assumptions this simple formula depends on
    ---------------------------------------------------
    1. The account currency matches the pair's QUOTE currency (the
       second symbol - USD in EURUSD). This holds for a USD account
       trading EURUSD (your current setup), but would be wrong for,
       say, a USD account trading GBPJPY.
    2. `lot_size` is fixed for the whole backtest - there's no
       compounding or equity-scaled position sizing here.

    Parameters
    ----------
    price_profit : float
        The raw price-difference profit (entry vs exit price), already
        signed correctly for direction.
    lot_size : float
        Position size in standard lots (1.0 = 100,000 units).
    contract_size : float, default STANDARD_LOT_UNITS
        Units of base currency per 1.0 lot.

    Returns
    -------
    float
        Profit/loss in account currency.
    """
    return price_profit * lot_size * contract_size


def calculate_commission(lot_size: float, commission_per_lot: float = DEFAULT_COMMISSION_PER_LOT) -> float:
    """
    Calculate the round-turn commission cost for one trade.

    WHY this is returned as a NEGATIVE number
    ----------------------------------------------
    Commission is a cost, not income - it reduces the account balance.
    Real MT4/broker reports write commission as a negative figure (so
    that "Profit + Commission + Swap" gives the true net result by
    simple addition). Matching that convention here means export.py can
    hand this number straight to Quant Analyzer without any
    reinterpretation.

    Parameters
    ----------
    lot_size : float
        Position size in standard lots.
    commission_per_lot : float, default DEFAULT_COMMISSION_PER_LOT
        Round-turn commission per standard lot, in account currency.

    Returns
    -------
    float
        The commission cost for this trade (negative).
    """
    return -commission_per_lot * lot_size


def calculate_nights_held(entry_time, exit_time) -> int:
    """
    Count how many calendar-day boundaries a trade was open across -
    used as a simple stand-in for "how many overnight rollovers did
    this position sit through".

    WHY calendar days, not the broker's exact rollover cutoff time
    --------------------------------------------------------------------
    Real swap is charged at a specific rollover time each day, and MT4
    famously charges triple swap on Wednesdays to cover the weekend.
    Modeling that precisely needs a broker-specific rollover calendar
    this pipeline doesn't have. Counting calendar days crossed is a
    deliberately simple approximation - close enough to produce
    realistic-shaped swap costs, not a precise replica of any specific
    broker's rules.

    Parameters
    ----------
    entry_time, exit_time : str or datetime-like

    Returns
    -------
    int
        Number of calendar-day boundaries crossed (0 if the trade
        opened and closed on the same calendar day).
    """
    entry_date = pd.Timestamp(entry_time).normalize()
    exit_date = pd.Timestamp(exit_time).normalize()
    return (exit_date - entry_date).days


def calculate_swap(
    entry_time,
    exit_time,
    direction: str,
    lot_size: float,
    swap_long_per_night: float = DEFAULT_SWAP_LONG_PER_NIGHT,
    swap_short_per_night: float = DEFAULT_SWAP_SHORT_PER_NIGHT,
) -> float:
    """
    Calculate the total swap (overnight financing) cost or credit for
    one trade.

    The logic
    ---------
    Swap is charged (or paid) once per night a position stays open, and
    the rate differs for long vs short positions, reflecting the real
    interest-rate differential between the two currencies in the pair.
    A trade opened and closed within the same calendar day incurs no
    swap at all.

    Parameters
    ----------
    entry_time, exit_time : str or datetime-like
    direction : str
        LONG_SIGNAL or SHORT_SIGNAL.
    lot_size : float
        Position size in standard lots - swap scales with size, the
        same way commission does.
    swap_long_per_night : float, default DEFAULT_SWAP_LONG_PER_NIGHT
    swap_short_per_night : float, default DEFAULT_SWAP_SHORT_PER_NIGHT

    Returns
    -------
    float
        Total swap for the trade (can be positive or negative,
        depending on direction and the configured rates).
    """
    nights_held = calculate_nights_held(entry_time, exit_time)
    rate_per_night = swap_long_per_night if direction == LONG_SIGNAL else swap_short_per_night
    return rate_per_night * nights_held * lot_size


def calculate_exit_levels(
    entry_price: float,
    direction: str,
    stop_loss_pips: float,
    take_profit_pips: float,
    pip_size: float = PIP_SIZE,
) -> tuple[float, float]:
    """
    Compute the fixed stop-loss and take-profit price levels for a new
    position, set once at entry and never moved afterward.

    Parameters
    ----------
    entry_price : float
    direction : str
        LONG_SIGNAL or SHORT_SIGNAL.
    stop_loss_pips, take_profit_pips : float
        Distance from entry, in pips.
    pip_size : float, default PIP_SIZE

    Returns
    -------
    tuple[float, float]
        (stop_loss_price, take_profit_price).
    """
    stop_distance = stop_loss_pips * pip_size
    target_distance = take_profit_pips * pip_size

    if direction == LONG_SIGNAL:
        return entry_price - stop_distance, entry_price + target_distance
    return entry_price + stop_distance, entry_price - target_distance


def check_exit_hit(
    direction: str,
    high: float,
    low: float,
    stop_loss_price: float,
    take_profit_price: float,
) -> str | None:
    """
    Check whether one bar's high/low range touched the stop-loss or
    take-profit level for an open position.

    WHY the stop wins when both are touched in the same bar
    -----------------------------------------------------------
    OHLC bars only record the bar's open/high/low/close - not the
    actual order in which prices were touched intrabar. When a single
    bar's range is wide enough to reach both levels, which one the
    market hit FIRST is genuinely unknowable from this data. Assuming
    the stop hit first is the conservative choice: it never overstates
    performance by assuming the friendlier outcome for an ambiguous
    bar.

    Parameters
    ----------
    direction : str
        LONG_SIGNAL or SHORT_SIGNAL.
    high, low : float
        The current bar's high and low.
    stop_loss_price, take_profit_price : float

    Returns
    -------
    str or None
        EXIT_REASON_STOP_LOSS, EXIT_REASON_TAKE_PROFIT, or None if
        neither level was touched this bar.
    """
    if direction == LONG_SIGNAL:
        hit_stop = low <= stop_loss_price
        hit_target = high >= take_profit_price
    else:
        hit_stop = high >= stop_loss_price
        hit_target = low <= take_profit_price

    if hit_stop:
        return EXIT_REASON_STOP_LOSS
    if hit_target:
        return EXIT_REASON_TAKE_PROFIT
    return None


def run_backtest(
    df: pd.DataFrame,
    starting_balance: float = 10_000.0,
    lot_size: float = 1.0,
    contract_size: float = STANDARD_LOT_UNITS,
    commission_per_lot: float = DEFAULT_COMMISSION_PER_LOT,
    swap_long_per_night: float = DEFAULT_SWAP_LONG_PER_NIGHT,
    swap_short_per_night: float = DEFAULT_SWAP_SHORT_PER_NIGHT,
    stop_loss_pips: float = DEFAULT_STOP_LOSS_PIPS,
    take_profit_pips: float = DEFAULT_TAKE_PROFIT_PIPS,
    daily_drawdown_percent: float = 0.0,
    pip_size: float = PIP_SIZE,
    timestamp_col: str = "time_utc",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    signal_col: str = "signal",
) -> pd.DataFrame:
    """
    Simulate a fixed stop-loss/take-profit strategy on every LONG/SHORT
    signal in `df`, and return a log of completed trades.

    The mechanic
    ------------
    A signal opens a position at the next bar's open. From that point
    on, EVERY bar's high/low is checked against the position's fixed
    stop-loss and take-profit levels (set once, at entry, from
    `stop_loss_pips`/`take_profit_pips`). The position closes the
    instant either level is touched - no other signal can end it, and
    any signal arriving while it's open is ignored. Once flat, the
    account waits for the next fresh signal.

    Why next-bar execution
    ----------------------
    A crossover signal on bar i is only knowable once bar i has fully
    closed, so the earliest a real order could be placed is at the
    NEXT bar's open. This applies only to ENTRIES here - exits happen
    the instant a level is touched, on whichever bar that is, since a
    resting stop/limit order at a broker triggers intrabar rather than
    waiting for the next bar.

    WHY this loops bar-by-bar instead of using vectorized pandas
    operations (unlike indicators.py and signals.py)
    ----------------------------------------------------------------
    Whether bar i should trigger an entry, an exit, or nothing depends
    on whether a position is currently open and, if so, its
    entry-derived stop/target levels - both of which depend on
    everything that happened on every earlier bar. That's inherent to
    simulating a *stateful* process, not a performance shortcut
    skipped.

    Parameters
    ----------
    df : pd.DataFrame
        Output of signals.generate_signals(). Must contain
        `timestamp_col`, `open_col`, `high_col`, `low_col`, and
        `signal_col`, in chronological order.
    lot_size : float, default 1.0
        Position size in standard lots, used to convert raw price
        profit into a real money amount - see calculate_money_profit().
    starting_balance : float, default 10_000.0
        Starting account balance used for daily-loss tracking.
    contract_size : float, default STANDARD_LOT_UNITS
        Units of base currency per 1.0 lot.
    commission_per_lot : float, default DEFAULT_COMMISSION_PER_LOT
        Round-turn commission per standard lot - see
        calculate_commission().
    swap_long_per_night, swap_short_per_night : float
        Per-night carrying cost/credit per standard lot, by direction -
        see calculate_swap().
    stop_loss_pips, take_profit_pips : float, default 30 / 75
        Fixed exit distances in pips - see calculate_exit_levels().
    daily_drawdown_percent : float, default 0.0
        Maximum allowed realized daily loss as a percentage of the
        account balance at the start of the day. When breached, no new
        trades are opened until the next day.
    pip_size : float, default PIP_SIZE
    timestamp_col, open_col, high_col, low_col, signal_col : str
        Column names, in case they differ from the defaults.

    Returns
    -------
    pd.DataFrame
        Columns: entry_time, exit_time, entry_price, exit_price,
        direction, lot_size, profit, money_profit, commission, swap,
        exit_reason - one row per COMPLETED trade. A trade still open
        when the data ends is not included (see "Possible edge cases"
        in the module docstring).

    Raises
    ------
    ValueError
        If any required column is missing - a clear, fail-fast check
        rather than a confusing AttributeError deep inside the loop.
    """
    required_columns = [timestamp_col, open_col, high_col, low_col, signal_col]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"run_backtest() is missing column(s) {missing} - "
            "did you run signals.generate_signals() first?"
        )

    completed_trades = []

    # State carried across iterations - this is exactly why this
    # function must be a loop, not a vectorized column operation.
    # `direction` doubles as the "are we in a position" flag: None
    # means flat, LONG_SIGNAL/SHORT_SIGNAL means a position is open.
    direction = None
    pending_signal = None
    entry_time = None
    entry_price = None
    stop_loss_price = None
    take_profit_price = None
    account_balance = starting_balance
    current_day = None
    day_start_balance = starting_balance
    daily_floor = starting_balance * (1 - daily_drawdown_percent / 100.0)
    daily_halted = False

    # WHY itertuples(index=False) instead of iterrows(): see
    # indicators.py/signals.py for the same note - it preserves each
    # column's real type and is noticeably faster for row-by-row loops.
    for row in df.itertuples(index=False):
        current_time = getattr(row, timestamp_col)
        current_open = getattr(row, open_col)
        current_high = getattr(row, high_col)
        current_low = getattr(row, low_col)
        current_signal = getattr(row, signal_col)

        row_day = pd.Timestamp(current_time).normalize()
        if current_day is None or row_day != current_day:
            current_day = row_day
            day_start_balance = account_balance
            daily_floor = day_start_balance * (1 - daily_drawdown_percent / 100.0)
            daily_halted = account_balance <= daily_floor

        # STEP 1: act on a signal queued from the PREVIOUS bar, at
        # THIS bar's open - the next-bar execution rule explained
        # above. Only ever queued while flat (see STEP 3), so this is
        # always a plain entry, never a reversal.
        if pending_signal is not None and not daily_halted:
            direction = pending_signal
            entry_time = current_time
            entry_price = current_open
            stop_loss_price, take_profit_price = calculate_exit_levels(
                entry_price, direction, stop_loss_pips, take_profit_pips, pip_size
            )
            pending_signal = None
        elif daily_halted:
            pending_signal = None

        # STEP 2: if a position is open (whether carried over from an
        # earlier bar or just opened above, on this same bar), check
        # THIS bar's high/low for a stop/target touch.
        if direction is not None:
            exit_reason = check_exit_hit(
                direction, current_high, current_low, stop_loss_price, take_profit_price
            )
            if exit_reason is not None:
                exit_price = stop_loss_price if exit_reason == EXIT_REASON_STOP_LOSS else take_profit_price
                profit = (
                    exit_price - entry_price
                    if direction == LONG_SIGNAL
                    else entry_price - exit_price
                )
                money_profit = calculate_money_profit(profit, lot_size, contract_size)
                commission = calculate_commission(lot_size, commission_per_lot)
                swap = calculate_swap(
                    entry_time,
                    current_time,
                    direction,
                    lot_size,
                    swap_long_per_night,
                    swap_short_per_night,
                )
                account_balance += money_profit + commission + swap
                completed_trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": current_time,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "direction": direction,
                        "lot_size": lot_size,
                        "profit": profit,
                        "money_profit": money_profit,
                        "commission": commission,
                        "swap": swap,
                        "exit_reason": exit_reason,
                    }
                )
                direction = None
                entry_time = None
                entry_price = None
                stop_loss_price = None
                take_profit_price = None

                if account_balance <= daily_floor:
                    daily_halted = True
                    pending_signal = None

        # STEP 3: queue a new signal for the NEXT bar's open, but ONLY
        # if currently flat. Unlike the stop-and-reverse version, a
        # signal arriving while a position is open is simply ignored -
        # only the stop or target can end an open trade.
        if not daily_halted and direction is None and current_signal in (LONG_SIGNAL, SHORT_SIGNAL):
            pending_signal = current_signal

    return pd.DataFrame(completed_trades, columns=TRADE_LOG_COLUMNS)


if __name__ == "__main__":
    # A small, hand-verifiable timeline: a LONG entry that gets stopped
    # out, then a SHORT entry that hits its target and stays flat
    # afterward (no further signal arrives).
    example = pd.DataFrame(
        {
            "time_utc": [f"2024-01-01 0{i}:00:00" for i in range(8)],
            "open":     [1.0980, 1.0990, 1.0995, 1.0965, 1.0960, 1.0940, 1.0920, 1.0915],
            "high":     [1.0985, 1.0998, 1.1000, 1.0995, 1.0962, 1.0945, 1.0925, 1.0918],
            "low":      [1.0975, 1.0988, 1.0955, 1.0960, 1.0885, 1.0935, 1.0912, 1.0860],
            "signal":   [None, LONG_SIGNAL, None, None, SHORT_SIGNAL, None, None, None],
        }
    )

    trades = run_backtest(example, lot_size=1.0)
    print(trades.to_string(index=False))