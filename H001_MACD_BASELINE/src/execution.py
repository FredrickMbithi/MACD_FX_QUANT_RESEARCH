"""
execution.py
"""
from __future__ import annotations
import pandas as pd
from signals import LONG_SIGNAL, SHORT_SIGNAL, BULL_TREND_REGIME, BEAR_TREND_REGIME

# Constants
STANDARD_LOT_UNITS = 100_000
PIP_SIZE = 0.0001
JPY_PIP_SIZE = 0.01

# Fixed Risk:Reward exits. TP and SL are both derived from the same
# catastrophic SL distance (in pips), scaled by risk_reward_ratio, so a
# single risk unit consistently prices both sides of every trade.
DEFAULT_CATASTROPHIC_SL_PIPS = 75.0
DEFAULT_RISK_REWARD_RATIO = 2.0
DEFAULT_COMMISSION_PER_LOT = 7.0
DEFAULT_SWAP_LONG_PER_NIGHT = -2.5
DEFAULT_SWAP_SHORT_PER_NIGHT = 0.3

TRADE_LOG_COLUMNS = [
    "entry_time", "exit_time", "entry_price", "exit_price", "direction",
    "lot_size", "profit", "money_profit", "commission", "swap", "exit_reason",
    "sl_price", "tp_price",
]

def get_pip_size(symbol: str) -> float:
    sym = symbol.upper()
    if "JPY" in sym or "XAU" in sym:
        return JPY_PIP_SIZE
    return PIP_SIZE

def calculate_money_profit(price_profit: float, lot_size: float, contract_size: float) -> float:
    return price_profit * lot_size * contract_size

def calculate_commission(lot_size: float, commission_per_lot: float) -> float:
    return -commission_per_lot * lot_size

def calculate_swap(entry_time, exit_time, direction, lot_size, swap_long, swap_short) -> float:
    nights = (pd.Timestamp(exit_time).normalize() - pd.Timestamp(entry_time).normalize()).days
    rate = swap_long if direction == LONG_SIGNAL else swap_short
    return rate * nights * lot_size

def run_backtest(df, starting_balance, lot_size, contract_size, commission_per_lot,
                 swap_long_per_night, swap_short_per_night, catastrophic_sl_pips,
                 daily_drawdown_percent, symbol, risk_reward_ratio=DEFAULT_RISK_REWARD_RATIO,
                 timestamp_col="time_utc", open_col="open", high_col="high", low_col="low",
                 signal_col="signal", regime_col="regime") -> pd.DataFrame:

    required_columns = [timestamp_col, open_col, high_col, low_col, signal_col, regime_col]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"run_backtest() is missing column(s) {missing} - "
            "did you run signals.generate_signals() first?"
        )

    resolved_pip_size = get_pip_size(symbol)
    completed_trades = []
    account_balance = starting_balance

    # State tracking
    direction = None
    entry_time = None
    entry_price = None
    sl_price = None
    tp_price = None

    pending_entry = None
    pending_exit = False
    exit_reason_queued = None

    daily_halted = False
    current_day = None
    day_start_balance = starting_balance
    daily_floor = None

    for row in df.itertuples(index=False):
        current_time = getattr(row, timestamp_col)
        current_open = getattr(row, open_col)
        current_high = getattr(row, high_col)
        current_low = getattr(row, low_col)
        current_signal = getattr(row, signal_col)
        current_regime = getattr(row, regime_col)

        # Reset the daily floor at the start of every new calendar day,
        # anchored to the balance the day actually started with - not to
        # the account's all-time starting balance.
        row_day = pd.Timestamp(current_time).normalize()
        if current_day is None or row_day != current_day:
            current_day = row_day
            day_start_balance = account_balance
            daily_floor = day_start_balance * (1 - daily_drawdown_percent / 100.0)

        if daily_drawdown_percent > 0 and account_balance <= daily_floor:
            daily_halted = True
        else:
            daily_halted = False

        # STEP 1: Execute Pending Logic Exits (from the previous bar's decision)
        if pending_exit and direction is not None:
            exit_price = current_open
            profit = (exit_price - entry_price) if direction == LONG_SIGNAL else (entry_price - exit_price)
            money_profit = calculate_money_profit(profit, lot_size, contract_size)
            comm = calculate_commission(lot_size, commission_per_lot)
            swap = calculate_swap(entry_time, current_time, direction, lot_size, swap_long_per_night, swap_short_per_night)
            account_balance += (money_profit + comm + swap)

            completed_trades.append({
                "entry_time": entry_time, "exit_time": current_time, "entry_price": entry_price,
                "exit_price": exit_price, "direction": direction, "lot_size": lot_size,
                "profit": profit, "money_profit": money_profit, "commission": comm,
                "swap": swap, "exit_reason": exit_reason_queued,
                "sl_price": sl_price, "tp_price": tp_price,
            })

            direction = None
            pending_exit = False
            exit_reason_queued = None

        # STEP 2: Execute Pending Entries
        if pending_entry is not None and direction is None and not daily_halted:
            direction = pending_entry
            entry_price = current_open
            entry_time = current_time

            # Set catastrophic failsafe SL and fixed-RR TP off the same risk
            # distance, so risk_reward_ratio directly controls the payoff
            # profile (e.g. 2.0 == TP is 2x further from entry than SL).
            sl_dist = catastrophic_sl_pips * resolved_pip_size
            reward_dist = sl_dist * risk_reward_ratio
            if direction == LONG_SIGNAL:
                sl_price = entry_price - sl_dist
                tp_price = entry_price + reward_dist
            else:
                sl_price = entry_price + sl_dist
                tp_price = entry_price - reward_dist
            pending_entry = None

        # STEP 3: Intrabar SL/TP Check
        if direction is not None:
            hit_sl = False
            hit_tp = False
            if direction == LONG_SIGNAL:
                if current_low <= sl_price: hit_sl = True
                if current_high >= tp_price: hit_tp = True
            else:
                if current_high >= sl_price: hit_sl = True
                if current_low <= tp_price: hit_tp = True

            if hit_sl or hit_tp:
                # Tie-break: without tick data we can't know which level was
                # touched first if both fall inside the same bar's range.
                # Assume SL first — conservative, avoids overstating results.
                exit_reason = "catastrophic_stop_loss" if hit_sl else "take_profit"
                level_price = sl_price if exit_reason == "catastrophic_stop_loss" else tp_price

                # Gap-aware fill: if the open already cleared the level, the
                # real fill is at open, not at the theoretical level price.
                if direction == LONG_SIGNAL:
                    exit_price = min(current_open, level_price) if exit_reason == "catastrophic_stop_loss" \
                        else max(current_open, level_price)
                else:
                    exit_price = max(current_open, level_price) if exit_reason == "catastrophic_stop_loss" \
                        else min(current_open, level_price)

                profit = (exit_price - entry_price) if direction == LONG_SIGNAL else (entry_price - exit_price)
                money_profit = calculate_money_profit(profit, lot_size, contract_size)
                comm = calculate_commission(lot_size, commission_per_lot)
                swap = calculate_swap(entry_time, current_time, direction, lot_size, swap_long_per_night, swap_short_per_night)
                account_balance += (money_profit + comm + swap)

                completed_trades.append({
                    "entry_time": entry_time, "exit_time": current_time, "entry_price": entry_price,
                    "exit_price": exit_price, "direction": direction, "lot_size": lot_size,
                    "profit": profit, "money_profit": money_profit, "commission": comm,
                    "swap": swap, "exit_reason": exit_reason,
                    "sl_price": sl_price, "tp_price": tp_price,
                })

                direction = None
                pending_exit = False  # Cancel any queued logic exits since SL/TP already closed the trade

        # STEP 4: End of Bar - Evaluate Thesis for Logic Exits
        if direction is not None:
            if direction == LONG_SIGNAL:
                if current_regime != BULL_TREND_REGIME:
                    pending_exit = True
                    exit_reason_queued = "logic_regime_change"
                elif current_signal == SHORT_SIGNAL:
                    pending_exit = True
                    exit_reason_queued = "logic_opposite_signal"

            elif direction == SHORT_SIGNAL:
                if current_regime != BEAR_TREND_REGIME:
                    pending_exit = True
                    exit_reason_queued = "logic_regime_change"
                elif current_signal == LONG_SIGNAL:
                    pending_exit = True
                    exit_reason_queued = "logic_opposite_signal"

        # STEP 5: End of Bar - Queue New Entries
        if not daily_halted:
            if current_signal in (LONG_SIGNAL, SHORT_SIGNAL):
                if direction is None:
                    # Flat: any signal opens a new position.
                    pending_entry = current_signal
                elif pending_exit and current_signal == -direction:
                    # About to be flat (stop-and-reverse): only queue the
                    # reversal if the new signal genuinely opposes the
                    # direction being closed. A regime-change exit with the
                    # signal still agreeing with the old direction is not a
                    # reversal - staying flat until a fresh signal arrives
                    # avoids immediately re-opening the position we just
                    # closed.
                    pending_entry = current_signal

    return pd.DataFrame(completed_trades, columns=TRADE_LOG_COLUMNS)