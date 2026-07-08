"""
main.py
=======

Purpose
-------
Coordinate the H001_MACD_BASELINE workflow end to end: load config,
validate data, calculate indicators, generate signals, execute trades,
and export a Quant Analyzer-compatible CSV.

    Load config
        |
    Validate data
        |
    Calculate indicators
        |
    Generate signals
        |
    Execute trades
        |
    Export Quant Analyzer CSV

Responsibilities
-----------------
This file is ONLY responsible for calling the right function, in the
right order, and reporting progress as it goes. It deliberately does
NOT:
- implement any indicator, signal, execution, or export logic itself
  (that lives in indicators.py/signals.py/execution.py/export.py)
- decide what "valid" data looks like (validate.py)
- know any file-I/O or logging mechanics (utils.py)
If you're ever tempted to write a formula, a loop with trading logic,
or a file-handling detail directly in this file, that logic belongs in
one of the other six files instead - main.py only sequences them.

Inputs
------
A config.yaml file (path passed as a command-line argument, or
defaulting to "config.yaml" in the current directory). See
config_sample.yaml alongside this file for the exact keys expected.

Outputs
-------
- A completed Quant Analyzer CSV at the path built from config.yaml's
  output section.
- A timestamped log file recording every stage's progress and any
  validation issues found.
- The same progress messages mirrored to the console.

Assumptions
-----------
- config.yaml exists and has the structure documented in
  config_sample.yaml - this file does not validate the config's own
  structure (see utils.load_config()'s docstring for why).
- The raw CSV at config["data"]["raw_csv_path"] has the schema
  validate.py expects (time_utc, open, high, low, close).

Possible edge cases
--------------------
- validate_ohlc() finds ERROR-level issues: the pipeline logs them and
  stops immediately, before indicators are ever calculated - see the
  "Validate data" stage below for why this can't be a warning-and-
  continue situation.
- validate_ohlc() finds only WARNING-level issues: logged, and the
  pipeline continues - warnings (e.g. unsorted rows) don't indicate
  corrupted data, just something worth a human's attention.
- Zero completed trades: export.py still writes a valid CSV containing
  only a header row; this is logged as a notable outcome, not an error
  - a strategy that never triggers isn't necessarily broken.
- No reference_csv_path configured: the final shape-check stage is
  skipped entirely, rather than treated as a failure.

Future improvements
--------------------
- Support multiple timeframes/symbols in one config.yaml run (today,
  one run produces one CSV for one symbol/timeframe).
- Add a --validate-only flag that runs just the "Validate data" stage,
  for quickly checking new data before committing to a full backtest.
"""

from __future__ import annotations

import sys
from pathlib import Path

# WHY sys.path is modified here, rather than making src/ a package and
# using "from src import execution": every file under src/ (execution.py,
# export.py, etc.) uses FLAT imports internally - e.g. execution.py
# does "from signals import LONG_SIGNAL, SHORT_SIGNAL", not
# "from .signals import ...". That's a deliberate choice in those
# files: a relative import would break their own standalone __main__
# demos ("python3 execution.py" directly), which is how every file in
# this project has been hand-verified throughout its build. Adding
# src/ to sys.path here makes those same flat imports resolve
# correctly when the files are imported from main.py too - one
# mechanism, no relative-import/flat-import mismatch between the two
# ways these files get run.
sys.path.insert(0, str(Path(__file__).parent / "src"))

import execution
import export
import indicators
import signals
import utils
import validate


def calculate_risk_based_lot_size(
    account_balance: float,
    risk_percent: float,
    stop_loss_pips: float,
    pip_size: float = execution.PIP_SIZE,
    contract_size: float = execution.STANDARD_LOT_UNITS,
) -> float:
    """
    Convert an account-risk percentage into a fixed lot size.

    The sizing is based on the stop-loss distance only, so the trade's
    planned loss at the stop is approximately account_balance *
    risk_percent.
    """
    risk_amount = account_balance * (risk_percent / 100.0)
    loss_per_standard_lot = stop_loss_pips * pip_size * contract_size
    if loss_per_standard_lot <= 0:
        raise ValueError("stop_loss_pips must be positive to size risk-based trades")
    return risk_amount / loss_per_standard_lot


def main(config_path: str = "config.yaml") -> None:
    """
    Run the full H001_MACD_BASELINE pipeline once, start to finish.

    Parameters
    ----------
    config_path : str, default "config.yaml"
        Path to the YAML config file driving this run.
    """
    # STAGE 0: load config. Every stage below reads its settings from
    # this dict rather than hardcoding values directly in this file -
    # changing a stop-loss distance or a file path should only ever
    # require editing config.yaml, never this script.
    config = utils.load_config(config_path)

    logger = utils.setup_logger(
        name="H001_MACD_BASELINE",
        log_dir=config["output"]["log_dir"],
    )
    logger.info("Pipeline started (config: %s)", config_path)

    # STAGE 1: validate data. This runs before anything else even looks
    # at the data, because every later stage assumes clean input.
    # Catching a bad timestamp or a high < low bar here is cheap;
    # catching it after indicators/signals/execution have already run
    # on it would not be - and worse, it might not be caught at all,
    # just silently produce wrong results.
    raw_df = utils.load_ohlc_csv(config["data"]["raw_csv_path"])
    report = validate.validate_ohlc(raw_df)

    for issue in report.warnings():
        logger.warning("%s: %s", issue.check, issue.message)

    if not report.is_valid:
        for issue in report.errors():
            logger.error("%s: %s", issue.check, issue.message)
        logger.error(
            "Data validation failed - stopping before indicators are calculated."
        )
        sys.exit(1)

    logger.info(
        "Data validated: %d row(s), 0 error(s), %d warning(s)",
        len(raw_df),
        len(report.warnings()),
    )

    # STAGE 2: calculate indicators.
    strategy_cfg = config["strategy"]
    indicators_df = indicators.add_trix_indicators(
        raw_df,
        period=strategy_cfg.get("trix_period", 14),
        signal_span=strategy_cfg.get("trix_signal_span", 9),
    )
    logger.info("Indicators calculated (TRIX/Signal)")

    # STAGE 3: generate signals.
    signal_cfg = config.get("signals", {})
    signals_df = signals.generate_signals(
        indicators_df,
        vol_window=signal_cfg.get("vol_window", signals.DEFAULT_VOL_WINDOW),
        feature_set=signal_cfg.get("feature_set", signals.DEFAULT_HMM_FEATURE_SET),
        signal_source="trix",
        refit_interval=signal_cfg.get("refit_interval", 100),
        train_window=signal_cfg.get("train_window", signals.DEFAULT_TRAIN_WINDOW),
    )
    signal_count = int(signals_df["signal"].notna().sum())
    logger.info("Signals generated: %d crossover(s) found", signal_count)

    # STAGE 4: execute trades. H001 is a stop-and-reverse strategy - no
    # stop_loss_distance/take_profit_distance exists anymore (see
    # execution.py's module docstring for why).
    position_cfg = config["position"]
    risk_cfg = config.get("risk", {})
    account_cfg = config.get("account", {})
    initial_balance = float(account_cfg.get("initial_balance", 10_000.0))
    risk_percent = float(risk_cfg.get("per_trade_percent", 0.25))
    daily_drawdown_percent = float(risk_cfg.get("daily_drawdown_percent", 0.5))
    stop_loss_pips = float(risk_cfg.get("stop_loss_pips", execution.DEFAULT_STOP_LOSS_PIPS))
    take_profit_pips = float(risk_cfg.get("take_profit_pips", execution.DEFAULT_TAKE_PROFIT_PIPS))
    lot_size = calculate_risk_based_lot_size(
        account_balance=initial_balance,
        risk_percent=risk_percent,
        stop_loss_pips=stop_loss_pips,
    )
    logger.info(
        "Risk sizing enabled: initial balance %.2f, risk %.3f%%, daily drawdown %.3f%%, stop %.1f pips, lot size %.6f",
        initial_balance,
        risk_percent,
        daily_drawdown_percent,
        stop_loss_pips,
        lot_size,
    )
    costs_cfg = config["costs"]
    trades = execution.run_backtest(
        signals_df,
        starting_balance=initial_balance,
        lot_size=lot_size,
        commission_per_lot=costs_cfg["commission_per_lot"],
        swap_long_per_night=costs_cfg["swap_long_per_night"],
        swap_short_per_night=costs_cfg["swap_short_per_night"],
        stop_loss_pips=stop_loss_pips,
        take_profit_pips=take_profit_pips,
        daily_drawdown_percent=daily_drawdown_percent,
    )
    logger.info("Backtest complete: %d trade(s) executed", len(trades))

    # STAGE 5: export Quant Analyzer CSV.
    output_cfg = config["output"]
    utils.ensure_directory_exists(output_cfg["output_dir"])
    output_path = f"{output_cfg['output_dir']}/{output_cfg['quant_analyzer_filename']}"
    export.export_trades_to_csv(trades, output_path, symbol=config["data"]["symbol"])
    logger.info("Exported Quant Analyzer CSV to %s", output_path)

    # STAGE 6: sanity-check the export's column count against what
    # Quant Analyzer's Format #3 expects (15). An earlier version of
    # this stage compared the export's header text against a reference
    # file - that check is gone now, because we discovered Quant
    # Analyzer's auto-detection doesn't read header text at all; it
    # applies Format #3 by column POSITION. Column count is the thing
    # that can now actually drift and silently break an import (e.g. a
    # future edit removing a column from QA_FORMAT3_COLUMNS would shift
    # position 13 away from PL without any error being raised).
    expected_columns = output_cfg.get("expected_column_count", 15)
    if not export.verify_column_count(output_path, expected_columns):
        logger.warning(
            "Export has an unexpected column count (expected %d) - "
            "check QA_FORMAT3_COLUMNS in export.py before trusting this import.",
            expected_columns,
        )
    else:
        logger.info("Export column count matches Quant Analyzer's Format #3 - OK")

    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    config_path_arg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(config_path_arg)
