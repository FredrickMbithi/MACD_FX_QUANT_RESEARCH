"""
main.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import execution
import export
import indicators
import signals
import utils
import validate

def calculate_risk_based_lot_size(
    account_balance: float, risk_percent: float, stop_loss_pips: float,
    pip_size: float | None = None, symbol: str | None = None,
    contract_size: float = execution.STANDARD_LOT_UNITS,
) -> float:
    if pip_size is not None: resolved_pip_size = pip_size
    elif symbol is not None: resolved_pip_size = execution.get_pip_size(symbol)
    else: resolved_pip_size = execution.PIP_SIZE

    risk_amount = account_balance * (risk_percent / 100.0)
    loss_per_standard_lot = stop_loss_pips * resolved_pip_size * contract_size
    if loss_per_standard_lot <= 0:
        raise ValueError("stop_loss_pips must be positive to size risk-based trades")
    return risk_amount / loss_per_standard_lot


def main(config_path: str = "config.yaml") -> None:
    config = utils.load_config(config_path)
    logger = utils.setup_logger(name="H001_MACD_BASELINE", log_dir=config["output"]["log_dir"])
    logger.info("Pipeline started (config: %s)", config_path)

    # STAGE 1: Validate data
    raw_df = utils.load_ohlc_csv(config["data"]["raw_csv_path"])
    report = validate.validate_ohlc(raw_df)
    if not report.is_valid:
        logger.error("Data validation failed - stopping.")
        sys.exit(1)

    # STAGE 2: Indicators
    strategy_cfg = config["strategy"]
    indicators_df = indicators.add_macd_indicators(
        raw_df, fast_period=strategy_cfg.get("macd_fast_period", 12),
        slow_period=strategy_cfg.get("macd_slow_period", 26),
        signal_period=strategy_cfg.get("macd_signal_period", 9),
    )

    # STAGE 3: Signals
    signal_cfg = config.get("signals", {})
    signals_df = signals.generate_signals(
        indicators_df, signal_source=config["strategy"].get("signal_source", "macd"),
        regime_engine="trend", trend_feature_set=signal_cfg.get("feature_set", signals.DEFAULT_TREND_FEATURE_SET),
        vol_window=signal_cfg.get("vol_window", signals.DEFAULT_VOL_WINDOW),
        trend_refit_interval=signal_cfg.get("refit_interval", signals.DEFAULT_TREND_REFIT_INTERVAL),
        trend_train_window=signal_cfg.get("train_window", signals.DEFAULT_TREND_TRAIN_WINDOW),
    )

    # STAGE 4: Execute trades (Updated for Logic Exits)
    risk_cfg = config.get("risk", {})
    symbol = config["data"]["symbol"]
    initial_balance = float(config.get("account", {}).get("initial_balance", 10_000.0))
    risk_percent = float(risk_cfg.get("per_trade_percent", 0.25))
    daily_drawdown_percent = float(risk_cfg.get("daily_drawdown_percent", 0.5))
    
    # We now use a wide catastrophic SL for risk sizing
    catastrophic_sl_pips = float(risk_cfg.get("catastrophic_sl_pips", execution.DEFAULT_CATASTROPHIC_SL_PIPS))
    contract_size = 100.0 if "XAU" in symbol.upper() else execution.STANDARD_LOT_UNITS

    lot_size = calculate_risk_based_lot_size(
        account_balance=initial_balance, risk_percent=risk_percent,
        stop_loss_pips=catastrophic_sl_pips, symbol=symbol, contract_size=contract_size,
    )
    
    logger.info("Risk sizing: risk %.3f%%, catastrophic stop %.1f pips, lot size %.6f", 
                risk_percent, catastrophic_sl_pips, lot_size)

    trades = execution.run_backtest(
        signals_df, starting_balance=initial_balance, lot_size=lot_size,
        contract_size=contract_size, commission_per_lot=config["costs"]["commission_per_lot"],
        swap_long_per_night=config["costs"]["swap_long_per_night"],
        swap_short_per_night=config["costs"]["swap_short_per_night"],
        catastrophic_sl_pips=catastrophic_sl_pips, # Passing new param
        daily_drawdown_percent=daily_drawdown_percent, symbol=symbol,
    )
    logger.info("Backtest complete: %d trade(s) executed", len(trades))

    # STAGE 5: Export
    utils.ensure_directory_exists(config["output"]["output_dir"])
    output_path = f"{config['output']['output_dir']}/{config['output']['quant_analyzer_filename']}"
    export.export_trades_to_csv(trades, output_path, symbol=symbol)

    # STAGE 6: Verify the export's column count still matches what Quant
    # Analyzer's Format #3 expects (see export.py's module docstring).
    expected_column_count = config["output"]["expected_column_count"]
    if not export.verify_column_count(output_path, expected_count=expected_column_count):
        logger.error(
            "Export column count mismatch: expected %d columns in %s",
            expected_column_count, output_path,
        )
        sys.exit(1)

    logger.info("Pipeline finished successfully.")

if __name__ == "__main__":
    config_path_arg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(config_path_arg)