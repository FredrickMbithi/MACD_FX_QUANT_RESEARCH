# Experiments Design

## Objective

Validate the hypothesis in [02_hypothesis.md](02_hypothesis.md) by backtesting a MACD crossover strategy on historical EUR/USD data and exporting the completed trade log for Quant Analyzer.

The current `config.yaml` runs one dataset per execution. The active baseline run is EUR/USD H1. M15/H4 and regime comparisons remain part of the research plan, but they require separate configured runs and downstream analysis rather than one automatic multi-timeframe pass in `main.py`.

## Test Period

### Data Span

- **Start Date**: 2016-01-01
- **End Date**: 2026-07-02
- **Configured Timeframe**: H1
- **Duration**: 10.5 calendar years
- **Rationale**: Long enough to capture multiple market regimes (trending, ranging, crises, expansions) and multiple interest rate cycles (Fed hikes 2018–2019, pandemic 2020, inflation 2021–2023, rate cuts 2024–2025).

### Data Integrity Checks (Completed in validate.py)

- Required columns exist: `time_utc`, `open`, `high`, `low`, `close`
- No missing OHLC values, invalid timestamps, duplicate timestamps, non-numeric prices, non-positive prices, or non-finite prices
- High/low/open/close relationships are internally consistent
- Timestamp ordering, timestamp gaps, duplicate full rows, price-range outliers, and thin-volume bars are logged as warnings for review where applicable
- `tick_volume` is optional; if present, negative values are errors and unusually low values are warnings

## Strategy Parameters

### MACD Indicator Parameters

| Parameter              | Value | Rationale                                                             |
| ---------------------- | ----- | --------------------------------------------------------------------- |
| **Fast EMA Period**    | 12    | Standard MACD parameter; reacts quickly to recent price momentum.     |
| **Slow EMA Period**    | 26    | Standard MACD parameter; anchors the MACD line to longer-term trend.  |
| **Signal Line Period** | 9     | Standard MACD parameter; EMA of MACD line; smooths crossover signals. |
| **Price Input**        | Close | MACD is conventionally calculated on close price.                     |

**Source**: The 12/26/9 parameters are Gerald Appel's original MACD design (1979) and are the industry standard on all major platforms (MT4, MT5, TradingView, cTrader). Using non-standard parameters (e.g., 10/20/5) would not be comparable to published MACD research.

### Entry Rules

#### LONG Entry

- **Signal**: MACD line crosses **above** the Signal line
- **Entry Price**: Open of the next bar after the crossover bar
- **Rationale**:
  - The crossover bar is the moment momentum shifts upward; the next bar's open is the first executable price.
  - Entering on crossover bar's close would require knowing the close before executing (look-ahead bias).
  - Entering on next bar's open is realistic for mechanical trading systems.

#### SHORT Entry

- **Signal**: MACD line crosses **below** the Signal line
- **Entry Price**: Open of the next bar after the crossover bar
- **Rationale**: Mirrors the LONG entry logic; consistent and unbiased.

### Exit Rules

#### Opposite-Signal Exit / Reversal

- **LONG Exit**: MACD line crosses **below** the Signal line
- **SHORT Exit**: MACD line crosses **above** the Signal line
- **Exit Price**: Open of the next bar after the opposite crossover bar
- **Reversal**: The same opposite signal closes the current position and opens the new opposite position at that same next-bar open
- **Rationale**:
  - Tests MACD's raw signal-to-signal trend-following behavior without imposing arbitrary fixed stops or targets.
  - Uses the same no-look-ahead execution rule as entries: the crossover is known only after the signal bar closes.
  - Makes trade duration an output of the indicator, not a hard-coded assumption.

### Position Sizing

- **Lot Size**: Fixed at 1.0 standard lot (100,000 units of EUR/USD) for all trades
- **Rationale**:
  - Simplifies backtest accounting (no optimization of position size yet).
  - PnL per trade is the full signal-to-signal price move × 1 lot; H001 does not cap risk with fixed stops.
  - Once we have a validated baseline, we can optimize position sizing (fixed fractional, Kelly criterion, etc.) in future experiments.

## Backtest Outputs

### Current Pipeline Outputs

| Output                        | Produced By  | Purpose                                                                                |
| ----------------------------- | ------------ | -------------------------------------------------------------------------------------- |
| **Crossover signal events**   | `signals.py` | Sparse `signal` column containing `LONG` or `SHORT` only on crossover bars             |
| **Position state**            | `signals.py` | Persistent `position` column: `1`, `-1`, or `0`                                        |
| **Completed trade log**       | `export.py`  | Position flips converted into completed round-trip trades with next-bar-open execution |
| **Quant Analyzer import CSV** | `export.py`  | General CSV loader format written to `output/quant_analyzer.csv`                       |
| **Run log**                   | `utils.py`   | Timestamped validation/run log written under `output/logs/`                            |

### PnL Model

| Component            | Current Implementation                                                                |
| -------------------- | ------------------------------------------------------------------------------------- |
| **Raw price profit** | LONG: exit price - entry price; SHORT: entry price - exit price                       |
| **Money profit**     | Raw price profit × lot size × 100,000                                                 |
| **CommSwap**         | `0.0`; commission and swap are not modeled because the raw data does not contain them |
| **Spread/slippage**  | Not modeled yet                                                                       |
| **Unused columns**   | `0.0` placeholders for QuantAnalyzer's configured General CSV layout                  |
| **MAE/MFE**          | Not calculated yet                                                                    |

### QuantAnalyzer General CSV Import

`output/quant_analyzer.csv` includes one header row, so the matching QuantAnalyzer General CSV setting is `Format.1.SkipRow=1`. Import it by explicitly selecting the General CSV loader and using this configured 15-column line format:

`Ticket,OpenTime,Action,Size,Symbol,OpenPrice,Unused,Unused,CloseTime,ClosePrice,CommSwap,Unused,Unused,PL,Comment`

The matching repo config lives at [../settings/plugins/LoaderGeneralCsv/GeneralCSVImport.ini](../settings/plugins/LoaderGeneralCsv/GeneralCSVImport.ini).

### Downstream Metrics to Track in Quant Analyzer

| Analysis                          | Method                                                                     | Purpose                                                         |
| --------------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **Trade-level performance**       | Win/loss, average win/loss, profit factor, expectancy from exported trades | Validate whether the raw crossover baseline has edge.           |
| **Equity curve and drawdown**     | Quant Analyzer equity and drawdown reports                                 | Measure robustness and recovery behavior.                       |
| **Performance by Timeframe**      | Run separate M15, H1, and H4 configs and compare resulting reports         | Validate hypothesis: H4 should outperform M15.                  |
| **Performance by Market Regime**  | Classify each period as Trending vs Ranging; calculate metrics separately  | Validate hypothesis: Trending should vastly outperform Ranging. |
| **Monthly / Quarterly Breakdown** | Group trades by month; calculate win rate and PF per month                 | Identify seasonal patterns or regime changes.                   |

## Why These Choices?

### Indicator Parameters (12, 26, 9)

- **Industry standard**: Every major platform defaults to these. Changing them would require proof they're suboptimal, which is an optimization problem (H002), not a baseline validation.
- **Historical validation**: Decades of published research (Appel, Murphy, de Villiers) validate these parameters across timeframes and markets.

### Entry Rule (Crossover)

- **Simple and unambiguous**: There is no interpretation needed; MACD either crosses or it doesn't.
- **Mechanically executable**: Can be coded without discretion, eliminating trader bias.
- **Published literature**: MACD crossover is the canonical entry signal; starting with anything else adds a layer of hypothesis before testing the base MACD hypothesis.

### Exit Rules (opposite crossover)

- **Indicator purity**: The baseline measures the complete MACD signal-to-signal move rather than a fixed reward/risk overlay.
- **Look-ahead safe**: Opposite signals close positions at the next bar's open, not on the signal bar before the crossover is knowable.
- **Research clarity**: If results are poor, the MACD signal itself is weak; if results are strong, later risk management can be tested as a separate contribution.

### Data Span (2016–2026)

- **Long enough for regime diversity**: Captures bull market (2016–2018), Fed hikes (2018–2019), pandemic (2020), inflation (2021–2023), rate cuts (2024–2025).
- **Large enough sample**: 10+ years generates 1000+ trades (likely), sufficient for statistical significance.
- **Recent data**: 2024–2026 data reflects current market structure; older data alone might be obsolete.

## Backtest Execution

The backtest is executed by [../main.py](../main.py) using the following modules:

- [../src/indicators.py](../src/indicators.py): Calculate MACD, Signal, Histogram
- [../src/signals.py](../src/signals.py): Detect LONG/SHORT crossovers
- [../src/export.py](../src/export.py): Convert position flips into completed trades and write the Quant Analyzer General CSV
- [../src/validate.py](../src/validate.py): Verify data integrity before backtest

## Next Isolation Experiment

To isolate the effect of the HMM observation space, keep the strategy, execution, and regime logic fixed and vary only the sample period and the HMM input.

| Experiment | Data Span | HMM Input |
| ---------- | --------- | --------- |
| A          | 2023–2026 | MACD      |
| B          | 2023–2026 | OHLCV     |
| C          | 2016–2026 | MACD      |
| D          | 2016–2026 | OHLCV     |

This makes the question sharper: does OHLCV improve the HMM independently of the sample period?

Important: the current codebase does not fit the HMM on raw OHLCV. The existing regime filter uses close-derived features instead: log returns and rolling volatility. If we want to label an experiment “OHLCV,” we should define the exact transformed observation vector first and keep it fixed across A–D.

## Success Criteria (Accept or Reject H₁)

For the current H1 baseline run, accept H₁ if the Quant Analyzer report shows all of the following:

1. Win Rate ≥ 55%
2. Profit Factor > 1.0
3. Sharpe Ratio > 0.5
4. Max Drawdown < 20%

Timeframe and regime-dependency criteria require additional runs/analysis:

1. Performance on M15 is worse than H1 and H4
2. Performance during ranging periods is < 50%

**Reject H₁ and iterate (H002) if any fail.**
