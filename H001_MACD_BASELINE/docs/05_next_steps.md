# Next Steps & Research Roadmap

## Overview

This document outlines the research roadmap for improving on the baseline MACD strategy (H001). Depending on the results from [04_findings.md](04_findings.md), the priority of these steps may shift, but the general philosophy remains: validate each hypothesis independently, measure the edge contribution of each refinement, and maintain a baseline to compare against.

---

## Phase 1: Validate & Understand H001 Results

Before improving the strategy, we must fully understand _why_ the baseline performs as it does.

### Step 1.1: Trade-by-Trade Analysis

**Objective**: Understand _which trades_ won and _which lost_, and why.

**Data to Examine** (from [../output/trades.csv](../output/trades.csv)):

1. **All losing trades**:
   - How many closed quickly on the next opposite crossover?
   - How many gave back an initially favorable move before the opposite signal appeared?
   - How long did the average losing signal-to-signal move last?
   - What was the market regime at entry? (trending vs. ranging)
   - Were they consecutive (losing streak)? Or scattered?

2. **All winning trades**:
   - Did most capture a clean signal-to-signal trend?
   - Or did they win despite multiple shallow reversals before the opposite crossover?
   - What was the market regime at entry?
   - Average win size vs. average loss size ratio.

3. **Biggest winners and losers**:
   - Which single trade made the most money? What market conditions?
   - Which single trade lost the most? What were the market conditions?
   - Can we replicate the winners and avoid the losers?

4. **Time-of-day patterns**:
   - Do certain hours perform better (e.g., London open better than Tokyo)?
   - This will inform the "session filter" idea (H003).

5. **Drawdown events**:
   - Which specific trades caused the largest equity drawdowns?
   - Were they clustered in time (crisis event) or scattered?
   - Could a regime filter have prevented them?

### Step 1.2: Equity Curve Forensics

**Objective**: Identify the exact periods where the strategy breaks and why.

**Analysis**:

- Plot equity curve month-by-month or quarter-by-quarter.
- Overlay with market regime (e.g., ADX) and major news events (rate decisions, geopolitical events).
- Q: "Did the strategy stop working after a specific date?" If yes, what changed in the market?
- Q: "Are the drawdowns predictable in hindsight?" If yes, a filter can avoid them.

### Step 1.3: Hypothesis Validation

**Question**: Does the empirical result match the hypothesis from [02_hypothesis.md](02_hypothesis.md)?

- **If Yes**: MACD has an edge on trending markets. Proceed with refinements (below).
- **If No**: Diagnose why.
  - Is MACD simply not suited to EUR/USD? (Compare to other pairs in H003–H005.)
  - Is the edge present but too small (< 0.5 Sharpe)? (Refinements may not help; abandon this line.)
  - Is the edge present only on certain timeframes? (Focus future work on that timeframe.)
  - Is there a regime-dependency effect? (Build regime filter for H002.)

---

## Phase 2: Next Hypotheses (H002, H003, ...)

### H002: MACD + Regime Filter (ADX)

**Hypothesis**: Adding an ADX filter to trade _only during trending periods_ will improve win rate and reduce drawdown, at the cost of fewer trades.

**Design**:

- Entry condition (unchanged): MACD crosses above/below signal.
- **Additional filter**: ADX(14) > 25 (or tune this threshold).
  - Only enter trades when trend is strong.
  - Skip trades when market is choppy (ADX < 20).
- Exit rules: Unchanged (opposite crossover closes/reverses positions).

**Expected Result**:

- Win rate: +5–10 percentage points (filtering out choppy-market losses).
- Total trades: −30% to −50% (fewer opportunities, but higher quality).
- Profit factor: +10–20% (same win size, fewer losses).
- Max drawdown: Significantly reduced (avoid range-bound whipsaws).

**Success Criteria**:

- Win Rate > 60% (vs ~55% baseline).
- Sharpe Ratio > 0.7 (vs baseline).
- Max Drawdown < 15% (vs ~20% baseline).

**Experiment**:

1. Backtest with ADX threshold = 25; measure results.
2. Backtest with ADX threshold = 20 and threshold = 30; compare.
3. Choose threshold that maximizes Sharpe ratio.
4. Compare H002 to H001: Is the added filter actually valuable, or are we just avoiding trades and cutting into Sharpe?

### H003: MACD + Session Filter

**Hypothesis**: EUR/USD trading MACD is most reliable during the London and New York sessions (high liquidity, tight spreads), and performs poorly during Asian session (wide spreads, illiquid).

**Design**:

- Entry condition (unchanged): MACD crosses above/below signal.
- **Additional filter**: Only enter trades during 08:00–17:00 GMT (London–New York overlap).
  - Close all remaining trades at 17:30 GMT if not exited (avoid overnight gap risk).
- Exit rules: Unchanged.

**Expected Result**:

- Win rate: +3–8% (trading during optimal liquidity).
- Total trades: −50% to −70% (only specific hours).
- Max drawdown: Reduced (no overnight gaps).
- Execution: Closer to backtest assumptions (tighter spreads, no slippage).

**Success Criteria**:

- Win Rate > 57%.
- Sharpe Ratio > 0.6.
- Max Drawdown < 18%.

### H004: MACD Parameter Optimization

**Hypothesis**: The standard MACD parameters (12, 26, 9) are not optimal for EUR/USD on specific timeframes. Optimizing to the data will improve edge.

**Design**:

- Test parameter combinations:
  - Fast EMA: [8, 10, 12, 14]
  - Slow EMA: [21, 24, 26, 28, 35]
  - Signal: [5, 9, 13]
- For each combination, backtest on the full dataset.
- Measure: Win rate, Sharpe, Profit factor.
- Choose the combination with the highest Sharpe ratio.

**Expected Result**:

- Small improvement possible (+2–5% win rate), but risk: overfitting to past data.
- Optimized parameters may not generalize to new/future data.

**Success Criteria**:

- Sharpe Ratio improves > 10% vs baseline.
- Out-of-sample test (2024–2026 data held back) still shows improvement (not overfitting).

**Caution**: This is a slippery slope. Optimization can easily produce false positives (curve-fitting). Use walk-forward analysis to validate.

### H005: MACD on Other Pairs

**Hypothesis**: MACD baseline edge exists on EUR/USD; test if it generalizes to other major pairs (GBP/USD, USD/JPY, AUD/USD) or if it's pair-specific.

**Design**:

- Backtest H001 (plain MACD) on 3–5 other pairs using the same data period and parameters.
- Compare results across pairs.

**Expected Result**:

- If edge is universal: MACD is a general market behavior, not EUR/USD-specific.
- If edge is pair-specific: Different pairs need different strategies or parameters.

### H006: MACD + Dynamic Stops (ATR)

**Hypothesis**: Fixed 20-pip stops are too tight in high-volatility environments and too wide in low-volatility environments. ATR-based stops adapt to market conditions and improve risk-adjusted returns.

**Design**:

- Entry: MACD crossover (unchanged).
- Stop-loss: 1.5 × ATR(14) from entry (not fixed 20 pips).
- Take-profit: 2.5 × ATR(14) from entry (maintains ~2:1 reward/risk).

**Expected Result**:

- Lower stop hit rate in high-volatility (wider stops absorb noise).
- Higher stop hit rate in low-volatility (tighter stops, less wasted capital).
- Overall: Better Sharpe ratio (risk adjusted per market conditions).

---

## Phase 3: Refinement & Combination (H007+)

Once we understand which individual filters work (H002, H003, H004), we can combine them.

### H007: MACD + Regime Filter + Session Filter

**Objective**: Combine the best ideas from H002 and H003.

**Design**:

- Entry: MACD crossover + ADX > 25 + 08:00–17:00 GMT.
- Exit: opposite crossover, or ATR-based risk exit from H006 if that experiment proves valuable.

**Expected**:

- Win Rate > 62%.
- Sharpe > 0.8.
- Much fewer trades (~500–1000 total), but very high quality.

### H008: Machine Learning Filter

**Hypothesis**: A classification model (Random Forest, XGBoost, or Neural Network) can predict which MACD crossovers are true signals and which are false, given market regime features.

**Design**:

- Features: ADX, RSI, Volume, Time of Day, Recent volatility, MACD histogram magnitude.
- Label: Does this MACD crossover lead to a profitable trade (Y/N)?
- Train on 2016–2023 data; validate on 2024–2026 data.
- Only enter trades if model predicts "Y".

**Expected**:

- Win rate could reach 65–70% if model learns genuine patterns.
- Risk: Overfitting; model may fail on truly new market conditions.

---

## HMM Observation Vector Clarification

The current HMM regime filter in [../src/signals.py](../src/signals.py) does **not** use raw OHLCV prices as its observation vector. It currently fits a 2D Gaussian HMM on:

- log return of close: `log(close_t / close_{t-1})`
- rolling volatility of that log return series

In code terms, the observation matrix is the two-column feature set returned by `compute_hmm_features()`:

```python
X = df[["log_return", "volatility"]]
```

That means any future experiment labeled "OHLCV" should first define a transformed feature vector explicitly, rather than using raw OHLC prices directly. A sensible candidate would be a stationary feature set such as returns, range, and normalized volume, and then hold that definition fixed across both sample periods in the A/B/C/D matrix.

---

## Phase 4: Walk-Forward & Live Testing

Once we have a refined, validated strategy (likely H007), we move to real-world testing:

### Step 4.1: Walk-Forward Validation

**Objective**: Prove the strategy is not overfit to past data.

**Method**:

- Train parameters on 2016–2022 data.
- Test on 2023–2026 data (held-out).
- If backtest shows 60% win rate but walk-forward shows 45%, the strategy is overfit.
- If both show ≈55–60%, strategy is robust.

### Step 4.2: Paper (Simulated) Trading

**Objective**: Trade the strategy on live price feeds without real money.

**Setup**:

- Connect to broker API (e.g., MT5 WebAPI, cTrader, Interactive Brokers).
- Run live MACD signals on real market data.
- Log all simulated trades; compare to backtest results.
- Measure: execution quality, slippage, spreads.

### Step 4.3: Live Trading (Micro Lot)

**Objective**: Trade for real, but with minimal capital at risk.

**Setup**:

- Trade 0.01 standard lots (1,000 units) instead of 1.0 lots.
- Risk $20 per trade instead of $200.
- Trade for 3–6 months; collect results.
- Measure: Actual win rate, slippage, market impact vs. backtest assumptions.

**Exit Criteria**:

- If live performance matches backtest (±5% win rate): Scale up to 0.1 lots, then 1.0 lot.
- If live performance < backtest by >10 percentage points: Debug; likely a gap between backtest and reality (execution, spread, data quality).

---

## Open Questions from Validation Checklist

From [01_research_question.md](01_research_question.md), section "Validation Checklist", here are the open questions:

### Indicator Validation

- [ ] Are MACD values calculated correctly? (Verify against MT4/TradingView on sample data.)
- [ ] Do the values match a trusted platform?
- [ ] Are there any off-by-one errors in the bar alignment?

### Signal Logic

- [ ] Does every MACD crossover get detected in the signal column?
- [ ] Are any valid signals being missed?
- [ ] Are false signals (noise-level crossovers) being generated?

### Entry Logic

- [ ] Is entry on the bar after the crossover (not on the crossover bar itself)? (Avoid look-ahead bias.)
- [ ] Is entry price realistic (no gap through entry price)?

### Exit Logic

- [ ] Do opposite-signal exits execute on the bar after the crossover, not the crossover bar itself?
- [ ] Should there be a maximum holding time, or should H001 remain purely signal-to-signal?

### Risk Management

- [ ] Is uncapped signal-to-signal risk acceptable for baseline research?
- [ ] Would a separate risk exit improve drawdown without hiding MACD signal quality?
- [ ] Should position size scale with account size (position sizing rules)?

### Look-Ahead Bias

- [ ] Are we using only historical data when calculating MACD? (Yes—EMA only looks backward.)
- [ ] Is entry on the next bar after the signal? (Yes—prevents using the close before it's known.)

---

## Priority Roadmap

**Immediate (H002–H003):**

1. ADX regime filter (H002) — highest expected ROI for effort.
2. Session filter (H003) — easy to implement, clear effect.
3. Trade-by-trade analysis (Phase 1) — understand what's happening.

**Short-term (H004–H006):**

1. ATR-based stops (H006) — risk-adaptive; pairs well with regime filter.
2. Parameter optimization (H004) — only if H001 shows edge but needs tuning.

**Medium-term (H007, Walk-forward):**

1. Combine best filters (H007).
2. Validate on held-out data (Walk-forward).

**Long-term (H008, Live Testing):**

1. Machine learning filter (H008) — only if simpler approaches plateau.
2. Paper trading and live micro-lot trading.

---

## Failure Modes & Exit Criteria

**If H001 results show:**

| Result                         | Interpretation                               | Action                                                                                  |
| ------------------------------ | -------------------------------------------- | --------------------------------------------------------------------------------------- |
| **Win Rate = 50%, PF = 1.0**   | MACD has no edge on EUR/USD.                 | Stop MACD research; try different indicators (RSI, Bollinger Bands, etc.).              |
| **Win Rate = 52%, PF = 1.05**  | Edge is marginal; refinements won't save it. | Acceptable for low-frequency system (few trades); not for scalping. Proceed cautiously. |
| **Win Rate = 55%, PF = 1.1**   | Edge exists but small.                       | H002–H007 refinements are worth the effort.                                             |
| **Win Rate > 60%, PF > 1.3**   | Strong edge.                                 | Focus on risk management and live deployment.                                           |
| **Sharpe < 0.3, Max DD > 30%** | Edge is too volatile; risky.                 | Stop; too risky for real trading even if technically profitable.                        |

---

## Summary of Future Hypotheses

| Hypothesis | Focus                          | Expected Win Rate | Status                                 |
| ---------- | ------------------------------ | ----------------- | -------------------------------------- |
| **H001**   | Baseline MACD (12, 26, 9)      | 50–57%            | In Progress                            |
| **H002**   | + ADX regime filter            | 58–65%            | Planned                                |
| **H003**   | + Session filter               | 55–62%            | Planned                                |
| **H004**   | Parameter optimization         | 52–60%            | Planned (if needed)                    |
| **H005**   | Other pairs (GBP, JPY, AUD)    | Pair-dependent    | Planned                                |
| **H006**   | ATR-based stops                | 55–62%            | Planned                                |
| **H007**   | Combo (Regime + Session + ATR) | 60–68%            | Planned (if H002, H003, H006 all pass) |
| **H008**   | Machine learning filter        | 65–72%            | Planned (phase 3+)                     |

---

## Version Control & Documentation

- Each hypothesis gets its own folder: `H001_MACD_BASELINE`, `H002_MACD_REGIME_FILTER`, etc.
- Each folder contains:
  - `docs/02_hypothesis.md` (formal hypothesis)
  - `docs/03_experiments.md` (design)
  - `docs/04_findings.md` (results)
  - `config.yaml` (parameters)
  - `src/` (code)
  - `output/trades.csv` (trade log)
  - `output/quant_analyzer.csv` (summary statistics)
- Always preserve H001 as the baseline for comparison.
- Track Sharpe ratio, Win rate, and Profit factor across all versions in a master spreadsheet.

---

**Next Action**: Run main.py on H001, populate [04_findings.md](04_findings.md) with results, then decide which of H002, H003, or H006 to pursue first based on the findings.
