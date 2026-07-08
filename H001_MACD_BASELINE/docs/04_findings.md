# Findings

> **Status**: Placeholder. Results will be populated after main.py execution.
> **Last Updated**: [To be filled in after backtest]
> **Data Period**: 2016-01-01 to 2026-07-02

## Summary

This section will contain the quantitative results of the MACD (12, 26, 9) crossover strategy backtest on EUR/USD across M15, H1, and H4 timeframes. The analysis will test the hypothesis from [02_hypothesis.md](02_hypothesis.md) and determine whether MACD generates a statistically meaningful edge or requires refinement (H002, H003, etc.).

---

## 1. Overall Strategy Performance

### Summary Statistics Table

| Metric                      | Value          | Target | Pass/Fail |
| --------------------------- | -------------- | ------ | --------- |
| **Total Trades**            | [To be filled] | > 100  | [ ]       |
| **Winning Trades**          | [To be filled] | —      | [ ]       |
| **Losing Trades**           | [To be filled] | —      | [ ]       |
| **Win Rate %**              | [To be filled] | > 55%  | [ ]       |
| **Average Win (pips)**      | [To be filled] | —      | [ ]       |
| **Average Loss (pips)**     | [To be filled] | —      | [ ]       |
| **Profit Factor**           | [To be filled] | > 1.0  | [ ]       |
| **Expectancy (pips/trade)** | [To be filled] | > 0    | [ ]       |
| **Total Net PnL (USD)**     | [To be filled] | > 0    | [ ]       |
| **Maximum Drawdown %**      | [To be filled] | < 20%  | [ ]       |
| **Sharpe Ratio**            | [To be filled] | > 0.5  | [ ]       |
| **Calmar Ratio**            | [To be filled] | > 1.0  | [ ]       |

### Interpretation

**[To be filled after backtest]**

Expected narrative structure:

- Did we pass all acceptance criteria from [03_experiments.md](03_experiments.md)?
- If yes: MACD has a baseline edge; proceed to H002 refinements (filters, regime detection, etc.).
- If no: Which criteria failed? What does this tell us about MACD?
  - If Win Rate ≈ 50%: MACD alone has no edge; false signals dominate in ranging periods.
  - If Sharpe Ratio < 0.5: Profits exist but are volatile and risky; need variance reduction.
  - If Drawdown > 20%: MACD exhibits catastrophic losing streaks (likely during crises); need regime filter or position sizing.

---

## 2. Performance by Timeframe

### M15 (15-minute bars)

| Metric                    | Value          | Notes                                                 |
| ------------------------- | -------------- | ----------------------------------------------------- |
| **Total Trades**          | [To be filled] | High frequency expected due to bar count.             |
| **Win Rate %**            | [To be filled] | Hypothesis: Lower than H1/H4 due to noise.            |
| **Profit Factor**         | [To be filled] | Expected: 0.8–1.1 (whipsaw on tight timeframe).       |
| **Average PnL per Trade** | [To be filled] | —                                                     |
| **Max Drawdown %**        | [To be filled] | Expected: Higher than H1/H4 due to noise sensitivity. |
| **Sharpe Ratio**          | [To be filled] | Expected: Lower than H1/H4.                           |

**Analysis:** [To be filled]

### H1 (Hourly bars)

| Metric                    | Value          | Notes                                                             |
| ------------------------- | -------------- | ----------------------------------------------------------------- |
| **Total Trades**          | [To be filled] | Medium frequency; balance between signal clarity and sample size. |
| **Win Rate %**            | [To be filled] | Hypothesis: Higher than M15, comparable to or slightly below H4.  |
| **Profit Factor**         | [To be filled] | Expected: 1.0–1.3 (good balance).                                 |
| **Average PnL per Trade** | [To be filled] | —                                                                 |
| **Max Drawdown %**        | [To be filled] | Expected: Moderate.                                               |
| **Sharpe Ratio**          | [To be filled] | Expected: Moderate to good.                                       |

**Analysis:** [To be filled]

### H4 (4-hour bars)

| Metric                    | Value          | Notes                                                    |
| ------------------------- | -------------- | -------------------------------------------------------- |
| **Total Trades**          | [To be filled] | Lowest frequency due to bar count; likely < 1000 trades. |
| **Win Rate %**            | [To be filled] | Hypothesis: Highest win rate (noise filtered out).       |
| **Profit Factor**         | [To be filled] | Expected: 1.1–1.5 (cleanest signals).                    |
| **Average PnL per Trade** | [To be filled] | —                                                        |
| **Max Drawdown %**        | [To be filled] | Expected: Lowest (fewer whipsaws).                       |
| **Sharpe Ratio**          | [To be filled] | Expected: Highest.                                       |

**Analysis:** [To be filled]

### Timeframe Comparison Chart

**[To be inserted: Bar chart comparing Win Rate, Profit Factor, Sharpe Ratio across M15, H1, H4]**

### Conclusion on Timeframe Effect

**Hypothesis Prediction**: H4 > H1 > M15 (in win rate and Sharpe ratio)

**Actual Result**: [To be filled]

**Verdict**: [ ] Confirmed / [ ] Partially Confirmed / [ ] Refuted

---

## 3. Performance by Market Regime

### Trending vs. Ranging Classification

**Method**: [To be filled — e.g., ADX threshold, EMA slope, range size...]

For each trade, we classify the period as:

- **Trending**: ADX > 25 (or equivalent), clear directional bias
- **Ranging**: ADX < 20, price oscillating within bounds
- **Transitional**: 20 ≤ ADX ≤ 25 (excluded for clarity)

### Performance During Trending Periods

| Metric                | M15            | H1             | H4             |
| --------------------- | -------------- | -------------- | -------------- |
| **Trades (count)**    | [To be filled] | [To be filled] | [To be filled] |
| **Win Rate %**        | [To be filled] | [To be filled] | [To be filled] |
| **Profit Factor**     | [To be filled] | [To be filled] | [To be filled] |
| **Avg PnL per Trade** | [To be filled] | [To be filled] | [To be filled] |

**Expected**: Win Rate > 60%, PF > 1.2 on H4 during trending periods.

**Analysis**: [To be filled]

### Performance During Ranging Periods

| Metric                | M15            | H1             | H4             |
| --------------------- | -------------- | -------------- | -------------- |
| **Trades (count)**    | [To be filled] | [To be filled] | [To be filled] |
| **Win Rate %**        | [To be filled] | [To be filled] | [To be filled] |
| **Profit Factor**     | [To be filled] | [To be filled] | [To be filled] |
| **Avg PnL per Trade** | [To be filled] | [To be filled] | [To be filled] |

**Expected**: Win Rate ≈ 45–50%, PF < 1.0 (negative edge).

**Analysis**: [To be filled]

### Regime Comparison Chart

**[To be inserted: Bar chart comparing Win Rate during Trending vs. Ranging for each timeframe]**

### Conclusion on Regime Effect

**Hypothesis Prediction**: Trending periods show >55% win rate; Ranging periods show <50% win rate.

**Actual Result**: [To be filled]

**Verdict**: [ ] Confirmed / [ ] Partially Confirmed / [ ] Refuted

---

## 4. Equity Curve Analysis

### Cumulative Net PnL Over Time

**[To be inserted: Line chart showing cumulative equity from 2016-01-01 to 2026-07-02]**

**Expected shape**:

- Upward trend overall (if hypothesis is correct)
- Drawdowns during ranging periods or crisis events (Mar 2020, Sept 2023, etc.)
- Recovery after drawdowns

### Key Observations

| Event / Period             | Years     | Observed PnL Behavior | Explanation                                                           |
| -------------------------- | --------- | --------------------- | --------------------------------------------------------------------- |
| **Inception**              | 2016      | [To be filled]        | —                                                                     |
| **Brexit Volatility**      | 2016–2017 | [To be filled]        | Expected: Trending spike in EUR volatility; MACD should perform well. |
| **Fed Hikes / Tapering**   | 2018–2019 | [To be filled]        | Expected: Trending period; MACD should perform well.                  |
| **Pandemic Crash**         | Mar 2020  | [To be filled]        | Expected: Extreme volatility, gaps, possible slippage; drawdown.      |
| **Inflation / Rate Hikes** | 2021–2023 | [To be filled]        | Expected: Strong trending; MACD should perform well.                  |
| **Soft Landing**           | 2023–2024 | [To be filled]        | Expected: Range-bound consolidation; MACD struggles.                  |
| **Forecast Period**        | 2025–2026 | [To be filled]        | Current market conditions.                                            |

### Maximum Drawdown Analysis

| Metric                           | Value          | Notes                                    |
| -------------------------------- | -------------- | ---------------------------------------- |
| **Max Drawdown %**               | [To be filled] | Worst peak-to-trough on equity curve.    |
| **Max Drawdown Duration (days)** | [To be filled] | How long to recover from worst drawdown. |
| **Drawdown Start Date**          | [To be filled] | When did worst drawdown occur?           |
| **Number of Drawdown > 10%**     | [To be filled] | How many times did equity fall > 10%?    |
| **Average Recovery Time**        | [To be filled] | Typical time to recover from drawdown.   |

**Analysis**: [To be filled]

### Monthly / Quarterly Breakdown

**[To be inserted: Heatmap or table showing Win Rate, PF, ROI % per month/quarter over the 10-year period]**

This will reveal seasonal patterns or regime changes. For example:

- Does MACD perform better/worse in certain months (e.g., summer doldrums)?
- Are there quarters where MACD completely breaks down (ranging markets)?

### Conclusion on Equity Curve

**Questions to answer**:

1. Is the equity curve smooth or jagged? (Smooth = consistent edge; Jagged = unreliable edge)
2. Do drawdowns recover quickly or slowly? (Quick = mean-reverting; Slow = regime change)
3. Is there a clear inflection point where strategy stops working? (Yes → regime filter needed)

**Analysis**: [To be filled]

---

## 5. Key Insights & Validation Against Hypothesis

### Does MACD Have a Baseline Edge?

**[To be filled]**

Criteria from [02_hypothesis.md](02_hypothesis.md):

- [ ] Win Rate > 55%
- [ ] Profit Factor > 1.0
- [ ] Sharpe Ratio > 0.5
- [ ] Max Drawdown < 20%
- [ ] Trending periods >> Ranging periods (confirmed by regime analysis)

### Why or Why Not?

**[To be filled]**

Expected explanations:

- **If edge exists**: MACD crossovers genuinely mark trend changes in EUR/USD. The edge is larger on H4 (noise filtering) and during trending periods (directional bias).
- **If edge does NOT exist**: MACD produces false signals as often as valid ones in EUR/USD. The strategy needs regime detection, parameter optimization, or combination with other indicators.

### Unexpected Findings

**[To be filled]**

Any surprises? E.g.,

- Did H1 outperform H4 (contrary to hypothesis)?
- Did ranging periods still produce positive returns (contrary to hypothesis)?
- Were there specific years/quarters where the strategy broke completely?

---

## 6. Recommendations for H002 (Next Iteration)

Based on H001 results, the next experiment (H002) should prioritize:

**[To be filled after backtest results are analyzed]**

Expected candidates:

1. **Regime Filter**: Add ADX threshold to trade only during trending periods.
2. **Parameter Optimization**: Test MACD parameters (10/20/5, 5/35/5, etc.) to find optimal settings for EUR/USD.
3. **Entry Filter**: Combine MACD with RSI or volatility to reduce false signals.
4. **Dynamic Stops**: Replace fixed 20-pip stop with ATR-based stops.
5. **Session Filter**: Trade only during liquid sessions (8:00–17:00 GMT) to avoid overnight slippage.

---

## Appendix: Trade Log

**[To be attached: Full CSV export of all trades]**

Columns:

- Trade ID
- Entry Date/Time
- Entry Price
- Entry Signal (LONG/SHORT)
- Exit Date/Time
- Exit Price
- Exit Reason (Stop/Target/MaxDuration)
- PnL (gross, net)
- Win/Loss
- Duration (bars)
- Timeframe (M15/H1/H4)
- Regime (Trending/Ranging)

**See**: [../output/trades.csv](../output/trades.csv)
