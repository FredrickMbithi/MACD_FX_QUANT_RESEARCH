# Findings

> **Status**: H1 baseline run complete.
> **Last Updated**: 2026-07-11
> **Data Period**: 2016-01-01 to 2026-07-02 (trades observed: 2017-06-07 to 2026-01-xx)
> **Run**: `main.py` on `config.yaml` (EURUSD, H1, MACD 12/26/9 + HMM trend-regime gate, risk-based position sizing, catastrophic SL / fixed-RR TP). Source: [../output/quant_analyzer.csv](../output/quant_analyzer.csv).

## Summary

This section contains the quantitative results of the MACD (12, 26, 9) crossover strategy, gated by a walk-forward HMM trend-regime filter, backtested on EURUSD H1 only (the currently configured dataset - see [03_experiments.md](03_experiments.md) for why M15/H4 require separate runs). The result: **H1 does NOT clear the H001 acceptance criteria** from [02_hypothesis.md](02_hypothesis.md). The baseline is unprofitable net of costs, with a win rate and profit factor both below breakeven thresholds.

---

## 1. Overall Strategy Performance

### Summary Statistics Table

| Metric                      | Value        | Target | Pass/Fail |
| ---------------------------- | ------------- | ------ | --------- |
| **Total Trades**            | 420           | > 100  | [x]       |
| **Winning Trades**          | 180           | —      | —         |
| **Losing Trades**           | 240           | —      | —         |
| **Win Rate %**              | 42.86%        | > 55%  | [ ]       |
| **Average Win (pips)**      | 72.2          | —      | —         |
| **Average Loss (pips)**     | -54.7         | —      | —         |
| **Average Win (USD)**       | $23.64        | —      | —         |
| **Average Loss (USD)**      | -$18.59       | —      | —         |
| **Profit Factor**           | 0.954         | > 1.0  | [ ]       |
| **Expectancy (pips/trade)** | -0.35         | > 0    | [ ]       |
| **Expectancy (USD/trade)**  | -$0.49        | > 0    | [ ]       |
| **Total Net PnL (USD)**     | -$206.04      | > 0    | [ ]       |
| **Total Return**            | -2.06%        | > 0    | [ ]       |
| **Maximum Drawdown %**      | 11.76%        | < 20%  | [x]       |
| **Sharpe Ratio**            | -0.132        | > 0.5  | [ ]       |
| **Calmar Ratio**            | -0.020        | > 1.0  | [ ]       |

Starting balance $10,000, ending balance $9,793.96, over ~9.0 years (420 trades, ~46.5 trades/year). Sharpe is trade-level (mean/std of per-trade return on starting balance), annualized by `sqrt(trades_per_year)`; Calmar is CAGR ÷ max drawdown %. Position size is a fixed 0.033333 lots for every trade (risk-based sizing computed once from the initial balance, not re-sized as equity compounds - see [../src/main.py](../main.py)'s `calculate_risk_based_lot_size` call).

### Interpretation

**3 of 5 acceptance criteria fail.** Win rate (42.86%) is below random and well under the 55% target; profit factor (0.954) is under 1.0, meaning gross losses exceed gross wins; Sharpe is negative. Only trade count (420 > 100) and max drawdown (11.76% < 20%) pass.

- Win Rate ≈ 43% (below random): the MACD+HMM combination is generating a marginally negative edge overall on H1 EURUSD, not the "false signals dominate in ranging periods only" pattern hypothesized - see the exit-reason breakdown below, which shows the losing edge is concentrated in stop-outs, not just ranging whipsaw.
- Profit Factor 0.954 confirms: gross losses ($3,804.66 from stops alone, see below) modestly exceed gross wins ($3,598.62 combined from regime-change + TP exits).
- Max Drawdown 11.76% is within the 20% risk budget - the catastrophic SL and daily-drawdown halt are doing their job of capping tail risk, even though the underlying edge is negative.

### Breakdown by Exit Reason

| Exit Reason               | Count | Total PL (USD) | Avg PL (USD) | Notes                                                          |
| -------------------------- | ----- | --------------- | ------------- | ---------------------------------------------------------------- |
| `catastrophic_stop_loss`   | 150   | -$3,804.66      | -$25.36       | ~35.7% of all trades; full 75-pip SL, this is where the edge is lost. |
| `logic_regime_change`      | 219   | +$1,072.15      | +$4.90        | ~52.1% of all trades; small positive average - HMM regime exits cut losers early and let some winners run partway. |
| `take_profit`               | 51    | +$2,526.47      | +$49.54       | ~12.1% of all trades; full 150-pip TP (2x the 75-pip SL, `risk_reward_ratio=2.0`), largest per-trade win. |

**Key observation**: the regime-change exit (`logic_regime_change`) is mildly profitable on average and is the single most common exit path (52% of trades) - the HMM filter is doing useful work cutting trades short. The strategy's negative overall edge is driven almost entirely by the 150 trades (35.7%) that run the full 75 pips against the position before the regime filter or opposite signal has a chance to intervene.

---

## 2. Performance by Timeframe

**Not run in this experiment.** `config.yaml` is configured for a single dataset per execution (EURUSD H1); M15 and H4 require separate configured runs against `data/raw/EURUSD_M15_*.csv` and `data/raw/EURUSD_H4_*.csv` respectively, per [03_experiments.md](03_experiments.md)'s scoping note. This section will be populated once those runs are executed.

---

## 3. Performance by Market Regime

**Not run in this experiment.** The current `output/quant_analyzer.csv` export (Quant Analyzer's fixed 15-column Format #3) carries `Exit Reason` per trade but not a per-trade regime label at entry, so trades cannot yet be split into Trending vs. Ranging cohorts directly from this file. The `logic_regime_change` exit-reason breakdown above is a related but distinct signal (it tells us how a trade *ended*, not what regime it was *opened in*) and should not be substituted for a true regime-conditioned breakdown.

To populate this section: extend the trade log with the HMM `regime` value at entry (available in `signals_df["regime"]` inside `execution.run_backtest()`, currently not carried into the completed-trade dict), then re-run and group by regime.

---

## 4. Equity Curve Analysis

### Cumulative Net PnL Over Time

Equity is monotonically computed as `starting_balance + cumulative sum of per-trade PL`, ordered by trade close time. Full series available by reconstructing from `output/quant_analyzer.csv`'s `PL` column.

### Key Observations (Yearly Breakdown)

| Year | Trades | Net PL (USD) | Win Rate % |
| ---- | ------ | ------------- | ----------- |
| 2017 | 18     | +$106.54      | 61.1%       |
| 2018 | 25     | +$2.97        | 40.0%       |
| 2019 | 38     | -$123.63      | 39.5%       |
| 2020 | 76     | -$531.23      | 25.0%       |
| 2021 | 28     | -$138.11      | 35.7%       |
| 2022 | 72     | -$271.44      | 43.1%       |
| 2023 | 30     | +$283.34      | 53.3%       |
| 2024 | 51     | -$10.52       | 43.1%       |
| 2025 | 69     | +$362.78      | 55.1%       |
| 2026 | 13     | +$113.26      | 61.5%       |

- **2020 (pandemic year) is the single worst year**: 76 trades (highest trade count of any year - elevated volatility drove more crossovers), only 25% win rate, -$531 net. Consistent with the hypothesis's prediction that extreme volatility/gap risk hurts a next-bar-open, no-slippage-modeled backtest.
- **2023 and 2025 are the strongest years** (53-55% win rate, both net positive) - closer to the hypothesized trending-market performance.
- **2019, 2021, 2022 are all net losers** with win rates in the 35-43% range - three consecutive multi-year stretches of underperformance, not an isolated event.
- **2017 and 2026 have the best win rates** (61%+) but small sample sizes (18 and 13 trades respectively - 2026 is a partial year), so treat with caution.

### Maximum Drawdown Analysis

| Metric                           | Value                          | Notes                                                              |
| --------------------------------- | -------------------------------- | --------------------------------------------------------------------- |
| **Max Drawdown %**               | 11.76%                          | Peak-to-trough on the trade-close equity curve.                       |
| **Drawdown Start Date**          | 2019-01-09                      | Equity high-water mark before the drawdown began.                     |
| **Drawdown Trough Date**         | 2022-10-14                      | Lowest point of the drawdown.                                         |
| **Max Drawdown Duration**        | ~1,374 days (~3.8 years)        | From the 2019-01-09 peak to the 2022-10-14 trough.                    |
| **Recovery**                      | Not recovered by end of backtest | Final equity ($9,793.96) is still below the all-time high ($10,138.25) reached in early 2019. |
| **Number of Drawdown Episodes > 10%** | 2                            | Two separate stretches where drawdown exceeded 10% from a prior peak. |

**Analysis**: the equity curve is jagged, not smooth - the strategy spent roughly 3.8 of its 9 backtested years underwater relative to its early-2019 high, and had not recovered to that high-water mark by the end of the test period. This is a longer, deeper drawdown than a Sharpe of -0.13 alone conveys, and lines up with the losing 2019/2021/2022 years in the yearly breakdown above.

---

## 5. Key Insights & Validation Against Hypothesis

### Does MACD (+ HMM regime gate) Have a Baseline Edge on H1?

**No.** Against the criteria in [02_hypothesis.md](02_hypothesis.md):

- [ ] Win Rate > 55% — **FAIL** (42.86%)
- [ ] Profit Factor > 1.0 — **FAIL** (0.954)
- [ ] Sharpe Ratio > 0.5 — **FAIL** (-0.132)
- [x] Max Drawdown < 20% — **PASS** (11.76%)
- [ ] Trending periods >> Ranging periods — **NOT YET TESTED** (Section 3 requires regime-tagged trade data not yet exported)

**H₁ is rejected for the H1 timeframe as currently configured.** Under the null hypothesis in [02_hypothesis.md](02_hypothesis.md), a win rate of 42.86% and profit factor of 0.954 are both consistent with **no edge or a mild negative edge**, not with H₁'s predicted positive edge.

### Why?

- The exit-reason breakdown (Section 1) shows the losing edge is concentrated in the 150 trades (35.7%) that run the full 75-pip catastrophic stop. The `logic_regime_change` exit path (52% of trades) is itself mildly profitable, meaning the HMM regime filter is adding some value by cutting losers early - but it isn't preventing enough of the full-stop losses to make the strategy net profitable.
- The multi-year drawdown (2019 peak → 2022 trough, not recovered) suggests this isn't just noise from a handful of bad trades; 2019, 2021, and 2022 were each individually net-losing years.
- 2020's outsized trade count (76, ~1.8x the yearly average) and worst win rate (25%) point at elevated/gappy volatility as a specific driver of losses, consistent with the hypothesis's prediction that MACD lags during volatility spikes and news-driven moves.

### Unexpected Findings

- The hypothesis in [02_hypothesis.md](02_hypothesis.md) predicted MACD alone would struggle in ranging markets but succeed in trends; this run already includes the HMM trend-regime *gate* (entries require MACD + regime agreement, per `signals.generate_signals()`), not raw MACD. That the gated version still fails suggests either the regime filter isn't classifying trend/range accurately enough on H1, or H1 crossovers carry a negative edge even within HMM-labeled trend regimes - Section 3's regime-conditioned breakdown (once available) will help distinguish these.
- The catastrophic stop-loss (75 pips) and fixed 2:1 take-profit (150 pips) are wide relative to how most trades actually resolve: only 12.1% of trades reach the full TP and 35.7% reach the full SL, with the majority (52.1%) closing on a regime-change or opposite-signal exit well before either level. The realized average win (72.2 pips) is roughly half the nominal 150-pip TP.

---

## 6. Recommendations for H002 (Next Iteration)

Given the H1 baseline (with HMM regime gating already applied) fails on win rate, profit factor, and Sharpe:

1. **Regime-conditioned breakdown (prerequisite)**: carry the entry-time HMM `regime` value into the completed-trade log so Section 3 can actually be computed - needed to tell whether the negative edge is uniform or concentrated in specific regimes before choosing further filters.
2. **Investigate the catastrophic-stop-loss cohort specifically**: 150 trades (35.7%) account for all of the gross losses. Understanding what precedes a full-SL trade (regime at entry, volatility, time of day) is higher priority than broad parameter optimization.
3. **Session filter (H003 candidate)**: 2020's spike in trade count and worst-of-all-years win rate points at volatility-driven whipsaw; a liquidity-session filter may disproportionately help.
4. **Re-examine the fixed 75-pip SL / 150-pip TP**: most trades never reach either level (52.1% exit via regime-change/opposite-signal first), so the risk:reward ratio actually realized (72.2 avg win vs. -54.7 avg loss ≈ 1.3:1) is well short of the nominal 2:1 - tightening the SL or dynamically sizing it (ATR-based, H006) may reduce the catastrophic-stop cohort's damage without giving up much upside.
5. **Timeframe comparison (M15/H4)**: still outstanding: run the same MACD+HMM configuration on M15 and H4 (Section 2) before concluding MACD+HMM has no edge at all - the hypothesis specifically predicted H4 would outperform H1 due to reduced noise.

---

## Appendix: Trade Log

Full export: [../output/quant_analyzer.csv](../output/quant_analyzer.csv) (420 rows, Quant Analyzer Format #3 - `Ticket, Open Time, Action, Size, Symbol, Open Price, Stop Loss, Take Profit, Close Time, Close Price, Commission, Exit Reason, PL, Raw Money Profit, Swap`).

Biggest winner: 2023-08-21 → 2023-09-05 sell, +$49.92 (`take_profit`).
Biggest loser: 2019-08-16 → 2019-08-30 buy, -$26.40 (`catastrophic_stop_loss`).
