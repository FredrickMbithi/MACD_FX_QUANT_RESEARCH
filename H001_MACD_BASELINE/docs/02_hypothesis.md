# Hypothesis

## Statement

A basic MACD (12, 26, 9) crossover strategy captures changes in market momentum and generates a statistically meaningful edge—defined as win rate significantly above random (>50%) and positive expectancy—during trending market regimes, but will produce whipsaw losses and negative expectancy during range-bound or choppy market conditions due to frequent false crossover signals.

## Formal Statistical Hypotheses

### Null Hypothesis (H₀)

The MACD crossover strategy produces zero edge. Win rate equals 50% (random). Profit factor ≤ 1.0. Mean trade returns are not significantly different from zero.

**Mathematically:**

- H₀: P(win) = 0.5
- H₀: PF (profit factor) ≤ 1.0
- H₀: E[return] = 0 (expectancy is zero)

### Alternative Hypothesis (H₁)

The MACD crossover strategy produces a positive, statistically significant edge. Win rate > 50%. Profit factor > 1.0. Mean trade returns are significantly positive.

**Mathematically:**

- H₁: P(win) > 0.5
- H₁: PF > 1.0
- H₁: E[return] > 0 (expectancy is positive)

## Why This Hypothesis?

From the research questions ([01_research_question.md](01_research_question.md)):

1. **MACD is a dual-purpose indicator**: It measures both trend _and_ momentum. On trending bars, recent prices diverge from the long-term trend faster than the trend itself changes—the fast EMA pulls away from the slow EMA, causing MACD to cross its signal line at points where momentum is _changing_, which often coincides with directional moves.

2. **MACD has documented weaknesses**: Research confirms MACD lags price, generates false signals in choppy markets, and works best when combined with filters. A baseline test of _raw_ MACD without filters should expose these weaknesses clearly.

3. **The strategy is intentionally simple**: No regime filters, no volatility adjustments, no entry optimization. This baseline establishes what MACD alone can do, so we know what value future enhancements actually add.

## Expected Edge / Edge Conditions

### When We Expect an Edge (Trending Markets)

- **Price is in a clear uptrend**: Lows are rising, highs are rising, recent candles close above prior candles.
  - Mechanism: MACD will cross above signal repeatedly on pullbacks within the trend, and most of these crossovers mark bounce points in the direction of the larger move.
  - Expected win rate: 55–65% on strong trending days.
  - Timeframe: H4 and H1 will show more consistent edges than M15 due to reduced noise.

- **Price is in a clear downtrend**: Highs are falling, lows are falling, recent candles close below prior candles.
  - Mechanism: MACD will cross below signal on bounce-backs against the trend; most SHORT signals mark turning points downward.
  - Expected win rate: 55–65% on strong trending days.

- **Volatility is elevated and directional**: Large moves with few reversals.
  - Mechanism: MACD momentum is large relative to noise; crossovers are genuine trend changes, not noise.
  - Expected ROI per trade: Larger due to extended trending moves.

### When We Expect NO Edge (Ranging Markets)

- **Price is range-bound**: Lows cluster, highs cluster, new extremes are infrequent.
  - Mechanism: MACD oscillates around zero without a persistent bias. Crossovers happen frequently because momentum keeps reversing within the range. Each crossover appears to signal a move, but the opposite crossover closes it before a meaningful trend develops.
  - Expected win rate: ~45–50% (worse than random, after accounting for spread/slippage).
  - Expected loss: Whipsaw losses dominate; profit factor < 1.0.

- **Volatility is very low**: Price drifts sideways with tiny moves.
  - Mechanism: MACD histogram is very small; crossovers are marginal (barely distinguishable from flat). Spread and slippage cost more than the average move makes.
  - Expected ROI: Negative in most cases.

- **Choppy/noisy price action**: Price oscillates in both directions with no clear bias.
  - Mechanism: MACD is whipsawed: a bullish crossover triggers a LONG, price reverses, and a bearish crossover closes/reverses the trade soon after.
  - Expected win rate: ~45% (losing edge).
  - Most common in low-liquidity sessions (e.g., Tokyo early morning) or illiquid pairs.

## Market Conditions Where Strategy Works / Doesn't Work

### Works Best

| Condition                      | Why                                                           | Expected Outcome                      |
| ------------------------------ | ------------------------------------------------------------- | ------------------------------------- |
| **Strong trending market**     | Momentum is directional; MACD leads pullback reversals.       | Win rate 55–65%, PF 1.2–1.5           |
| **H4 timeframe**               | Noise is filtered out; false signals are rare.                | Win rate 50–58%, fewer small whipsaws |
| **Directional news flow**      | Clear catalyst behind the trend; momentum extends.            | Win rate 55–70%, large wins           |
| **Pre-breakout consolidation** | MACD is flat before the breakout; first crossover catches it. | Win rate 60–70% on breakouts          |
| **Low-spread environment**     | EUR/USD at tight spreads minimizes friction.                  | Cost per trade: 1–2 pips vs 5–10 pips |

### Works Poorly

| Condition                                | Why                                                         | Expected Outcome                                     |
| ---------------------------------------- | ----------------------------------------------------------- | ---------------------------------------------------- |
| **Ranging / choppy market**              | MACD whipsaws; every crossover reverses quickly.            | Win rate 40–48%, PF 0.7–0.9                          |
| **After major reversals**                | MACD lags; by the time it signals, the move is almost over. | Win rate 45–50%, small wins/large losses             |
| **Low volatility / tight consolidation** | Moves are tiny; spread and slippage exceed average profit.  | Negative expectancy                                  |
| **News-driven volatility spikes**        | MACD lags; gaps before the opposite signal appears.         | Slippage spikes; actual results differ from backtest |
| **Overnight / illiquid sessions**        | Wide spreads, low liquidity, fast reversals.                | Execution worse than simulated; PF degrades          |

## Reference to Research Questions

This hypothesis directly addresses the key research questions from [01_research_question.md](01_research_question.md):

1. **"What market inefficiency am I trying to exploit?"**
   - Answer: Momentum continuation on pullbacks within trends. Markets do not reverse instantaneously; price pauses (pullback) before continuing the larger move. MACD crossovers often mark these pause endpoints.

2. **"Why should this strategy have an edge?"**
   - Answer: In trending markets, the fast EMA (12-period) reacts faster than the slow EMA (26-period) to recent price changes. When momentum shifts, the crossover often marks the end of a pullback and the resumption of the larger trend. If this happens more often than random, a statistical edge exists.
   - Caveat: This edge should _disappear_ in ranging/choppy markets, which is the test that validates or refutes the hypothesis.

3. **"What assumptions am I making?"**
   - Assumption 1: MACD crossovers indicate genuine momentum changes, not noise.
   - Assumption 2: Trending markets exist frequently enough to generate enough trades for statistical significance.
   - Assumption 3: Next-bar entry and opposite-signal exit prices are executable (no slippage/gaps in backtest).
   - Assumption 4: Historical data is representative of future market behavior.
   - **All four assumptions will be tested during execution.py runs and validated in findings.**

4. **"Under what market conditions should this strategy work?"**
   - Answer: Trending markets, high volatility, liquid sessions, strong directional news flow (see "Works Best" table above).

## Edge Validation Criteria

After running the backtest, H₁ is accepted if **ALL** of the following are true:

1. **Win rate > 55%** (meaningfully above 50%, accounting for small sample noise)
2. **Profit Factor > 1.0** (total wins > total losses on a gross basis)
3. **Sharpe Ratio > 0.5** (risk-adjusted returns are positive)
4. **Drawdown < 20%** (the strategy doesn't experience catastrophic losing streaks)
5. **Performance in trending periods >> performance in ranging periods** (edge is regime-dependent, as hypothesized)

If any of these are false, we reject H₁ and improve the strategy with filters/refinements (H002, H003, etc.).
