# Research Question

Determine whether a basic MACD crossover strategy has a statistically meaningful edge before introducing any filters or enhancements.

## Purpose

Understand what MACD is before implementing it.

### Ordered Notes

1. Who developed MACD?
   Answer: Gerald Appel developed the MACD in the late 1970s. Thomas Aspray later added the histogram component in 1986.

2. What problem was MACD designed to solve?
   Answer: It was designed to identify underlying price trends and measure momentum despite market noise and volatility.

3. What does the MACD line represent?
   Answer: The MACD line represents the difference between a fast EMA and a slow EMA, which shows momentum changes in price.

4. What does the Signal line represent?
   Answer: The Signal line is a smoothed average of the MACD line and is used as a trigger for trading signals.

5. What does the Histogram represent?
   Answer: The Histogram represents the difference between the MACD line and the Signal line, showing whether momentum is accelerating or decelerating.

6. Why is the zero line important?
   Answer: The zero line marks the boundary between bullish and bearish momentum and helps show when the short-term and long-term averages are in balance.

7. Is MACD a trend indicator, momentum indicator, or both?
   Answer: MACD is both a trend-following indicator and a momentum indicator.

8. What are the commonly cited strengths?
   Answer: Common strengths include trend identification, momentum confirmation, divergence signals, and clear visual interpretation through the histogram.

9. What are the commonly cited weaknesses?
   Answer: Common weaknesses include lag, false signals in sideways markets, and the lack of fixed overbought or oversold levels.

10. What does existing research say about its effectiveness?
    Answer: Research suggests MACD works best when parameters are optimized for a specific market and timeframe, and it often performs better when combined with other filters rather than used alone.

## What is this hypothesis?

A basic MACD crossover strategy captures changes in market momentum and is expected to perform reasonably well during trending markets but poorly during ranging markets due to frequent false crossover signals.

### 1. Research Question

What market inefficiency or behavior am I trying to exploit?
Why should this strategy have an edge?
What assumptions am I making?
Under what market conditions should this strategy work?

### 2. Data Quality

Is my data complete?
Are there missing or duplicate bars?
Is the timestamp correct?
Is there look-ahead bias?
Is there survivorship bias?
Is the data representative of real trading?

### 3. Indicator Validation

Are the indicators calculated correctly?
Do the values match a trusted platform (e.g., MT4)?
Are the indicators using only historical data?
Are there any off-by-one errors?

### 4. Signal Logic

Does every signal satisfy the intended rules?
Are any valid signals being missed?
Are false signals being generated?
Does the visual chart match the generated signals?
Are long and short signals mutually exclusive?

### 5. Entry Logic

When exactly is a trade entered?
Is the entry executable in real markets?
Is there any look-ahead bias?
Does every signal produce one trade?
Are duplicate entries prevented?

### 6. Exit Logic

Why does the trade exit?
Is the opposite-signal exit logically timed?
Should exits be signal-driven, fixed, or dynamic?
What happens when a reversal signal appears immediately after entry?
Should trades have a maximum holding time?

### 7. Risk Management

Is uncapped signal-to-signal risk too wide?
Is risk consistent across trades?
Is the strategy dependent on a specific exit rule?
What happens if the exit rule changes?
Would ATR-based stops perform better?
Would trailing stops improve results?

### 8. Trade Statistics

How many trades are generated?
Is the sample size large enough?
What is the win rate?
What is the average winner?
What is the average loser?
What is the expectancy per trade?
What is the profit factor?
What is the average trade duration?

### 9. Equity Performance

Is the equity curve smooth?
Does growth depend on only a few trades?
How severe are drawdowns?
How long is the recovery after drawdowns?
Is the strategy consistently profitable over time?

### 10. Robustness

Does the strategy still work if MACD parameters change?
Does it survive small parameter variations?
Does it overfit the data?
Which parameters matter most?
Which parameters have little effect?

### 11. Market Regimes

Does the strategy perform better in trends?
Does it fail in ranging markets?
Does volatility affect performance?
Which market regimes are most profitable?
Which regimes should be avoided?

### 12. Time Analysis

Does performance change over different years?
Is the strategy stable across time periods?
Are there periods where it consistently loses?
Does market evolution affect results?

### 13. Cost Analysis

How sensitive is the strategy to spread?
How sensitive is it to commissions?
How much slippage can it tolerate?
Is it still profitable under realistic trading costs?

### 14. Generalization

Does the strategy work on other currency pairs?
Does it work on different timeframes?
Does it work on other asset classes?
Is the edge specific to one market?

### 15. Forward Testing

Does live performance resemble backtest performance?
Are signals generated correctly in real time?
Are fills close to expected prices?
Does execution introduce unexpected issues?
Does the strategy maintain its edge out of sample?

### 16. Final Evaluation

Was the original hypothesis supported?
What evidence supports the conclusion?
What are the strategy's strengths?
What are its weaknesses?
What assumptions proved incorrect?
What should be changed next?
Is the strategy ready for further development, forward testing, or should it be discarded?
