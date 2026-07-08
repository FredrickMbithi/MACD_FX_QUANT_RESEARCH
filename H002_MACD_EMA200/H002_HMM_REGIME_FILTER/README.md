# H002 HMM Regime Filter

## Objective

Test whether a Hidden Markov Model can identify latent EURUSD H4 market regimes where the frozen H001 MACD crossover strategy has materially better trade statistics than it has unconditionally.

The HMM is a classifier only. It does not generate entries, exits, stops, targets, or MACD parameters.

## Hypothesis

Null:

```text
Regime has no effect on MACD trade performance.
```

Alternative:

```text
At least one latent regime has higher out-of-sample expectancy and
profit factor than the unconditional MACD baseline.
```

Trade outcomes are measured in R-multiples:

```text
R = trade_price_profit / fixed_stop_distance
```

## Data

Use the same EURUSD H4 OHLC data as H001:

```text
time_utc, open, high, low, close, tick_volume
```

The raw data is not copied into this project. `config/h002.yaml` points back to the existing H001 raw CSV.

## Feature Set

Primary HMM features:

| Feature | Formula | Reason |
|---|---|---|
| `log_return_1` | `log(C_t) - log(C_{t-1})` | Stationary first difference of price. |
| `log_return_3` | `log(C_t) - log(C_{t-3})` | Short horizon directional pressure. |
| `realized_volatility_12` | `sqrt(sum(r_i^2))` | Recent second moment of returns. |
| `normalized_atr_14` | `ATR_14 / close` | Intrabar range adjusted for price level. |
| `ema_slope_24_6_atr` | `(EMA_t - EMA_{t-6}) / (6 * ATR_t)` | Directional drift per unit of volatility. |
| `adx_14` | Wilder trend-strength measure | Trend persistence without direction. |

MACD histogram is disabled by default because it is mechanically close to the entry signal. It may be used only as a secondary robustness experiment.

## Normalization

Each walk-forward fold fits a robust scaler on training features only:

```text
x_scaled = (x - training_median) / training_IQR
```

The test fold uses the training median and IQR. Full-sample scaling is forbidden.

## Model

Primary model:

```text
Gaussian HMM
covariance_type = diag
states = 2, 3, 4, 5
```

The model grid is selected by BIC/AIC and regime persistence, not by filtered strategy PnL.

GMM-HMM is intentionally not the first model because it has many more parameters and a higher false-discovery risk on a 393-trade baseline.

## Walk-Forward Protocol

Default:

```text
train window: 4 years
test window: 6 months
step: 6 months
```

For each fold:

1. Fit scaler on training features only.
2. Fit HMM grid on training features only.
3. Select the HMM by BIC/AIC and persistence.
4. Classify training bars with filtered probabilities.
5. Label training MACD trades by entry regime.
6. Select profitable regimes from training trades only.
7. Classify test bars with the fitted training HMM.
8. Label test MACD trades by entry regime.
9. Trade only original MACD entries whose entry regime passes the training filter.
10. Aggregate out-of-sample filtered results across folds.

## Causal Regime Labels

Regime probabilities use filtered probabilities:

```text
P(Z_t | X_1, ..., X_t)
```

They do not use smoothed probabilities:

```text
P(Z_t | X_1, ..., X_T)
```

Each trade is labeled with the regime active at entry only.

## Statistical Tests

The project reports:

```text
bootstrap expectancy confidence intervals
permutation test for mean R difference vs baseline
Benjamini-Hochberg corrected p-values
Cohen's d
profit factor
trade-level Sharpe
max drawdown in R
MAE and MFE in R
```

An attractive regime is not accepted unless it survives out-of-sample walk-forward validation and multiple-testing correction.

## Project Structure

```text
H002_HMM_REGIME_FILTER/
  config/
    h002.yaml
  data/
    raw/
    processed/
    features/
    trades/
  models/
  reports/
  src/
    baseline.py
    backtest.py
    config.py
    features.py
    hmm.py
    regime_analysis.py
    regime_classifier.py
    reporting.py
    statistics.py
    train.py
    validation.py
  tests/
  requirements-h002.txt
  README.md
```

## Module Responsibilities

`features.py`

Builds bar-level market-state features using only current and prior bars.

`hmm.py`

Fits Gaussian HMMs, calculates AIC/BIC, estimates expected state durations, and produces causal filtered state probabilities.

`train.py`

Runs the walk-forward experiment end to end. It is the main research entry point.

`regime_classifier.py`

Converts fitted HMM probabilities into bar-level regime labels and optional confirmation labels.

`regime_analysis.py`

Labels MACD trades by entry regime and computes per-regime trade diagnostics.

`backtest.py`

Applies the regime filter to the already-generated MACD trade log. It does not modify MACD logic.

`validation.py`

Defines walk-forward folds and fold splits.

`statistics.py`

Implements bootstrap intervals, permutation tests, effect sizes, and multiple-testing correction.

`reporting.py`

Writes reproducible Markdown and CSV reports.

## Run

Install H002-specific dependencies:

```bash
python -m pip install -r H002_HMM_REGIME_FILTER/requirements-h002.txt
```

Run the walk-forward research pipeline:

```bash
cd H002_HMM_REGIME_FILTER
PYTHONPATH=src python src/train.py config/h002.yaml
```

Outputs:

```text
data/features/h002_features.csv
data/processed/h002_bar_regimes.csv
data/trades/h002_labeled_trades.csv
data/trades/h002_filtered_trades.csv
reports/model_selection.csv
reports/walk_forward_folds.csv
reports/regime_summary.csv
reports/regime_significance.csv
reports/baseline_vs_filtered.csv
reports/h002_report.md
```

## Success Criteria

H002 is successful only if:

```text
filtered expectancy > baseline expectancy out of sample
filtered profit factor > baseline profit factor out of sample
drawdown is lower or economically acceptable
bootstrap CI for expectancy improvement excludes zero
permutation p-value remains significant after correction
results persist across walk-forward folds
results survive realistic transaction costs
profitable regimes have enough trades to be credible
```

If these conditions are not met, the correct conclusion is that H001 is not supported by this HMM design.

