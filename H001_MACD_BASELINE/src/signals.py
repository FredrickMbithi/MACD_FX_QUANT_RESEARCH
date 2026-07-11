"""
signals.py
"""
from __future__ import annotations
import logging
import warnings
import numpy as np
import pandas as pd
from hmmlearn import hmm

# Signal Constants
LONG_SIGNAL = 1
SHORT_SIGNAL = -1
FLAT_SIGNAL = 0

# Regime Constants
BULL_TREND_REGIME = 1
BEAR_TREND_REGIME = -1
CONSOLIDATION_REGIME = 0

# Diagnostics & Optimization Fallbacks
DEFAULT_OCCUPANCY_WARN_THRESHOLD = 0.01  # 1% minimum state presence
DEFAULT_TREND_REFIT_INTERVAL = 168        # Refit weekly (assuming hourly/H4 layout)
DEFAULT_TREND_TRAIN_WINDOW = 2190         # FIXED: 365 days * 6 bars/day = 1 year lookback for H4
DEFAULT_VOL_WINDOW = 20
DEFAULT_MIN_COVAR = 1e-3                  # prevents a state's variance collapsing to ~0
DEFAULT_INIT_SEEDS = (42, 7, 123, 2024, 99)  # multi-start EM search

DEFAULT_TREND_FEATURE_SET = ["returns", "volatility"]

logger = logging.getLogger("H001_MACD_BASELINE.signals")

def check_state_occupancy(model, X: np.ndarray, warn_threshold: float | None = None) -> float:
    """
    Calculates the proportion of data points assigned to the least-occupied state.
    Returns the minority fraction.
    """
    try:
        hidden_states = model.predict(X)
        unique, counts = np.unique(hidden_states, return_counts=True)
        total = len(hidden_states)

        # Account for any hidden states that got exactly 0 points assigned
        occupancies = {state: 0.0 for state in range(model.n_components)}
        for state, count in zip(unique, counts):
            occupancies[state] = count / total

        minority_frac = min(occupancies.values())

        if warn_threshold is not None and minority_frac < warn_threshold:
            logger.warning(
                "State collapse risk detected! Minority state occupancy is %.2f%% (Threshold: %.2f%%)",
                minority_frac * 100, warn_threshold * 100
            )
        return minority_frac
    except Exception as e:
        logger.error("Failed to calculate state occupancy: %s", str(e))
        return 0.0

def fit_trend_regime_walk_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_components: int = 3,
    refit_interval: int = DEFAULT_TREND_REFIT_INTERVAL,
    train_window: int = DEFAULT_TREND_TRAIN_WINDOW,
    warn_threshold: float = DEFAULT_OCCUPANCY_WARN_THRESHOLD,
    min_covar: float = DEFAULT_MIN_COVAR,
    init_seeds: tuple[int, ...] = DEFAULT_INIT_SEEDS,
) -> np.ndarray:
    """
    Executes a walk-forward optimization for the HMM regime model.

    Two changes vs. the original implementation, both aimed at the same root
    cause (a state getting deterministically starved to 0% occupancy):

    1. Features are standardized per training window before fitting. Diag-
       covariance GaussianHMM's k-means-based init is scale-sensitive; with
       unscaled returns/volatility, whichever feature has more spread
       dominates initialization and can strand a state with zero mass
       regardless of what the window's data actually looks like.
    2. Each refit tries a handful of random seeds and keeps the best
       (highest-likelihood) candidate that clears the occupancy bar, instead
       of relying on a single fixed seed that can lock in the same bad local
       optimum on every call.

    Still includes the original soft fallback: if nothing clears the bar,
    the previous stable model is retained and a rejection is logged - but
    now acceptances are logged too, so you can tell "stale for 6 bars" from
    "stale for the whole run" just by reading the log.
    """
    X = df[feature_cols].to_numpy()
    n_samples = len(df)
    regimes = np.zeros(n_samples, dtype=int)

    model = None
    scaler_mean = None
    scaler_std = None

    # Suppress internal convergence optimization prints from escaping to stderr
    warnings.filterwarnings("ignore", category=UserWarning, module="hmmlearn")

    for i in range(n_samples):
        # Time to refit the rolling window
        if i >= train_window and (i - train_window) % refit_interval == 0:
            start_idx = i - train_window
            train_data = X[start_idx:i]

            # Standardize per-window so no single feature (e.g. volatility
            # spikes) dominates KMeans init / diag-covariance EM.
            train_mean = train_data.mean(axis=0)
            train_std = train_data.std(axis=0)
            train_std = np.where(train_std < 1e-12, 1.0, train_std)  # guard constant columns
            train_data_scaled = (train_data - train_mean) / train_std

            best_candidate = None
            best_minority_frac = -1.0
            best_score = -np.inf

            for seed in init_seeds:
                candidate_model = hmm.GaussianHMM(
                    n_components=n_components,
                    covariance_type="diag",
                    n_iter=100,
                    min_covar=min_covar,
                    random_state=seed,
                )
                try:
                    candidate_model.fit(train_data_scaled)
                    minority_frac = check_state_occupancy(
                        candidate_model, train_data_scaled, warn_threshold=None
                    )
                    score = candidate_model.score(train_data_scaled)
                except (ValueError, np.linalg.LinAlgError) as e:
                    logger.error("Hard fit failure at bar %d (seed=%d): %s", i, seed, str(e))
                    continue

                clears_bar = minority_frac >= warn_threshold
                best_clears_bar = best_minority_frac >= warn_threshold
                is_better = (
                    (clears_bar and not best_clears_bar)
                    or (clears_bar and best_clears_bar and score > best_score)
                    or (not clears_bar and not best_clears_bar and minority_frac > best_minority_frac)
                )
                if is_better:
                    best_candidate = candidate_model
                    best_minority_frac = minority_frac
                    best_score = score

            if best_minority_frac >= warn_threshold:
                # Model is mathematically robust and states are fully populated
                model = best_candidate
                scaler_mean, scaler_std = train_mean, train_std
                logger.info(
                    "Accepted refit at bar %d: minority occupancy %.2f%%, log-lik %.2f (best of %d seeds)",
                    i, best_minority_frac * 100, best_score, len(init_seeds)
                )
            else:
                # Soft Fallback Triggered: Reject the unstable fit and retain the previous window's parameters
                logger.warning(
                    "Rejecting refit at bar %d due to state underpopulation "
                    "(best of %d seeds: %.2f%%, threshold: %.2f%%). Reusing previous stable model.",
                    i, len(init_seeds), best_minority_frac * 100, warn_threshold * 100
                )

        # Mapping the structural regimes if a validated model state exists
        if model is not None and scaler_mean is not None and scaler_std is not None:
            current_bar = X[i:i+1]
            current_bar_scaled = (current_bar - scaler_mean) / scaler_std
            try:
                predicted_state = model.predict(current_bar_scaled)[0]

                # Dynamic mapping of hidden states based on historical means of the returns feature
                # (Assumes 'returns' is the first feature in feature_cols). Scaling is a monotonic
                # per-feature transform, so this ordering is unaffected by standardization.
                means = model.means_[:, 0]
                sorted_states = np.argsort(means)

                if predicted_state == sorted_states[-1]:
                    regimes[i] = BULL_TREND_REGIME
                elif predicted_state == sorted_states[0]:
                    regimes[i] = BEAR_TREND_REGIME
                else:
                    regimes[i] = CONSOLIDATION_REGIME
            except Exception:
                regimes[i] = regimes[i-1] if i > 0 else CONSOLIDATION_REGIME
        else:
            # Seed with consolidation if no initial model training window has passed yet
            regimes[i] = CONSOLIDATION_REGIME

    return regimes

def generate_signals(
    df: pd.DataFrame,
    signal_source: str = "macd",
    regime_engine: str = "trend",
    trend_feature_set: list[str] | None = None,
    vol_window: int = DEFAULT_VOL_WINDOW,
    trend_refit_interval: int = DEFAULT_TREND_REFIT_INTERVAL,
    trend_train_window: int = DEFAULT_TREND_TRAIN_WINDOW,
) -> pd.DataFrame:
    """
    Generates execution entry signals combined with structural trend regime tracking.
    """
    if signal_source != "macd":
        raise ValueError(
            f"unsupported signal_source {signal_source!r} - only 'macd' is currently implemented"
        )

    signals_df = df.copy()
    signals_df["signal"] = FLAT_SIGNAL

    # Engineering required features for the HMM. Warm-up rows (the first
    # bar's undefined return, the first vol_window-1 bars' undefined
    # rolling std) are back-filled from the first valid observation rather
    # than zero-filled - fillna(0.0) would fabricate a "flat market" reading
    # for bars where volatility is simply unknown yet, and those fabricated
    # rows fall inside the very first HMM training window.
    if "returns" not in signals_df.columns:
        signals_df["returns"] = signals_df["close"].pct_change().bfill()
    if "volatility" not in signals_df.columns:
        signals_df["volatility"] = signals_df["returns"].rolling(window=vol_window).std().bfill()

    features = trend_feature_set if trend_feature_set is not None else DEFAULT_TREND_FEATURE_SET

    # Compute the robust walk-forward regimes
    if regime_engine == "trend":
        signals_df["regime"] = fit_trend_regime_walk_forward(
            signals_df, feature_cols=features, refit_interval=trend_refit_interval, train_window=trend_train_window
        )
    else:
        signals_df["regime"] = CONSOLIDATION_REGIME

    macd_vals = signals_df["macd"].to_numpy()
    macd_sig = signals_df["macd_signal"].to_numpy()
    regimes = signals_df["regime"].to_numpy()
    sig_out = np.zeros(len(signals_df), dtype=int)

    for i in range(1, len(signals_df)):
        macd_is_bullish = macd_vals[i] > macd_sig[i]
        macd_is_bearish = macd_vals[i] < macd_sig[i]

        macd_crossed_bullish = (macd_vals[i-1] <= macd_sig[i-1]) and macd_is_bullish
        macd_crossed_bearish = (macd_vals[i-1] >= macd_sig[i-1]) and macd_is_bearish

        regime_turned_bullish = (regimes[i-1] != BULL_TREND_REGIME) and (regimes[i] == BULL_TREND_REGIME)
        regime_turned_bearish = (regimes[i-1] != BEAR_TREND_REGIME) and (regimes[i] == BEAR_TREND_REGIME)

        # LOGIC: Buy if MACD triggers during a Bull trend, OR Bull trend starts while MACD is already triggered
        long_condition = (macd_crossed_bullish and regimes[i] == BULL_TREND_REGIME) or \
                         (regime_turned_bullish and macd_is_bullish)

        short_condition = (macd_crossed_bearish and regimes[i] == BEAR_TREND_REGIME) or \
                          (regime_turned_bearish and macd_is_bearish)

        if long_condition:
            sig_out[i] = LONG_SIGNAL
        elif short_condition:
            sig_out[i] = SHORT_SIGNAL

    signals_df["signal"] = sig_out
    return signals_df