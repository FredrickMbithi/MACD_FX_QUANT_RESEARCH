"""
signals.py
==========

HMM-regime-filtered TRIX signal generation.

This module detects TRIX zero-line or signal-line crosses and turns them
into a long/short stop-and-reverse position stream. A crossover only
becomes a tradable signal when it agrees with the current market regime,
where the regime is inferred from a 2-state Gaussian HMM fit on log
returns and rolling volatility derived from the close price (see
`compute_hmm_features`) -- not on the TRIX series itself, so the regime
filter is an independent read on the market rather than a restatement of
the same smoothed price transform TRIX already is: bullish crosses
require the HMM to be in its bullish state, bearish crosses require
it to be in its bearish state.

By default the regime is fit walk-forward (see
`fit_regime_hmm_walk_forward`): the HMM is refit periodically using a
rolling window of data available up to that point, so no bar's regime
label depends on future observations. A look-ahead-biased
`fit_regime_hmm_naive` is also provided, fitting once on the whole
series -- useful for quick exploratory analysis, but never for anything
a backtest's P&L depends on.

Requires `hmmlearn` (pip install hmmlearn).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

CLOSE_COL = "close"
SIGNAL_EVENT_COL = "signal"
REGIME_COL = "regime"
TRIX_COL = "trix"
TRIX_SIGNAL_COL = "trix_signal"

LOG_RETURN_COL = "log_return"
VOLATILITY_COL = "volatility"
HIGH_LOW_RANGE_COL = "high_low_range"
OPEN_CLOSE_RETURN_COL = "open_close_return"
VOLUME_ZSCORE_COL = "volume_zscore"

LONG_SIGNAL = "LONG"
SHORT_SIGNAL = "SHORT"

BULLISH_REGIME = "BULLISH"
BEARISH_REGIME = "BEARISH"

DEFAULT_HMM_N_ITER = 100
DEFAULT_TRAIN_WINDOW = 10_000
DEFAULT_VOL_WINDOW = 14
DEFAULT_COVARIANCE_TYPE = "diag"
DEFAULT_HMM_FEATURE_SET = "close_return_volatility"
DEFAULT_SIGNAL_SOURCE = "trix"

# Rule-of-thumb samples needed per (state x feature-dimension) before the
# early walk-forward fits are trusted to have a stable covariance estimate.
# Only used to emit a warning, never to change behavior.
MIN_SAMPLES_PER_STATE_DIM = 30

# NOTE: this used to be a hardcoded 1e-8 with no way to override it. That
# floor is several orders of magnitude below the natural variance of the
# (log_return, volatility) features (~1e-6), which let a single near-singular
# initial guess produce a spuriously large first-iteration log-likelihood.
# hmmlearn's ConvergenceMonitor.converged only checks
# `history[1] - history[0] < tol` -- it does not check the sign of the
# difference -- so that guess's inevitable correction on the next EM step
# (a large *decrease*) was misread as "converged" and training stopped after
# 2 iterations on every single walk-forward refit. Confirmed empirically:
# with min_covar=1e-8, every refit converged in exactly 2 iterations; with
# min_covar=1e-4 they run ~10 iterations with a monotonically increasing
# log-likelihood, which is what a real EM fit should look like. Do not lower
# this back toward 1e-8 without re-checking model.monitor_.history directly
# (not model.monitor_.converged, which is misleading -- see NOTE above).
DEFAULT_MIN_COVAR = 1e-4


def detect_bullish_zero_cross(series: pd.Series) -> pd.Series:
    """
    Return True where a series crosses from at-or-below zero to above zero.

    Uses `<=` on the "previous" side (rather than a strict `<`) so that a
    bar sitting exactly on zero for one tick, then moving above it, is
    treated as a cross -- consistent with the signal-line crossover logic
    in `generate_signals`, which uses the same `<=`/`>=` boundary on its
    "previous" side.
    """
    previous = series.shift(1)
    return (previous <= 0) & (series > 0)


def detect_bearish_zero_cross(series: pd.Series) -> pd.Series:
    """
    Return True where a series crosses from at-or-above zero to below zero.

    See `detect_bullish_zero_cross` for why the "previous" side uses `>=`
    rather than a strict `>`.
    """
    previous = series.shift(1)
    return (previous >= 0) & (series < 0)


def compute_hmm_features(
    df: pd.DataFrame,
    close_col: str = CLOSE_COL,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "tick_volume",
    vol_window: int = DEFAULT_VOL_WINDOW,
    feature_set: str = DEFAULT_HMM_FEATURE_SET,
) -> pd.DataFrame:
    """
    Derive the stationary features the regime HMM is fit on: log returns
    (directional momentum) and rolling realized volatility, both computed
    from the raw close price series rather than from the TRIX itself.

    Supported feature sets:
    - ``close_return_volatility``: 2D vector of close log return and its
        rolling volatility.
    - ``ohlcv_stationary``: 4D vector built from close return, normalized
        high-low range, open-close return, and volume z-score.

    Returns a DataFrame indexed like `df` but with the leading rows
    dropped wherever any feature is still NaN. Close returns are clipped
    at 5 standard deviations (using only data observable up to each bar)
    to prevent isolated price spikes from destabilizing the Gaussian HMM
    covariance matrix, without leaking future volatility into past bars.

    Raises `KeyError` if a column required by the chosen `feature_set` is
    missing from `df` -- this is deliberate for every column the feature
    set depends on (including `volume_col` for ``ohlcv_stationary``), so
    a missing column fails loudly here rather than silently producing an
    all-NaN feature (which `dropna()` would then turn into an empty
    feature frame, and downstream an empty/all-NaN regime with zero
    trades and no warning).
    """
    # NOTE: this only flags the *current* bar's close as bad. A non-positive
    # *previous* close also corrupts this row's log return (it's a ratio via
    # shift(1)), so both are checked -- otherwise the warning under-reports
    # exactly which rows are getting dropped as bad data.
    non_positive_current = df[close_col] <= 0
    non_positive_previous = df[close_col].shift(1) <= 0
    non_positive = non_positive_current | non_positive_previous.fillna(False)
    if non_positive.any():
        warnings.warn(
            f"compute_hmm_features: {int(non_positive.sum())} row(s) have a "
            f"non-positive {close_col!r} price in the current or previous "
            "bar; their log return will be dropped as bad data rather than "
            "passed to the HMM as inf/NaN.",
            stacklevel=2,
        )

    features = pd.DataFrame(index=df.index)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_return = pd.Series(np.log(df[close_col] / df[close_col].shift(1)), index=df.index)

    clean_returns = log_return.replace([np.inf, -np.inf], np.nan)

    # NOTE: clip bound must be causal (expanding, not full-sample) or the
    # clip applied to an early bar would depend on volatility observed at
    # bars that hadn't happened yet -- a look-ahead leak even though the
    # downstream HMM regime fit is walk-forward. `min_periods=vol_window`
    # means the first `vol_window` bars aren't clipped at all (matches the
    # rolling-volatility warmup already implied by DEFAULT_VOL_WINDOW).
    expanding_std = clean_returns.expanding(min_periods=vol_window).std()
    clean_returns = clean_returns.clip(lower=-5 * expanding_std, upper=5 * expanding_std)

    if feature_set == "close_return_volatility":
        features[LOG_RETURN_COL] = clean_returns
        features[VOLATILITY_COL] = features[LOG_RETURN_COL].rolling(window=vol_window).std()
    elif feature_set == "ohlcv_stationary":
        if open_col not in df.columns or high_col not in df.columns or low_col not in df.columns:
            raise KeyError(
                "compute_hmm_features: ohlcv_stationary requires open, high, and low columns"
            )
        if volume_col not in df.columns:
            # Previously this fell back to an all-NaN empty Series instead
            # of raising, which meant dropna() emptied the whole feature
            # frame, fit_regime_hmm_walk_forward saw n <= min_train_size
            # and returned an all-NaN regime, and generate_signals silently
            # produced zero trades -- no error, no warning. Fail loudly
            # instead, same as the open/high/low check above.
            raise KeyError(
                "compute_hmm_features: ohlcv_stationary requires a volume "
                f"column ({volume_col!r} not found)"
            )

        high = pd.to_numeric(df[high_col], errors="coerce")
        low = pd.to_numeric(df[low_col], errors="coerce")

        with np.errstate(divide="ignore", invalid="ignore"):
            open_close_return = pd.Series(
                np.log(df[close_col] / df[open_col]), index=df.index
            )

        range_ = pd.Series((high - low) / df[close_col], index=df.index)
        volume_series = pd.to_numeric(df[volume_col], errors="coerce")
        volume_log = pd.Series(np.log1p(volume_series), index=df.index)
        volume_mean = volume_log.rolling(window=vol_window).mean()
        volume_std = volume_log.rolling(window=vol_window).std()

        features[LOG_RETURN_COL] = clean_returns
        features[HIGH_LOW_RANGE_COL] = range_.replace([np.inf, -np.inf], np.nan)
        features[OPEN_CLOSE_RETURN_COL] = open_close_return.replace([np.inf, -np.inf], np.nan)
        features[VOLUME_ZSCORE_COL] = (volume_log - volume_mean) / volume_std
    else:
        raise ValueError(
            f"compute_hmm_features: unsupported feature_set {feature_set!r}"
        )

    return features.dropna()


def fit_regime_hmm_naive(
    features: pd.DataFrame,
    n_states: int = 2,
    random_state: int = 0,
    hmm_n_iter: int = DEFAULT_HMM_N_ITER,
    covariance_type: str = DEFAULT_COVARIANCE_TYPE,
    min_covar: float = DEFAULT_MIN_COVAR,
) -> tuple[GaussianHMM, pd.Series]:
    """
    Fit a Gaussian HMM on the *entire* feature series (log return and
    volatility, see `compute_hmm_features`) and decode the most likely
    hidden-state path via Viterbi. LOOK-AHEAD BIASED.
    """
    observations = features.to_numpy()

    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=hmm_n_iter,
        random_state=random_state,
        min_covar=min_covar,
    )
    model.fit(observations)
    hidden_states = model.predict(observations)

    bullish_state = int(np.argmax(model.means_[:, 0]))

    regime = pd.Series(
        np.where(hidden_states == bullish_state, BULLISH_REGIME, BEARISH_REGIME),
        index=features.index,
        name=REGIME_COL,
    )
    return model, regime


def fit_regime_hmm_walk_forward(
    features: pd.DataFrame,
    min_train_size: int = 60,
    refit_interval: int = 100,
    train_window: int | None = DEFAULT_TRAIN_WINDOW,
    n_states: int = 2,
    random_state: int = 0,
    hmm_n_iter: int = DEFAULT_HMM_N_ITER,
    covariance_type: str = DEFAULT_COVARIANCE_TYPE,
    min_covar: float = DEFAULT_MIN_COVAR,
) -> pd.Series:
    """
    Causal, walk-forward regime labels -- no future information leaks in.
    """
    values = features.to_numpy()
    n = len(values)
    n_dims = values.shape[1] if values.ndim > 1 else 1
    regime = pd.Series(index=features.index, dtype="object")

    # Soft check only (never changes behavior): with a low min_train_size
    # relative to n_states * n_dims, the earliest walk-forward fits are
    # working with very few samples per state-dimension, which can make
    # the first several covariance estimates noisy. train_window growing
    # over time mitigates this after the first few refits, but it's worth
    # flagging rather than silently trusting the early labels.
    min_recommended = MIN_SAMPLES_PER_STATE_DIM * n_states * n_dims
    if min_train_size < min_recommended:
        warnings.warn(
            f"fit_regime_hmm_walk_forward: min_train_size={min_train_size} is "
            f"low for a {n_states}-state HMM on {n_dims}-dimensional features "
            f"(~{min_train_size / (n_states * n_dims):.1f} samples per "
            f"state-dimension in the earliest fits). Consider raising "
            f"min_train_size toward {min_recommended} for more stable early "
            "covariance estimates, especially with feature_set='ohlcv_stationary'.",
            stacklevel=2,
        )

    if n <= min_train_size:
        return regime

    previous_label = None
    t = min_train_size
    while t < n:
        train_end = t + 1
        train_start = 0 if train_window is None else max(0, train_end - train_window)
        train = values[train_start:train_end]
        window_end = min(t + refit_interval, n)

        try:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type=covariance_type,
                n_iter=hmm_n_iter,
                random_state=random_state,
                min_covar=min_covar,
            )
            model.fit(train)
            hidden_states = model.predict(train)

            bullish_state = int(np.argmax(model.means_[:, 0]))
            label = BULLISH_REGIME if hidden_states[-1] == bullish_state else BEARISH_REGIME
        except (ValueError, np.linalg.LinAlgError, RuntimeError, FloatingPointError) as exc:
            # Scope is deliberate: these four cover the numerical failure
            # modes actually seen from hmmlearn/scipy during EM fitting on
            # pathological windows (singular/near-singular covariance,
            # Cholesky decomposition failures, numerical overflow in the
            # forward-backward pass). Anything outside this set (e.g. a
            # TypeError from a genuine programming bug) is intentionally
            # left to propagate and crash the run rather than being masked
            # by carrying the previous regime label forward.
            warnings.warn(
                f"fit_regime_hmm_walk_forward: HMM fit failed for the window "
                f"ending at feature-index {t} (covariance_type={covariance_type!r}): "
                f"{exc}. Carrying forward the previous regime label instead of "
                "crashing the run.",
                stacklevel=2,
            )
            label = previous_label

        if label is not None:
            regime.iloc[t:window_end] = label
        previous_label = label
        t = window_end

    return regime


def apply_regime_filter(
    regime: pd.Series,
    bullish_crossover: pd.Series,
    bearish_crossover: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Keep only crossovers that agree with the current HMM-inferred regime.
    """
    bullish_filtered = bullish_crossover & (regime == BULLISH_REGIME)
    bearish_filtered = bearish_crossover & (regime == BEARISH_REGIME)
    return bullish_filtered, bearish_filtered


def generate_positions(
    bullish_crossover: pd.Series,
    bearish_crossover: pd.Series,
) -> pd.Series:
    """
    Convert filtered crossover events into positions: 1 long, -1 short, 0 flat.
    """
    raw_position = pd.Series(index=bullish_crossover.index, dtype="float64")
    raw_position[bullish_crossover] = 1
    raw_position[bearish_crossover] = -1
    return raw_position.ffill().fillna(0).astype(int)


def generate_signals(
    df: pd.DataFrame,
    close_col: str = CLOSE_COL,
    trix_col: str = TRIX_COL,
    trix_signal_col: str = TRIX_SIGNAL_COL,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "tick_volume",
    vol_window: int = DEFAULT_VOL_WINDOW,
    feature_set: str = DEFAULT_HMM_FEATURE_SET,
    signal_source: str = DEFAULT_SIGNAL_SOURCE,
    n_states: int = 2,
    random_state: int = 0,
    causal: bool = True,
    min_train_size: int = 60,
    refit_interval: int = 100,
    train_window: int | None = DEFAULT_TRAIN_WINDOW,
    hmm_n_iter: int = DEFAULT_HMM_N_ITER,
    covariance_type: str = DEFAULT_COVARIANCE_TYPE,
    min_covar: float = DEFAULT_MIN_COVAR,
) -> pd.DataFrame:
    """
    Add HMM-regime, crossover, signal, and position columns to `df`.
    """
    result = df.copy()

    # TODO: signal_source is currently a no-op -- "trix" is the only
    # accepted value and every other value raises. This parameter is
    # scaffolding for a planned second signal source (e.g. a raw
    # zero-line-only mode, or a non-TRIX oscillator); if that's not
    # actually planned, remove the parameter rather than leave dead
    # surface area for callers to trip over.
    if signal_source != "trix":
        raise ValueError(f"generate_signals: unsupported signal_source {signal_source!r}")

    trix = result[trix_col]
    if trix_signal_col in result.columns:
        trix_signal = result[trix_signal_col]
        previous_trix = trix.shift(1)
        previous_signal = trix_signal.shift(1)
        raw_bullish_crossover = (previous_trix <= previous_signal) & (trix > trix_signal)
        raw_bearish_crossover = (previous_trix >= previous_signal) & (trix < trix_signal)
    else:
        raw_bullish_crossover = detect_bullish_zero_cross(trix)
        raw_bearish_crossover = detect_bearish_zero_cross(trix)

    hmm_features = compute_hmm_features(
        result,
        close_col=close_col,
        open_col=open_col,
        high_col=high_col,
        low_col=low_col,
        volume_col=volume_col,
        vol_window=vol_window,
        feature_set=feature_set,
    )

    if causal:
        regime = fit_regime_hmm_walk_forward(
            hmm_features,
            min_train_size=min_train_size,
            refit_interval=refit_interval,
            train_window=train_window,
            n_states=n_states,
            random_state=random_state,
            hmm_n_iter=hmm_n_iter,
            covariance_type=covariance_type,
            min_covar=min_covar,
        )
    else:
        _, regime = fit_regime_hmm_naive(
            hmm_features,
            n_states=n_states,
            random_state=random_state,
            hmm_n_iter=hmm_n_iter,
            covariance_type=covariance_type,
            min_covar=min_covar,
        )

    regime = regime.reindex(result.index)
    result[REGIME_COL] = regime

    bullish_crossover, bearish_crossover = apply_regime_filter(
        regime,
        raw_bullish_crossover,
        raw_bearish_crossover,
    )

    result["bullish_crossover"] = bullish_crossover
    result["bearish_crossover"] = bearish_crossover

    signal_events = pd.Series(index=result.index, dtype="object")
    signal_events[bullish_crossover] = LONG_SIGNAL
    signal_events[bearish_crossover] = SHORT_SIGNAL
    result[SIGNAL_EVENT_COL] = signal_events

    result["position"] = generate_positions(bullish_crossover, bearish_crossover)

    return result