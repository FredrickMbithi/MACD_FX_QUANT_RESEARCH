from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TIMESTAMP_COL = "time_utc"
OPEN_COL = "open"
HIGH_COL = "high"
LOW_COL = "low"
CLOSE_COL = "close"
VOLUME_COL = "tick_volume"


@dataclass(frozen=True)
class RobustScaler:
    median: pd.Series
    iqr: pd.Series

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "RobustScaler":
        median = frame.median(numeric_only=True)
        q75 = frame.quantile(0.75, numeric_only=True)
        q25 = frame.quantile(0.25, numeric_only=True)
        iqr = (q75 - q25).replace(0.0, np.nan)
        return cls(median=median, iqr=iqr)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        scaled = (frame - self.median) / self.iqr
        return scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def fit_transform(cls, frame: pd.DataFrame) -> pd.DataFrame:
        raise TypeError("Use RobustScaler.fit(frame).transform(frame) to make train-only scaling explicit.")


def parse_ohlc(
    raw: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COL,
    price_cols: tuple[str, ...] = (OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL),
) -> pd.DataFrame:
    df = raw.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col).drop_duplicates(timestamp_col)
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=list(price_cols)).reset_index(drop=True)


def log_return(close: pd.Series, periods: int = 1) -> pd.Series:
    return np.log(close).diff(periods)


def realized_volatility(close: pd.Series, window: int) -> pd.Series:
    returns = log_return(close)
    return np.sqrt((returns**2).rolling(window=window, min_periods=window).sum())


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    previous_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    return true_range(high, low, close).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def normalized_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    return average_true_range(high, low, close, window) / close


def ema_slope_normalized(close: pd.Series, high: pd.Series, low: pd.Series, span: int, lag: int) -> pd.Series:
    ema = close.ewm(span=span, adjust=False).mean()
    atr = average_true_range(high, low, close, span)
    return (ema - ema.shift(lag)) / (lag * atr)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    atr = average_true_range(high, low, close, window)
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def macd_histogram(close: pd.Series, fast_span: int = 12, slow_span: int = 26, signal_span: int = 9) -> pd.Series:
    fast = close.ewm(span=fast_span, adjust=False).mean()
    slow = close.ewm(span=slow_span, adjust=False).mean()
    macd_line = fast - slow
    signal = macd_line.ewm(span=signal_span, adjust=False).mean()
    return macd_line - signal


def build_feature_frame(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    data_cfg = cfg["data"]
    feature_cfg = cfg["features"]

    timestamp_col = data_cfg.get("timestamp_col", TIMESTAMP_COL)
    high_col = data_cfg.get("high_col", HIGH_COL)
    low_col = data_cfg.get("low_col", LOW_COL)
    close_col = data_cfg.get("close_col", CLOSE_COL)

    df = parse_ohlc(raw, timestamp_col=timestamp_col)
    close = df[close_col]
    high = df[high_col]
    low = df[low_col]

    features = pd.DataFrame({timestamp_col: df[timestamp_col]})

    for window in feature_cfg.get("return_windows", [1, 3]):
        features[f"log_return_{window}"] = log_return(close, periods=window)

    rv_window = int(feature_cfg.get("realized_vol_window", 12))
    atr_window = int(feature_cfg.get("atr_window", 14))
    ema_span = int(feature_cfg.get("ema_slope_span", 24))
    ema_lag = int(feature_cfg.get("ema_slope_lag", 6))
    adx_window = int(feature_cfg.get("adx_window", 14))

    features[f"realized_volatility_{rv_window}"] = realized_volatility(close, rv_window)
    features[f"normalized_atr_{atr_window}"] = normalized_atr(high, low, close, atr_window)
    features[f"ema_slope_{ema_span}_{ema_lag}_atr"] = ema_slope_normalized(close, high, low, ema_span, ema_lag)
    features[f"adx_{adx_window}"] = adx(high, low, close, adx_window)

    if feature_cfg.get("include_macd_histogram", False):
        base = cfg["baseline"]
        hist = macd_histogram(
            close,
            fast_span=base["macd_fast_span"],
            slow_span=base["macd_slow_span"],
            signal_span=base["macd_signal_span"],
        )
        atr = average_true_range(high, low, close, atr_window)
        features["macd_histogram_atr"] = hist / atr

    warmup = int(feature_cfg.get("warmup_bars", 60))
    selected = [timestamp_col] + list(feature_cfg["selected"])
    clean = features[selected].iloc[warmup:].replace([np.inf, -np.inf], np.nan).dropna()
    return clean.reset_index(drop=True)


def feature_matrix(features: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    return features[list(cfg["features"]["selected"])].astype(float)
