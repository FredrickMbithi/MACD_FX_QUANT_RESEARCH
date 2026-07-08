from __future__ import annotations

import numpy as np
import pandas as pd

from hmm import filtered_state_probabilities


def classify_bars(model, scaled_features: pd.DataFrame, timestamps: pd.Series, timestamp_col: str = "time_utc") -> pd.DataFrame:
    probabilities = filtered_state_probabilities(model, scaled_features)
    state = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    result = pd.DataFrame({timestamp_col: pd.to_datetime(timestamps).to_numpy()})
    result["regime"] = state
    result["regime_probability"] = confidence
    for idx in range(probabilities.shape[1]):
        result[f"regime_probability_{idx}"] = probabilities[:, idx]
    return result


def add_confirmed_regime(
    regimes: pd.DataFrame,
    confirmation_bars: int = 1,
    regime_col: str = "regime",
    output_col: str = "confirmed_regime",
) -> pd.DataFrame:
    result = regimes.copy()
    if confirmation_bars <= 1:
        result[output_col] = result[regime_col]
        return result

    confirmed = []
    values = result[regime_col].to_numpy()
    for idx, state in enumerate(values):
        start = max(0, idx - confirmation_bars + 1)
        window = values[start : idx + 1]
        confirmed.append(state if len(window) == confirmation_bars and np.all(window == state) else np.nan)
    result[output_col] = confirmed
    return result
