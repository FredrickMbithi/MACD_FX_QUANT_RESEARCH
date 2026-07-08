from __future__ import annotations

import pandas as pd

from src.features import RobustScaler, log_return
from src.validation import build_walk_forward_folds


def test_log_return_uses_past_data_only():
    close = pd.Series([1.0, 1.1, 1.21])
    returns = log_return(close, periods=1)
    assert pd.isna(returns.iloc[0])
    assert round(returns.iloc[1], 10) == round(returns.iloc[2], 10)


def test_robust_scaler_uses_training_distribution():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    test = pd.DataFrame({"x": [100.0]})
    scaler = RobustScaler.fit(train)
    transformed = scaler.transform(test)
    assert transformed.iloc[0, 0] == 98.0


def test_walk_forward_folds_do_not_overlap():
    timestamps = pd.date_range("2016-01-01", "2022-01-01", freq="D")
    folds = build_walk_forward_folds(pd.Series(timestamps), train_years=3, test_months=6, step_months=6)
    assert folds
    for fold in folds:
        assert fold.train_end == fold.test_start
        assert fold.train_start < fold.train_end < fold.test_end
