from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_metric_ci(
    values: pd.Series,
    metric,
    samples: int = 10000,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> tuple[float, float]:
    clean = values.dropna().to_numpy(dtype=float)
    rng = np.random.default_rng(random_seed)
    estimates = np.empty(samples)
    for idx in range(samples):
        draw = rng.choice(clean, size=len(clean), replace=True)
        estimates[idx] = metric(pd.Series(draw))
    alpha = 1 - confidence
    return (
        float(np.nanquantile(estimates, alpha / 2)),
        float(np.nanquantile(estimates, 1 - alpha / 2)),
    )


def permutation_mean_difference(
    group: pd.Series,
    baseline: pd.Series,
    samples: int = 10000,
    random_seed: int = 42,
) -> dict:
    x = group.dropna().to_numpy(dtype=float)
    y = baseline.dropna().to_numpy(dtype=float)
    observed = float(x.mean() - y.mean())
    pooled = np.concatenate([x, y])
    rng = np.random.default_rng(random_seed)
    count = 0
    for _ in range(samples):
        shuffled = rng.permutation(pooled)
        diff = shuffled[: len(x)].mean() - shuffled[len(x) :].mean()
        if diff >= observed:
            count += 1
    return {"observed_difference": observed, "p_value": float((count + 1) / (samples + 1))}


def cohens_d(group: pd.Series, baseline: pd.Series) -> float:
    x = group.dropna().to_numpy(dtype=float)
    y = baseline.dropna().to_numpy(dtype=float)
    pooled_var = ((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2)
    if pooled_var <= 0:
        return np.nan
    return float((x.mean() - y.mean()) / np.sqrt(pooled_var))


def benjamini_hochberg(p_values: pd.Series, alpha: float = 0.05) -> pd.Series:
    p = p_values.astype(float).to_numpy()
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    previous = 1.0
    n = len(p)
    for rank, idx in enumerate(order[::-1], start=1):
        original_rank = n - rank + 1
        value = min(previous, p[idx] * n / original_rank)
        adjusted[idx] = value
        previous = value
    return pd.Series(adjusted, index=p_values.index)


def regime_significance_table(labeled_trades: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    stats_cfg = cfg["statistics"]
    baseline = labeled_trades["r_multiple"].astype(float)
    rows = []
    for regime, group in labeled_trades.dropna(subset=["entry_regime"]).groupby("entry_regime"):
        values = group["r_multiple"].astype(float)
        ci = bootstrap_metric_ci(
            values,
            metric=lambda s: s.mean(),
            samples=stats_cfg["bootstrap_samples"],
            confidence=stats_cfg["confidence_level"],
            random_seed=stats_cfg["random_seed"],
        )
        permutation = permutation_mean_difference(
            values,
            baseline,
            samples=stats_cfg["permutation_samples"],
            random_seed=stats_cfg["random_seed"],
        )
        rows.append(
            {
                "regime": int(regime),
                "trades": int(len(values)),
                "expectancy_r": float(values.mean()),
                "expectancy_ci_low": ci[0],
                "expectancy_ci_high": ci[1],
                "difference_vs_baseline": permutation["observed_difference"],
                "p_value": permutation["p_value"],
                "cohens_d": cohens_d(values, baseline),
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["p_value_bh"] = benjamini_hochberg(table["p_value"])
    return table


def bootstrap_mean_difference(
    candidate: pd.Series,
    benchmark: pd.Series,
    samples: int = 10000,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> dict:
    candidate_values = candidate.dropna().to_numpy(dtype=float)
    benchmark_values = benchmark.dropna().to_numpy(dtype=float)
    if len(candidate_values) == 0 or len(benchmark_values) == 0:
        return {
            "difference": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_difference_le_zero": np.nan,
        }

    rng = np.random.default_rng(random_seed)
    differences = np.empty(samples)
    for idx in range(samples):
        candidate_draw = rng.choice(candidate_values, size=len(candidate_values), replace=True)
        benchmark_draw = rng.choice(benchmark_values, size=len(benchmark_values), replace=True)
        differences[idx] = candidate_draw.mean() - benchmark_draw.mean()

    alpha = 1 - confidence
    return {
        "difference": float(candidate_values.mean() - benchmark_values.mean()),
        "ci_low": float(np.nanquantile(differences, alpha / 2)),
        "ci_high": float(np.nanquantile(differences, 1 - alpha / 2)),
        "p_difference_le_zero": float((differences <= 0).mean()),
    }


def strategy_significance_table(
    baseline_trades: pd.DataFrame,
    filtered_trades: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    stats_cfg = cfg["statistics"]
    keys = ["entry_time", "exit_time", "direction", "entry_price", "exit_price", "fold_id"]

    if filtered_trades.empty:
        excluded = baseline_trades.copy()
    else:
        filtered_keys = set(map(tuple, filtered_trades[keys].astype(str).to_numpy()))
        excluded_mask = ~baseline_trades[keys].astype(str).apply(tuple, axis=1).isin(filtered_keys)
        excluded = baseline_trades[excluded_mask].copy()

    comparisons = [
        ("filtered_vs_baseline_all", baseline_trades),
        ("filtered_vs_excluded_only", excluded),
    ]
    rows = []
    for name, benchmark in comparisons:
        diff = bootstrap_mean_difference(
            filtered_trades["r_multiple"],
            benchmark["r_multiple"],
            samples=stats_cfg["bootstrap_samples"],
            confidence=stats_cfg["confidence_level"],
            random_seed=stats_cfg["random_seed"],
        )
        rows.append(
            {
                "comparison": name,
                "filtered_trades": int(len(filtered_trades)),
                "benchmark_trades": int(len(benchmark)),
                "filtered_expectancy_r": float(filtered_trades["r_multiple"].mean()) if len(filtered_trades) else np.nan,
                "benchmark_expectancy_r": float(benchmark["r_multiple"].mean()) if len(benchmark) else np.nan,
                **diff,
            }
        )
    return pd.DataFrame(rows)
