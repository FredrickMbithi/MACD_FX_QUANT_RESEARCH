from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

from baseline import build_baseline_trades
from backtest import apply_regime_filter, compare_baseline_filtered, select_profitable_regimes
from config import load_config, resolve_path
from features import RobustScaler, build_feature_frame, feature_matrix
from hmm import model_selection_grid, save_fit, select_model
from regime_analysis import add_mae_mfe, add_trade_r_multiples, label_trades_by_entry_regime, summarize_by_regime
from regime_classifier import add_confirmed_regime, classify_bars
from reporting import write_markdown_report
from statistics import regime_significance_table, strategy_significance_table
from validation import build_walk_forward_folds, split_by_fold


def build_fold_stability_table(fold_comparison: pd.DataFrame) -> pd.DataFrame:
    if fold_comparison.empty:
        return pd.DataFrame()

    rows = []
    for fold_id, group in fold_comparison.groupby("fold_id"):
        baseline = group[group["strategy"] == "baseline"].iloc[0]
        filtered = group[group["strategy"] == "hmm_filtered"].iloc[0]
        filtered_expectancy = filtered["expectancy_r"]
        baseline_expectancy = baseline["expectancy_r"]
        rows.append(
            {
                "fold_id": int(fold_id),
                "baseline_trades": int(baseline["trades"]),
                "filtered_trades": int(filtered["trades"]),
                "baseline_expectancy_r": baseline_expectancy,
                "filtered_expectancy_r": filtered_expectancy,
                "delta_expectancy_r": filtered_expectancy - baseline_expectancy
                if pd.notna(filtered_expectancy)
                else pd.NA,
                "baseline_total_r": baseline["total_r"],
                "filtered_total_r": filtered["total_r"],
                "filtered_traded": bool(filtered["trades"] > 0),
                "filtered_outperformed_expectancy": bool(
                    pd.notna(filtered_expectancy) and filtered_expectancy > baseline_expectancy
                ),
                "filtered_positive_total_r": bool(filtered["total_r"] > 0),
                "allowed_regimes": filtered["allowed_regimes"],
            }
        )

    table = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "fold_id": "summary",
                "baseline_trades": int(table["baseline_trades"].sum()),
                "filtered_trades": int(table["filtered_trades"].sum()),
                "baseline_expectancy_r": pd.NA,
                "filtered_expectancy_r": pd.NA,
                "delta_expectancy_r": pd.NA,
                "baseline_total_r": float(table["baseline_total_r"].sum()),
                "filtered_total_r": float(table["filtered_total_r"].sum()),
                "filtered_traded": int(table["filtered_traded"].sum()),
                "filtered_outperformed_expectancy": int(table["filtered_outperformed_expectancy"].sum()),
                "filtered_positive_total_r": int(table["filtered_positive_total_r"].sum()),
                "allowed_regimes": "",
            }
        ]
    )
    return pd.concat([table, summary], ignore_index=True)


def run_research(config_path: str = "config/h002.yaml") -> None:
    cfg = load_config(config_path)
    project_root = Path(config_path).resolve().parent.parent
    raw_path = resolve_path(config_path, cfg["data"]["raw_csv_path"])
    raw = pd.read_csv(raw_path)

    features = build_feature_frame(raw, cfg)
    baseline_trades, signal_bars = build_baseline_trades(raw, cfg)
    trades = add_trade_r_multiples(baseline_trades, cfg)
    timestamp_col = cfg["data"]["timestamp_col"]
    trades = add_mae_mfe(trades, raw, cfg, timestamp_col=timestamp_col)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])

    wf = cfg["walk_forward"]
    folds = build_walk_forward_folds(
        features[timestamp_col],
        train_years=wf["train_years"],
        test_months=wf["test_months"],
        step_months=wf["step_months"],
    )

    model_rows = []
    train_labeled_frames = []
    test_labeled_frames = []
    filtered_frames = []
    regime_frames = []
    fold_summaries = []

    for fold in folds:
        train_features, test_features = split_by_fold(features, fold, timestamp_col=timestamp_col)
        if train_features.empty or test_features.empty:
            continue

        x_train = feature_matrix(train_features, cfg)
        x_test = feature_matrix(test_features, cfg)
        scaler = RobustScaler.fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        selection = model_selection_grid(x_train_scaled, cfg)
        selection_export = selection.drop(columns=["fit"]).copy()
        selection_export["fold_id"] = fold.fold_id
        model_rows.append(selection_export)

        fit = select_model(selection, min_persistence=cfg["hmm"]["min_state_persistence"])
        save_fit(fit, project_root / cfg["output"]["model_dir"] / f"h002_gaussian_hmm_fold_{fold.fold_id:02d}.pkl")

        train_regimes = classify_bars(fit.model, x_train_scaled, train_features[timestamp_col], timestamp_col=timestamp_col)
        test_regimes = classify_bars(fit.model, x_test_scaled, test_features[timestamp_col], timestamp_col=timestamp_col)
        train_regimes = add_confirmed_regime(train_regimes, confirmation_bars=cfg["filter"]["confirmation_bars"])
        test_regimes = add_confirmed_regime(test_regimes, confirmation_bars=cfg["filter"]["confirmation_bars"])
        train_regimes["fold_id"] = fold.fold_id
        test_regimes["fold_id"] = fold.fold_id
        train_regimes["sample"] = "train"
        test_regimes["sample"] = "test"
        regime_frames.extend([train_regimes, test_regimes])

        train_trade_mask = (trades["entry_time"] >= fold.train_start) & (trades["entry_time"] < fold.train_end)
        test_trade_mask = (trades["entry_time"] >= fold.test_start) & (trades["entry_time"] < fold.test_end)
        train_trades = trades[train_trade_mask].copy()
        test_trades = trades[test_trade_mask].copy()

        train_labeled = label_trades_by_entry_regime(train_trades, train_regimes, timestamp_col=timestamp_col)
        train_labeled["fold_id"] = fold.fold_id
        train_labeled["sample"] = "train"
        train_summary = summarize_by_regime(train_labeled)
        allowed_regimes = select_profitable_regimes(train_summary, cfg)

        test_labeled = label_trades_by_entry_regime(test_trades, test_regimes, timestamp_col=timestamp_col)
        test_labeled["fold_id"] = fold.fold_id
        test_labeled["sample"] = "test"
        test_labeled["allowed_regimes"] = ",".join(map(str, allowed_regimes))
        filtered = apply_regime_filter(test_labeled, allowed_regimes, cfg)
        filtered["fold_id"] = fold.fold_id

        fold_summary = compare_baseline_filtered(test_labeled, filtered)
        fold_summary["fold_id"] = fold.fold_id
        fold_summary["train_start"] = fold.train_start
        fold_summary["train_end"] = fold.train_end
        fold_summary["test_start"] = fold.test_start
        fold_summary["test_end"] = fold.test_end
        fold_summary["selected_n_states"] = fit.n_states
        fold_summary["selected_covariance"] = fit.covariance_type
        fold_summary["allowed_regimes"] = ",".join(map(str, allowed_regimes))
        fold_summaries.append(fold_summary)

        train_labeled_frames.append(train_labeled)
        test_labeled_frames.append(test_labeled)
        filtered_frames.append(filtered)

    regimes = pd.concat(regime_frames, ignore_index=True) if regime_frames else pd.DataFrame()
    train_labeled_all = pd.concat(train_labeled_frames, ignore_index=True) if train_labeled_frames else pd.DataFrame()
    labeled = pd.concat(test_labeled_frames, ignore_index=True) if test_labeled_frames else pd.DataFrame()
    filtered = pd.concat(filtered_frames, ignore_index=True) if filtered_frames else pd.DataFrame()
    model_selection = pd.concat(model_rows, ignore_index=True) if model_rows else pd.DataFrame()
    fold_comparison = pd.concat(fold_summaries, ignore_index=True) if fold_summaries else pd.DataFrame()

    summary = summarize_by_regime(labeled) if not labeled.empty else pd.DataFrame()
    significance = regime_significance_table(labeled, cfg) if not labeled.empty else pd.DataFrame()
    strategy_significance = strategy_significance_table(labeled, filtered, cfg) if not labeled.empty else pd.DataFrame()
    comparison = compare_baseline_filtered(labeled, filtered)
    fold_stability = build_fold_stability_table(fold_comparison)

    outputs = {
        "feature_path": features,
        "regime_path": regimes,
        "labeled_trades_path": labeled,
        "filtered_trades_path": filtered,
    }
    for key, frame in outputs.items():
        out_path = project_root / cfg["output"][key]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False)

    report_dir = project_root / cfg["output"]["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    model_selection.to_csv(report_dir / "model_selection.csv", index=False)
    fold_comparison.to_csv(report_dir / "walk_forward_folds.csv", index=False)
    fold_stability.to_csv(report_dir / "fold_stability.csv", index=False)
    train_labeled_all.to_csv(report_dir / "training_labeled_trades.csv", index=False)
    summary.to_csv(report_dir / "regime_summary.csv", index=False)
    significance.to_csv(report_dir / "regime_significance.csv", index=False)
    strategy_significance.to_csv(report_dir / "strategy_significance.csv", index=False)
    comparison.to_csv(report_dir / "baseline_vs_filtered.csv", index=False)
    write_markdown_report(report_dir / "h002_report.md", summary, significance, comparison)

    print("H002 research run complete.")
    print(f"Walk-forward folds completed: {len(folds)}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "config/h002.yaml"
    run_research(path)
