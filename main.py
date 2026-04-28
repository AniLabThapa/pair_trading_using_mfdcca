"""
main.py
=========================================================
Refactored & Clean Orchestration Script

=========================================================
"""

import logging
import numpy as np
from itertools import product
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from sensitivity_analysis import run_sensitivity_analysis
from joblib import Parallel, delayed

import pandas as pd
from itertools import product

from config import CONFIG
from data_processing import load_all_token_data_cached
from precompute import precompute_all_methods
from stats_analysis import compute_asset_statistics
from simulation import run_full_period_evaluation
from utils import generate_trading_weeks
from first_week_visualizations import (
    plot_first_week_capm_scatter,
    plot_first_week_raw_vs_residual,
    compute_and_save_delta_matrices,
)

# =========================================================
# LOGGING CONFIGURATION
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)
# =========================================================
# UTILITY FUNCTIONS
# =========================================================


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_metrics(result):
    """Unified extraction of metrics and weekly data."""
    if isinstance(result, tuple):
        if len(result) == 2:
            return result[0], result[1]
        return result[0], None
    return result, None


def generate_parameter_cases(method: str) -> List[Dict[str, Any]]:
    grid = CONFIG["parameter_grids"].get(method, {})

    if not grid:
        return [{}]

    names = list(grid.keys())
    values = []

    for name in names:
        spec = grid[name]
        if not isinstance(spec, dict):
            raise ValueError(f"Invalid parameter spec for {method}:{name}")

        # ✅ FIX: Use np.arange instead of while loop to avoid floating point drift
        is_int_param = all(isinstance(spec[k], int) for k in ["min", "max", "step"])

        v = np.arange(spec["min"], spec["max"] + spec["step"] / 2, spec["step"])

        if is_int_param:
            v = [int(x) for x in v]
        else:
            v = [round(float(x), 10) for x in v]

        values.append(v)

    return [dict(zip(names, combo)) for combo in product(*values)]


def create_daily_profit_csv(daily_returns: pd.Series) -> pd.DataFrame:
    if daily_returns is None or daily_returns.empty:
        return pd.DataFrame(columns=["Date", "Daily_Return_%", "Cumulative_Return_%"])

    daily_returns = daily_returns.sort_index()
    cumulative = (1 + daily_returns).cumprod() - 1

    return pd.DataFrame(
        {
            "Date": daily_returns.index,
            "Daily_Return_%": (daily_returns * 100).values,
            "Cumulative_Return_%": (cumulative * 100).values,
        }
    )


def create_weekly_rebalancing_csv(weekly_results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for w in weekly_results:
        rows.append(
            {
                "Week_Number": w.get("Week_Number", 0),
                "Week_Start": w.get("Week_Start", ""),
                "Week_End": w.get("Week_End", ""),
                "Lookback_Start": w.get("Lookback_Start", ""),
                "Lookback_End": w.get("Lookback_End", ""),
                "Cutoff_Date": w.get("Lookback_End", ""),
                "Pairs_Before_Filter": w.get("Num_Selected_Pairs", 0),
                "Pairs_After_Filter": w.get("Num_Filtered_Pairs", 0),
                "Weekly_Return_%": w.get("Weekly_Return_%", 0.0),
            }
        )
    return pd.DataFrame(rows)


# =========================================================
# CORE PIPELINE FUNCTIONS
# =========================================================


def evaluate_all_cases(
    method: str, cases: List[Dict[str, Any]], use_cache: bool
) -> pd.DataFrame:
    logger.info(f"Evaluating {method.upper()} with {len(cases)} cases")

    results = []

    for i, params in enumerate(cases, 1):
        logger.info(f"[{method.upper()} Case {i}/{len(cases)}] {params}")

        try:
            result = run_full_period_evaluation(
                method=method,
                params=params,
                start_date=CONFIG["full_period_start"],
                end_date=CONFIG["full_period_end"],
                use_cache=use_cache,
                return_weekly_data=False,
            )

            metrics, _ = extract_metrics(result)

            row = params.copy()
            row.update(metrics)
            row["case_number"] = i

            # ✅ FIX: Remove non-scalar Series before saving to CSV
            row.pop("Daily_Return_Series", None)

            results.append(row)

            logger.info(f"Sharpe={metrics.get('Sharpe_Ratio', 0):.4f}")

        except Exception as e:
            logger.error(f"Case failed: {params}", exc_info=True)

    df = pd.DataFrame(results)
    out_dir = ensure_dir(Path(CONFIG["results_dir"]) / "full_period" / method)
    df.to_csv(out_dir / "all_cases.csv", index=False)

    return df


def select_best_case(df: pd.DataFrame, method: str) -> Dict[str, Any]:
    best_row = df.loc[df["Sharpe_Ratio"].idxmax()]
    params = CONFIG["parameter_grids"].get(method, {}).keys()
    best_params = {k: best_row[k] for k in params if k in best_row}

    out_dir = Path(CONFIG["results_dir"]) / "full_period" / method

    best_row_clean = best_row.drop(labels=["Daily_Return_Series"], errors="ignore")
    pd.DataFrame([best_row_clean]).to_csv(out_dir / "best_case.csv", index=False)

    logger.info(f"Best {method.upper()} params: {best_params}")

    result: Dict[str, Any] = {str(k): best_row[k] for k in best_row.index}
    return result


# =========================================================


# =========================================================
def generate_full_period_detailed_results(
    method: str, params: Dict[str, Any], use_cache: bool
):
    # passed from best_params_snapshot[method] in Phase 4
    result = run_full_period_evaluation(
        method=method,
        params=params,
        start_date=CONFIG["full_period_start"],
        end_date=CONFIG["full_period_end"],
        use_cache=use_cache,
        return_weekly_data=True,
    )

    metrics, weekly_data = extract_metrics(result)

    out_dir = ensure_dir(Path(CONFIG["results_dir"]) / "full_period" / method)

    daily_returns = metrics.get("Daily_Return_Series")
    if isinstance(daily_returns, pd.Series):
        create_daily_profit_csv(daily_returns).to_csv(
            out_dir / "daily_profit.csv", index=False
        )

    if weekly_data and "weekly_results" in weekly_data:
        create_weekly_rebalancing_csv(weekly_data["weekly_results"]).to_csv(
            out_dir / "weekly_rebalancing.csv", index=False
        )


def create_method_comparison_table(best_cases: Dict[str, Dict[str, Any]]):
    """Create comparison table matching professor's exact format"""
    logger.info("\n" + "=" * 70)
    logger.info("Creating Method Comparison Table")
    logger.info("=" * 70)

    comparison_data = []

    # EXACT ORDER: BTC, INDEX, then Pair Trading methods
    methods_order = [
        "btc",
        "index",
        "pearson",
        "cointegration",
        "dcca",
        "mfdcca_raw",
        "mfdcca",
    ]

    for method in methods_order:
        if method not in best_cases:
            logger.warning(f"Method '{method}' not found in best_cases, skipping...")
            continue

        metrics = best_cases[method]

        comparison_data.append(
            {
                "Metrics": method.upper(),
                "Mean_return": round(metrics.get("Mean_Return", 0.0), 4),
                "Standard_deviation": round(metrics.get("Std_Dev", 0.0), 4),
                "Sharpe_ratio": round(metrics.get("Sharpe_Ratio", 0.0), 4),
                "Sortino_ratio": round(metrics.get("Sortino_Ratio", 0.0), 4),
                "Downside_deviation": round(metrics.get("Downside_Deviation", 0.0), 4),
                "Profit_factor": round(metrics.get("Profit_Factor", 0.0), 4),
                "Gross_profit": round(metrics.get("Gross_Profit", 0.0), 2),
                "Gross_loss": round(metrics.get("Gross_Loss", 0.0), 2),
            }
        )

    df = pd.DataFrame(comparison_data)

    # TRANSPOSE: Metrics as rows, Methods as columns
    df_transposed = df.set_index("Metrics").T
    df_transposed = df_transposed.reset_index()
    df_transposed = df_transposed.rename(columns={"index": "Metrics"})

    output_dir = ensure_dir(Path(CONFIG["results_dir"]) / "full_period")
    output_file = output_dir / "method_comparison.csv"
    df_transposed.to_csv(output_file, index=False)

    logger.info(f"✅ Saved comparison table to {output_file}")

    return df_transposed


# =========================================================
# SUB-PERIOD EVALUATION FUNCTION
# =========================================================


def evaluate_sub_periods(
    best_mfdcca_params: Dict[str, Any],  # ← FIX: explicit clean params, not best_cases
    use_cache: bool,
):
    logger.info("\n" + "=" * 70)
    logger.info("Starting Sub-Period Yearly Evaluation (MFDCCA ONLY)")
    logger.info("=" * 70)

    # ✅ Use the params passed in directly — guaranteed to be the correct
    # optimized values from Phase 3, not corrupted by Phase 4 overwrite.
    logger.info(f"MFDCCA params for sub-periods: {best_mfdcca_params}")

    # Validate that params are not None before proceeding
    missing = [k for k, v in best_mfdcca_params.items() if v is None]
    if missing:
        logger.error(f"❌ Sub-period aborted: params contain None values: {missing}")
        logger.error("   This means best_cases was read after being overwritten.")
        return {}

    all_period_results = {}

    for sub_period in CONFIG["sub_periods"]:
        period_name = sub_period["name"]
        period_start = sub_period["study_start"]
        period_end = sub_period["study_end"]

        logger.info(f"\n{'='*70}")
        logger.info(f"Evaluating Year: {period_name} (MFDCCA)")
        logger.info(f"{'='*70}")

        # ✅ Run evaluation for this year using clean params
        result = run_full_period_evaluation(
            method="mfdcca",
            params=best_mfdcca_params,
            start_date=period_start,
            end_date=period_end,
            use_cache=False,
            return_weekly_data=True,
        )

        metrics, weekly_data = extract_metrics(result)

        all_period_results[period_name] = {
            "mfdcca": {
                "metrics": metrics,
                "params": best_mfdcca_params,
                "weekly_data": weekly_data,
            }
        }

        year_dir = ensure_dir(
            Path(CONFIG["results_dir"]) / "sub_period" / period_name / "mfdcca"
        )

        # metrics.csv
        pd.DataFrame([metrics]).to_csv(year_dir / "metrics.csv", index=False)

        # daily_profit.csv
        daily_returns = metrics.get("Daily_Return_Series")
        if isinstance(daily_returns, pd.Series):
            create_daily_profit_csv(daily_returns).to_csv(
                year_dir / "daily_profit.csv", index=False
            )

        # weekly_rebalancing.csv
        if weekly_data and "weekly_results" in weekly_data:
            create_weekly_rebalancing_csv(weekly_data["weekly_results"]).to_csv(
                year_dir / "weekly_rebalancing.csv", index=False
            )

        logger.info(f"✅ {period_name} MFDCCA evaluation complete")

    return all_period_results


# =========================================================
# MAIN PIPELINE
# =========================================================
def main():
    logger.info("=" * 80)
    logger.info("STARTING FULL PERIOD PIPELINE")
    logger.info("=" * 80)

    # =====================================================
    # PHASE 0 — LOAD DATA (ONCE)
    # =====================================================
    logger.info("PHASE 0: Loading data once...")

    DATA_START = pd.Timestamp("2020-01-01")
    all_data = load_all_token_data_cached(
        start_date=DATA_START,
        end_date=CONFIG["full_period_end"],
        market_index=CONFIG["market_index"],
    )

    # =====================================================
    # PHASE 1 — ASSET STATISTICS
    # =====================================================
    logger.info("PHASE 1: Asset Statistics")

    stats_df = compute_asset_statistics(all_data)
    stats_dir = ensure_dir(Path(CONFIG["results_dir"]) / "statistics")
    stats_df.to_csv(stats_dir / "asset_statistics.csv", index=False)

    # =====================================================
    # PHASE 2 — PRECOMPUTE (CACHE ALL METHODS)
    # =====================================================
    logger.info("PHASE 2: Precompute all methods")

    use_cache = False
    try:
        precompute_all_methods(
            start_date=CONFIG["full_period_start"],
            end_date=CONFIG["full_period_end"],
            methods=[
                "mfdcca",
                "mfdcca_raw",
                "dcca",
                "pearson",
                "cointegration",
                "btc",
                "index",
            ],
        )
        use_cache = True
        logger.info("Precompute completed successfully")
    except Exception as e:
        logger.warning(f"Precompute failed → fallback to direct computation: {e}")

    # =====================================================
    # PHASE 3 — HYPERPARAMETER OPTIMIZATION
    # =====================================================
    logger.info("PHASE 3: Hyperparameter Optimization")

    methods = [
        "mfdcca",
        "mfdcca_raw",
        "dcca",
        "pearson",
        "cointegration",
        "btc",
        "index",
    ]

    best_cases: Dict[str, Dict[str, Any]] = {}

    for method in methods:

        if method == "btc":
            logger.info("\n" + "=" * 70)
            logger.info("Evaluating BTC Buy-and-Hold Benchmark")
            logger.info("=" * 70)

            result = run_full_period_evaluation(
                method="btc",
                params={},
                start_date=CONFIG["full_period_start"],
                end_date=CONFIG["full_period_end"],
                use_cache=False,
                return_weekly_data=False,
            )
            metrics, _ = extract_metrics(result)
            best_cases["btc"] = metrics

            logger.info(
                f"✅ BTC Benchmark complete: Sharpe={metrics.get('Sharpe_Ratio', 0):.4f}"
            )
            continue

        if method == "index":
            result = run_full_period_evaluation(
                method="index",
                params={},
                start_date=CONFIG["full_period_start"],
                end_date=CONFIG["full_period_end"],
                use_cache=False,
                return_weekly_data=False,
            )
            metrics, _ = extract_metrics(result)
            best_cases["index"] = metrics
            logger.info("Best case selected for INDEX method")
            continue

        cases = generate_parameter_cases(method)
        df = evaluate_all_cases(method, cases, use_cache)
        best_cases[method] = select_best_case(df, method)

        logger.info(f"Best case selected for {method.upper()}")

    # =====================================================
    # ✅ SNAPSHOT PARAMS HERE — immediately after Phase 3
    # =====================================================
    best_params_snapshot: Dict[str, Dict[str, Any]] = {}
    for method in methods:
        grid_keys = CONFIG["parameter_grids"].get(method, {}).keys()
        best_params_snapshot[method] = {
            k: best_cases[method][k]
            for k in grid_keys
            if k in best_cases.get(method, {})
        }

    logger.info("✅ Parameter snapshot captured for all methods")
    for method, params in best_params_snapshot.items():
        if params:
            logger.info(f"   {method.upper()}: {params}")

    # =====================================================
    # PHASE 3.5 — SENSITIVITY ANALYSIS
    # =====================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3.5: Sensitivity Analysis")
    logger.info("=" * 70)

    try:
        # ✅ Use snapshot — best_cases["mfdcca"] still has params at this point
        best_mfdcca_params = best_params_snapshot["mfdcca"]

        logger.info(f"Running sensitivity analysis with optimal parameters:")
        logger.info(f"  {best_mfdcca_params}")

        run_sensitivity_analysis(
            method="mfdcca",
            best_params=best_mfdcca_params,
            start_date=CONFIG["full_period_start"],
            end_date=CONFIG["full_period_end"],
            use_cache=use_cache,
        )
        logger.info("✅ Sensitivity analysis complete\n")

    except Exception as e:
        logger.error(f"Sensitivity analysis failed: {e}", exc_info=True)
        logger.warning("Continuing without sensitivity analysis...")

    # =====================================================
    # PHASE 4 — FINAL FULL-PERIOD EVALUATION (BEST PARAMS)
    # =====================================================
    logger.info("PHASE 4: Final Full-Period Evaluation")

    final_results = {}

    for method in methods:
        # ✅ Use snapshot — safe even after best_cases[method] is overwritten below
        params = best_params_snapshot[method]

        result = run_full_period_evaluation(
            method=method,
            params=params,
            start_date=CONFIG["full_period_start"],
            end_date=CONFIG["full_period_end"],
            use_cache=use_cache,
            return_weekly_data=True,
        )

        final_results[method] = result
        metrics, _ = extract_metrics(result)

        best_cases[method] = metrics

        # ✅ Pass params from snapshot (not best_cases[method] which is now metrics)
        generate_full_period_detailed_results(method, params, use_cache)

    create_method_comparison_table(best_cases)

    # =====================================================
    # PHASE 4.5 — SUB-PERIOD EVALUATION (YEARLY)
    # =====================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4.5: Sub-Period Yearly Evaluation")
    logger.info("=" * 70)

    all_period_results = evaluate_sub_periods(
        best_mfdcca_params=best_params_snapshot["mfdcca"],
        use_cache=use_cache,
    )

    if all_period_results:
        from simulation import save_mfdcca_yearly_summary

        save_mfdcca_yearly_summary(all_period_results)

    logger.info("✅ Sub-period evaluation complete\n")

    # =====================================================
    # PHASE 5 — VISUALIZATION & MATRICES
    # =====================================================
    logger.info("PHASE 5: Final Visualizations & Matrices")

    plot_first_week_capm_scatter(all_data)
    plot_first_week_raw_vs_residual(all_data)
    compute_and_save_delta_matrices(all_data)

    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

    # =====================================================
    # PHASE 6 — CAPM IMPACT TABLE
    # =====================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 6: CAPM Impact Analysis")
    logger.info("=" * 70)

    try:
        out_dir = ensure_dir(
            Path(CONFIG["results_dir"]) / "full_period" / "capm_impact"
        )

        # ── Save each variant's full metrics individually ──────────────
        metrics_capm = best_cases["mfdcca"]
        metrics_raw = best_cases["mfdcca_raw"]

        # Individual full metrics (all fields)
        pd.DataFrame([metrics_capm]).drop(
            columns=["Daily_Return_Series"], errors="ignore"
        ).to_csv(out_dir / "mfdcca_with_capm_metrics.csv", index=False)

        pd.DataFrame([metrics_raw]).drop(
            columns=["Daily_Return_Series"], errors="ignore"
        ).to_csv(out_dir / "mfdcca_raw_metrics.csv", index=False)

        # ── Save daily return series for both ──────────────────────────
        daily_capm = metrics_capm.get("Daily_Return_Series")
        daily_raw = metrics_raw.get("Daily_Return_Series")

        if isinstance(daily_capm, pd.Series) and not daily_capm.empty:
            create_daily_profit_csv(daily_capm).to_csv(
                out_dir / "mfdcca_with_capm_daily.csv", index=False
            )

        if isinstance(daily_raw, pd.Series) and not daily_raw.empty:
            create_daily_profit_csv(daily_raw).to_csv(
                out_dir / "mfdcca_raw_daily.csv", index=False
            )

        # ── Combined comparison table ──────────────────────────────────
        def _row(label, capm_flag, m):
            return {
                "Method": label,
                "CAPM_filter": capm_flag,
                "Sharpe_ratio": round(m.get("Sharpe_Ratio", 0.0), 4),
                "Sortino_ratio": round(m.get("Sortino_Ratio", 0.0), 4),
                "MDD_%": round(m.get("Max_Drawdown_%", 0.0), 4),
                "Calmar_ratio": round(m.get("Calmar_Ratio", 0.0), 4),
                "Profit_factor": round(m.get("Profit_Factor", 0.0), 4),
                "Gross_profit": round(m.get("Gross_Profit", 0.0), 4),
                "Gross_loss": round(m.get("Gross_Loss", 0.0), 4),
            }

        capm_df = pd.DataFrame(
            [
                _row("MFDCCA", "X", metrics_raw),  # X = without CAPM
                _row("MFDCCA", "O", metrics_capm),  # O = with CAPM
            ]
        )

        capm_df.to_csv(out_dir / "capm_impact_table.csv", index=False)

        logger.info(f"\n{capm_df.to_string(index=False)}")
        logger.info(f"\n✅ Saved to: {out_dir}")
        logger.info(f"   mfdcca_with_capm_metrics.csv")
        logger.info(f"   mfdcca_raw_metrics.csv")
        logger.info(f"   mfdcca_with_capm_daily.csv")
        logger.info(f"   mfdcca_raw_daily.csv")
        logger.info(f"   capm_impact_table.csv")

    except Exception as e:
        logger.error(f"CAPM impact analysis failed: {e}", exc_info=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Fatal error", exc_info=True)
        sys.exit(1)
