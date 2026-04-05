"""
 Simulation Module
=============================
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union, cast, Optional

from config import CONFIG
from data_processing import load_all_token_data_cached
from utils import generate_trading_weeks

from precompute import load_features_from_cache


from pair_selection import (
    select_pairs_mfdcca,
    select_pairs_dcca,
    select_pairs_pearson,
    select_pairs_cointegration,
)
from trading import (
    apply_divergence_filter,
    simulate_pair_trades,
    calculate_performance_metrics,
    create_empty_metrics,
)

logger = logging.getLogger(__name__)

# Type alias for feature dictionaries
FeatureDict = Union[Dict[str, Any], Dict[Tuple[str, str], Dict[str, Any]]]


def load_features_with_metadata(week_start, method):
    """Load cache with all metadata (backward compatible)"""
    cache_data = load_features_from_cache(week_start, method)

    # Handle case where cache_data is just features (old format)
    if not isinstance(cache_data, dict) or "method" not in cache_data:
        features = cache_data
        cache_data = {
            "features": features,
            "method": method,
            "week_start": week_start,
            "token_list": [],
            "lookback_start": None,
            "lookback_end": None,
        }

    # Backward compatibility: if old cache format
    if "token_list" not in cache_data:
        features = cache_data["features"]

        if isinstance(features, dict):
            if "token_list" in features:
                cache_data["token_list"] = features["token_list"]
            elif all(isinstance(k, tuple) for k in features.keys()):
                token_set = set()
                for t1, t2 in features.keys():
                    token_set.add(t1)
                    token_set.add(t2)
                cache_data["token_list"] = list(token_set)
            else:
                cache_data["token_list"] = []
        else:
            cache_data["token_list"] = []

    if "lookback_start" not in cache_data:
        cache_data["lookback_start"] = None
        cache_data["lookback_end"] = None

    return cache_data


def _evaluate_btc_benchmark(start_date, end_date):
    """BTC buy-and-hold benchmark strategy"""
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATING BTC BUY-AND-HOLD BENCHMARK")
    logger.info("=" * 60)

    all_data = load_all_token_data_cached(
        start_date=CONFIG["data_loading_start"],
        end_date=CONFIG["data_loading_end"],
        market_index=CONFIG["market_index"],
    )

    btc_symbol = CONFIG.get("btc_symbol", "BTC")

    if btc_symbol not in all_data:
        logger.error(f"BTC symbol '{btc_symbol}' not found in data")
        return create_empty_metrics()

    btc_data = all_data[btc_symbol]
    logger.info(f"BTC data: {len(btc_data)} days loaded")

    btc_returns = btc_data["close"].pct_change().dropna()
    all_dates = pd.DatetimeIndex(btc_returns.index)
    weekly_starts = generate_trading_weeks(all_dates, start_date, end_date)
    logger.info(
        f"Processing {len(weekly_starts)} weeks from {start_date.date()} to {end_date.date()}"
    )

    weekly_results = []

    for week_start in weekly_starts:
        week_idx = all_dates.searchsorted(week_start)

        if week_idx >= len(all_dates):
            continue

        week_end_idx = min(week_idx + 5, len(all_dates))
        week_days = all_dates[week_idx:week_end_idx]

        if len(week_days) < 2:
            continue

        week_btc_returns = btc_returns.loc[week_days]
        returns_array = week_btc_returns.to_numpy(dtype=np.float32)
        weekly_return = float(np.prod(1.0 + returns_array) - 1.0)

        weekly_results.append(
            {
                "Week_Number": len(weekly_results) + 1,
                "Week_Start": week_days[0],
                "Week_End": week_days[-1],
                "Weekly_Return_%": float(weekly_return * 100.0),
                "Daily_Returns": week_btc_returns,
                "Num_Pairs": 0,
                "Active_Pairs": set(),
            }
        )

    if not weekly_results:
        logger.warning("No weekly results generated for BTC benchmark")
        return create_empty_metrics()

    temp_dir = Path(CONFIG["results_dir"]) / "temp" / "btc"
    temp_dir.mkdir(parents=True, exist_ok=True)

    metrics = calculate_performance_metrics(
        weekly_results_list=weekly_results,
        result_dir=temp_dir,
        period_name="btc_benchmark",
    )

    logger.info(f"✅ BTC Benchmark: Sharpe={metrics.get('Sharpe_Ratio', 0):.4f}")
    return metrics


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================


def run_full_period_evaluation(
    method: str,
    params: Dict[str, Any],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    use_cache: bool = True,
    return_weekly_data: bool = False,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:

    logger.info(f"\n{'─'*60}")
    logger.info(f"EVALUATING: {method.upper()}")
    logger.info(f"Period: {start_date.date()} → {end_date.date()}")
    logger.info(f"Parameters: {params}")
    logger.info(f"Use precompute: {use_cache}")

    if method == "btc":
        metrics = _evaluate_btc_benchmark(start_date, end_date)
        if return_weekly_data:
            return metrics, None
        return metrics

    if method == "index":
        metrics = _evaluate_index_benchmark(start_date, end_date)
        if return_weekly_data:
            return metrics, None
        return metrics

    all_data = load_all_token_data_cached(
        start_date=CONFIG["data_loading_start"],
        end_date=CONFIG["data_loading_end"],
        market_index=CONFIG["market_index"],
    )

    market_dates = all_data[CONFIG["market_index"]].index
    all_dates = pd.DatetimeIndex(market_dates)

    weekly_starts = generate_trading_weeks(all_dates, start_date, end_date)
    logger.info(f"Processing {len(weekly_starts)} trading weeks\n")

    weekly_results = []

    for week_num, week_start in enumerate(weekly_starts, 1):

        # ====================================================================
        # STEP A: Resolve exact trading days for this week FIRST.
        # We need week_days before anything else so that we can build a
        # correctly-indexed zero daily_returns Series regardless of what
        # happens later (cache miss, no pairs, no divergence, etc.).
        # ====================================================================
        week_start_idx = all_dates.searchsorted(week_start)

        if week_start_idx >= len(all_dates) or all_dates[week_start_idx] != week_start:
            logger.warning(
                f"Week start {week_start} not found in trading dates — skipping"
            )
            # True skip: this date doesn't exist in the data at all.
            # This is the only legitimate `continue` in the loop.
            continue

        week_end_idx = min(week_start_idx + 5, len(all_dates))
        week_days = all_dates[week_start_idx:week_end_idx]

        if len(week_days) < 2:
            continue  # Can't compute a return without at least 2 days

        week_end = week_days[-1]

        # ====================================================================
        # FIX 1 + FIX 2 — Pre-initialise week to zero.
        #
        # OLD CODE had multiple `continue` statements that caused weeks with
        # no pairs / no cache / bad data to vanish from weekly_results.
        # This made MFDCCA's Sharpe calculated on ~81 weeks vs BTC's 261
        # weeks, artificially shrinking std_dev and inflating Sharpe 2-3x.
        #
        # NEW APPROACH:
        #   1. Initialise daily_returns as a Series of 0.0 with the correct
        #      date index (one entry per trading day after day-0 in the week).
        #      This is exactly the same index structure that simulate_pair_trades
        #      produces, so calculate_performance_metrics sees a uniform number
        #      of observations across all methods.
        #   2. Only overwrite these zeros if actual trades happen.
        #   3. ALWAYS append the week at the bottom — no more silent drops.
        # ====================================================================

        # --- Pre-initialised defaults (zero week) ---
        num_selected = 0
        num_filtered = 0
        weekly_return = 0.0
        filtered_pairs: List[Dict[str, Any]] = []
        active_pairs_this_week: set = set()

        week_lookback_start = None
        week_lookback_end = None

        # One 0.0 per return-day in the week (days 1..N, so N-1 entries).
        # Using week_days[1:] as the index mirrors simulate_pair_trades output.
        daily_returns = pd.Series(
            [0.0] * (len(week_days) - 1),
            index=week_days[1:],
        )

        # ====================================================================
        # STEP B: Attempt to load cache and run trading logic.
        # Any failure keeps the pre-initialised zeros; the week still lands
        # in weekly_results as a zero-return week.
        # ====================================================================
        try:
            cache_data = load_features_with_metadata(week_start, method)
            features = cache_data["features"]
            tokens_with_data = cache_data.get("token_list", [])
            week_lookback_start = cache_data.get("lookback_start")
            week_lookback_end = cache_data.get("lookback_end")

            if not tokens_with_data:
                logger.debug(f"Week {week_num}: No tokens in cache — zero week")
                # Fall through to append with zeros
            else:
                # ============================================================
                # PAIR SELECTION
                # ============================================================
                selected_pairs = _select_pairs(
                    method=method,
                    features=features,
                    params=params,
                )
                num_selected = len(selected_pairs)

                if num_selected > 0:
                    divergence_lookback = params.get("divergence_lookback")
                    if divergence_lookback is None:
                        logger.warning("Missing divergence_lookback — zero week")
                    else:
                        # Normalise numeric type safely
                        if hasattr(divergence_lookback, "item"):
                            divergence_lookback = int(divergence_lookback.item())
                        else:
                            divergence_lookback = int(divergence_lookback)

                        if (
                            divergence_lookback < 3
                        ):  # ← Slightly safer minimum (was <=1)
                            logger.warning(
                                f"divergence_lookback={divergence_lookback} too small — skipping divergence filter"
                            )
                            # Optional: proceed without filter if you want (fallback)
                        else:
                            required_length = divergence_lookback + 1

                            # Build divergence price window
                            lookback_dates = all_data[CONFIG["market_index"]].index
                            lookback_dates = lookback_dates[
                                (lookback_dates >= week_lookback_start)
                                & (lookback_dates <= week_lookback_end)
                            ]

                            if len(lookback_dates) >= 2:
                                div_dates = lookback_dates[-required_length:]
                                div_start = div_dates[0]
                                div_end = div_dates[-1]

                                div_price_data: Dict[str, pd.DataFrame] = {}

                                # === FIXED: Much more tolerant price window ===
                                for token in tokens_with_data:
                                    if token not in all_data:
                                        continue
                                    df: pd.DataFrame = all_data[token]
                                    mask = (df.index >= div_start) & (
                                        df.index <= div_end
                                    )
                                    token_data = df.loc[mask]

                                    # Only minimal check: at least 2 days (same as divergence filter itself)
                                    if len(token_data) >= 2:
                                        div_price_data[token] = token_data
                                    # Removed:
                                    #   - required_length - 2 (too strict)
                                    #   - token_data["close"].notna().all() (one NaN kills token)

                                if len(div_price_data) >= 2:
                                    # ============================================
                                    # DIVERGENCE FILTER
                                    # ============================================
                                    divergence_threshold = params[
                                        "divergence_threshold"
                                    ]

                                    # === FIXED: Slightly more flexible valid_pairs ===
                                    valid_pairs = []
                                    for p in selected_pairs:
                                        t1, t2 = p
                                        # Require both tokens (strict but now safe after relaxed pre-check)
                                        if (
                                            t1 in div_price_data
                                            and t2 in div_price_data
                                        ):
                                            valid_pairs.append(p)
                                        # Optional: log partial cases for debugging
                                        # else:
                                        #     logger.debug(f"Pair skipped due to missing data: {t1}-{t2}")

                                    if valid_pairs:
                                        filtered_pairs = apply_divergence_filter(
                                            candidate_pairs=valid_pairs,
                                            price_data=div_price_data,
                                            lookback_days=divergence_lookback,
                                            divergence_threshold=divergence_threshold,
                                        )
                                        num_filtered = len(filtered_pairs)

                                        # ========================================
                                        # EXECUTE TRADES
                                        # ========================================
                                        if filtered_pairs:
                                            trade_tokens = {
                                                p["long_token"] for p in filtered_pairs
                                            } | {
                                                p["short_token"] for p in filtered_pairs
                                            }
                                            week_price_data = {
                                                tok: all_data[tok].loc[
                                                    week_start:week_end
                                                ]
                                                for tok in trade_tokens
                                                if tok in all_data
                                            }
                                            trade_results = simulate_pair_trades(
                                                filtered_pairs=filtered_pairs,
                                                price_data=week_price_data,
                                                week_start=week_start,
                                                week_end=week_end,
                                            )
                                            weekly_return = trade_results[
                                                "Weekly_Return"
                                            ]
                                            active_pairs_this_week = trade_results[
                                                "Active_Pairs"
                                            ]
                                            actual_daily = trade_results[
                                                "Daily_Returns"
                                            ]
                                            if (
                                                isinstance(actual_daily, pd.Series)
                                                and not actual_daily.empty
                                            ):
                                                daily_returns = actual_daily

        except FileNotFoundError:
            logger.debug(f"Week {week_num}: Cache not found — zero week")
        except Exception as exc:
            logger.warning(f"Week {week_num}: Error ({exc}) — zero week")

        # ====================================================================
        # ALWAYS APPEND — every calendar week appears in weekly_results.
        #
        # Weeks without trades contribute (len(week_days)-1) zero daily
        # return observations.  This ensures:
        #   • All methods have the same number of daily observations as BTC
        #   • std_dev is computed on the full sample (not just active weeks)
        #   • Sharpe / Sortino / Max-Drawdown are genuinely comparable
        # ====================================================================
        weekly_results.append(
            {
                "Week_Number": week_num,
                "Week_Start": week_start,
                "Week_End": week_end,
                "Lookback_Start": week_lookback_start,
                "Lookback_End": week_lookback_end,
                "Num_Selected_Pairs": num_selected,
                "Num_Filtered_Pairs": num_filtered,
                "Weekly_Return_%": weekly_return * 100,
                "Daily_Returns": daily_returns,
                "Active_Pairs": active_pairs_this_week,
            }
        )

    # ========================================================================
    # CALCULATE METRICS
    # ========================================================================
    if not weekly_results:
        logger.warning("No weekly results generated")
        empty_metrics = create_empty_metrics()
        if return_weekly_data:
            return empty_metrics, {"weekly_results": []}
        return empty_metrics

    # Diagnostic log — confirms fix is working
    active_weeks = sum(1 for w in weekly_results if w.get("Num_Filtered_Pairs", 0) > 0)
    total_daily_obs = sum(
        len(w["Daily_Returns"])
        for w in weekly_results
        if isinstance(w.get("Daily_Returns"), pd.Series)
    )
    logger.info(f"\nWeekly results summary:")
    logger.info(f"  Total weeks  : {len(weekly_results)}")
    logger.info(f"  Active weeks : {active_weeks}")
    logger.info(f"  Zero weeks   : {len(weekly_results) - active_weeks}")
    logger.info(f"  Daily obs    : {total_daily_obs}")

    temp_dir = Path(CONFIG["results_dir"]) / "temp" / method
    temp_dir.mkdir(parents=True, exist_ok=True)

    metrics = calculate_performance_metrics(
        weekly_results_list=weekly_results,
        result_dir=temp_dir,
        period_name="evaluation",
    )

    logger.info(f"✅ Sharpe: {metrics.get('Sharpe_Ratio', 0):.4f}")

    if return_weekly_data:
        return metrics, {"weekly_results": weekly_results}
    return metrics


def save_mfdcca_yearly_summary(all_period_results):

    summary_data = []

    for period in CONFIG["sub_periods"]:
        year_name = period["name"]

        if year_name not in all_period_results:
            continue

        if "mfdcca" not in all_period_results[year_name]:
            continue

        metrics = all_period_results[year_name]["mfdcca"]["metrics"]

        summary_data.append(
            {
                "Year": year_name,
                "Mean_Return_%": metrics.get("Mean_Return", 0.0),
                "Std_Deviation_%": metrics.get("Std_Dev", 0.0),
                "Sharpe_Ratio": metrics.get("Sharpe_Ratio", 0.0),
                "Sortino_Ratio": metrics.get("Sortino_Ratio", 0.0),
                "Max_Drawdown_%": metrics.get("Max_Drawdown_%", 0.0),
                "Profit_Factor": metrics.get("Profit_Factor", 0.0),
            }
        )

    df = pd.DataFrame(summary_data)

    results_dir = Path(CONFIG["results_dir"]) / "sub_period"
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "mfdcca_yearly_summary.csv", index=False)

    return df


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _select_pairs(
    method: str, features: Optional[FeatureDict], params: Dict[str, Any]
) -> List[Tuple[str, str]]:
    """Unified pair selection logic with proper typing"""
    if features is None:
        return []

    if method in ("mfdcca", "mfdcca_raw"):
        mfdcca_features = cast(Dict[str, Any], features)
        return select_pairs_mfdcca(
            features=mfdcca_features,
            pair_hxy_threshold=params["pair_hxy_threshold"],
            threshold_h=params["threshold_h"],
            threshold_alpha=params["threshold_alpha"],
        )

    elif method == "dcca":
        dcca_features = cast(Dict[Tuple[str, str], Dict[str, Any]], features)
        token_set: set = set()
        for t1, t2 in dcca_features.keys():
            token_set.add(t1)
            token_set.add(t2)
        return select_pairs_dcca(
            features=dcca_features,
            pair_hxy_threshold=params["pair_hxy_threshold"],
            token_list=list(token_set),
        )

    elif method == "pearson":
        pearson_features = cast(Dict[str, Any], features)
        return select_pairs_pearson(
            features=pearson_features,
            rho_threshold=params["rho_threshold"],
        )

    elif method == "cointegration":
        coint_features = cast(Dict[Tuple[str, str], Dict[str, Any]], features)
        token_set = set()
        for t1, t2 in coint_features.keys():
            token_set.add(t1)
            token_set.add(t2)
        return select_pairs_cointegration(
            features=coint_features,
            pval_threshold=params["pval_threshold"],
            token_list=list(token_set),
        )

    return []


def _evaluate_index_benchmark(start_date, end_date):
    all_data = load_all_token_data_cached(
        start_date=CONFIG["data_loading_start"],
        end_date=CONFIG["data_loading_end"],
        market_index=CONFIG["market_index"],
    )

    token_data = {k: v for k, v in all_data.items() if k != CONFIG["market_index"]}
    if len(token_data) < 2:
        return create_empty_metrics()

    daily_returns = {
        token: df["close"].pct_change() for token, df in token_data.items()
    }
    returns_df = pd.DataFrame(daily_returns).dropna(how="any")

    all_dates = pd.DatetimeIndex(returns_df.index)
    weekly_starts = generate_trading_weeks(all_dates, start_date, end_date)

    weekly_results = []

    for week_start in weekly_starts:
        week_idx = all_dates.searchsorted(week_start)
        if week_idx + 5 > len(all_dates):
            continue

        week_end_idx = min(week_idx + 5, len(all_dates))
        week_days = all_dates[week_idx:week_end_idx]

        week_returns = returns_df.loc[week_days]
        daily_portfolio_returns = week_returns.mean(axis=1)
        weekly_return = (
            float(np.asarray(daily_portfolio_returns.add(1.0).prod()).item()) - 1.0
        )

        weekly_results.append(
            {
                "Week_Number": len(weekly_results) + 1,
                "Week_Start": week_days[0],
                "Week_End": week_days[-1],
                "Weekly_Return_%": weekly_return * 100,
                "Daily_Returns": daily_portfolio_returns,
                "Num_Pairs": 0,
            }
        )

    temp_dir = Path(CONFIG["results_dir"]) / "temp" / "index"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return calculate_performance_metrics(
        weekly_results_list=weekly_results,
        result_dir=temp_dir,
        period_name="index_benchmark",
    )
