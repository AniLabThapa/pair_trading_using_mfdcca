"""
 Precompute Module -
===========================================
"""

import logging
import pandas as pd
import torch
from pathlib import Path
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union

from config import CONFIG, DEVICE
from data_processing import load_all_token_data_cached
from capm import apply_capm_filter
from utils import generate_trading_weeks
from feature_extraction import (
    extract_mfdcca_features,
    extract_dcca_features,
    extract_pearson_features,
    extract_cointegration_features,
)

logger = logging.getLogger(__name__)

# Type alias for feature dictionaries
FeatureDict = Union[Dict[str, Any], Dict[Tuple[str, str], Dict[str, Any]]]


# ============================================================================
# CORE CACHE FUNCTIONS (Minimal & Essential)
# ============================================================================
def _get_cache_path(method: str, week_start: pd.Timestamp) -> Path:
    """Cache path using only the week start date"""
    cache_dir = Path(CONFIG["results_dir"]) / "precompute_v2" / method

    cache_dir.mkdir(parents=True, exist_ok=True)
    # Use ISO format date: 2021-01-04
    return cache_dir / f"{week_start.strftime('%Y-%m-%d')}.pt"


def save_features_to_cache(
    method: str,
    week_start: pd.Timestamp,
    features: FeatureDict,
    token_list: List[str],
    lookback_start: pd.Timestamp,
    lookback_end: pd.Timestamp,
    capm_results: Optional[Dict] = None,
) -> None:
    """Save features with metadata to cache."""
    cache_file = _get_cache_path(method, week_start)

    cache_data = {
        "features": features,
        "method": method,
        "week_start": week_start,
        "token_list": token_list,
        "lookback_start": lookback_start,
        "lookback_end": lookback_end,
        "capm_results": capm_results,
    }

    torch.save(cache_data, cache_file)
    logger.debug(f"Cached {week_start.date()}: {cache_file.name}")


def load_features_from_cache(week_start, method, cache_dir=None):
    """Load precomputed features from cache."""
    if cache_dir is None:
        cache_dir = Path(CONFIG["results_dir"]) / "precompute_v2" / method

    if isinstance(week_start, pd.Timestamp):
        week_str = week_start.strftime("%Y-%m-%d")
    else:
        week_str = str(week_start)

    cache_file = cache_dir / f"{week_str}.pt"

    if not cache_file.exists():
        raise FileNotFoundError(
            f"Precomputed features not found for week {week_str}. "
            "Run precompute_features_for_all_weeks first."
        )

    cache_data = torch.load(cache_file, weights_only=False)

    if cache_data["method"] != method:
        raise ValueError(
            f"Cached method mismatch: expected {method}, "
            f"found {cache_data['method']}"
        )

    # ✅ RETURN FULL CACHE DATA, NOT JUST FEATURES
    return cache_data  # NOT cache_data["features"]


# ============================================================================
# UNIFIED PRECOMPUTE FUNCTION (NEW - ADD THIS)
# ============================================================================


def precompute_all_methods(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    methods: Optional[List[str]] = None,
) -> Dict[str, int]:
    """
    ✨ OPTIMIZED: Compute ALL methods in ONE pass
    """
    if methods is None:
        methods = ["mfdcca", "dcca", "pearson", "cointegration"]

    logger.info(f"\n{'='*70}")
    logger.info(f"UNIFIED PRECOMPUTE: {', '.join(m.upper() for m in methods)}")
    logger.info(f"{'='*70}")

    # ✅ FIX: Use fixed data range from CONFIG
    all_data = load_all_token_data_cached(
        start_date=CONFIG["data_loading_start"],
        end_date=CONFIG["data_loading_end"],
        market_index=CONFIG["market_index"],
    )

    if not all_data:
        logger.error("No data loaded!")
        return {m: 0 for m in methods}

    # Get trading dates
    market_dates = all_data[CONFIG["market_index"]].index
    all_dates = pd.DatetimeIndex(market_dates)

    # Generate weeks - filter using start_date and end_date parameters
    weekly_starts = generate_trading_weeks(all_dates, start_date, end_date)
    logger.info(f"📅 Processing {len(weekly_starts)} weeks for {len(methods)} methods")
    logger.info(f"📊 Total tasks: {len(weekly_starts) * len(methods)}\n")

    successful_counts = {method: 0 for method in methods}

    # ═══════════════════════════════════════════════════════════
    # STEP 2: Loop weeks ONCE
    # ═══════════════════════════════════════════════════════════
    for week_num, week_start in enumerate(weekly_starts, 1):
        try:  # ← 8 spaces, not 12
            # Extract lookback window
            prev_dates = all_dates[all_dates < week_start]
            if len(prev_dates) < 200:
                continue

            lookback_start = prev_dates[-250]
            lookback_end = prev_dates[-1]

            lookback_data = {}
            for token, df in all_data.items():
                mask = (df.index >= lookback_start) & (df.index <= lookback_end)
                window_data = df[mask]
                if len(window_data) >= 200:
                    lookback_data[token] = window_data.copy()

            capm_results = apply_capm_filter(
                tokens=list(lookback_data.keys()),
                market_index=CONFIG["market_index"],
                price_data=lookback_data,
            )

            if not capm_results:
                logger.debug(f"Week {week_num} ({week_start.date()}): CAPM failed")
                continue

            residuals = {
                t: capm_results[t]["residuals"]
                for t in capm_results
                if "residuals" in capm_results[t]
            }

            for method in methods:
                try:
                    result = _compute_features(
                        method=method,
                        residuals=residuals,
                        price_data=lookback_data,
                        lookback_start=lookback_start,
                        lookback_end=lookback_end,
                    )

                    if result is None:
                        continue
                    features = result[0]
                    token_list_used = result[1]

                    if features is not None and len(token_list_used) >= 2:
                        save_features_to_cache(
                            method=method,
                            week_start=week_start,
                            features=features,
                            token_list=token_list_used,
                            lookback_start=lookback_start,
                            lookback_end=lookback_end,
                            capm_results=(
                                capm_results if method != "mfdcca_raw" else None
                            ),
                        )
                        successful_counts[method] += 1

                except Exception as e:
                    logger.debug(
                        f"Week {week_num} ({week_start.date()}) {method} failed: {e}"
                    )
                    continue

        except Exception as e:  # ← 8 spaces, matches try
            logger.debug(f"Week {week_num} ({week_start.date()}) failed: {e}")
            continue

    # ═══════════════════════════════════════════════════════════
    # STEP 3: Report results
    # ═══════════════════════════════════════════════════════════
    logger.info(f"\n{'='*70}")
    logger.info(f"UNIFIED PRECOMPUTE COMPLETE")
    logger.info(f"{'='*70}")
    for method in methods:
        success_rate = (
            (successful_counts[method] / len(weekly_starts)) * 100
            if weekly_starts
            else 0
        )
        logger.info(
            f"  {method.upper()}: {successful_counts[method]}/{len(weekly_starts)} "
            f"weeks ({success_rate:.1f}%)"
        )
    logger.info(f"{'='*70}\n")

    return successful_counts


# Add this type alias near the top of precompute.py with the other type aliases
ComputeResult = Tuple[FeatureDict, List[str]]


def _compute_features(
    method: str,
    residuals: Dict[str, pd.Series],
    price_data: Dict[str, pd.DataFrame],
    lookback_start: pd.Timestamp,
    lookback_end: pd.Timestamp,
) -> Optional[ComputeResult]:
    """
    Returns (features, token_list_used) tuple.
    token_list_used is the list of tokens the features were computed from —
    for mfdcca it is residuals.keys(), for mfdcca_raw it is raw_returns.keys().
    """
    if method == "mfdcca":
        token_list = list(residuals.keys())
        features = extract_mfdcca_features(
            residuals=residuals,
            token_list=token_list,
            q_list=CONFIG["q_list"],
            lookback_start=lookback_start,
            lookback_end=lookback_end,
        )
        return features, token_list

    elif method == "mfdcca_raw":
        raw_returns: Dict[str, pd.Series] = {}
        for token, df in price_data.items():
            if token == CONFIG["market_index"]:
                continue
            closes = pd.Series(df["close"].values, index=df.index).dropna()
            if len(closes) > 1:
                log_ret = (closes / closes.shift(1)).apply(np.log).dropna()
                if len(log_ret) >= 50:
                    raw_returns[token] = log_ret
        if len(raw_returns) < 2:
            return None
        token_list = list(raw_returns.keys())
        features = extract_mfdcca_features(
            residuals=raw_returns,
            token_list=token_list,
            q_list=CONFIG["q_list"],
            lookback_start=lookback_start,
            lookback_end=lookback_end,
        )
        return features, token_list

    elif method == "dcca":
        token_list = list(residuals.keys())
        features = extract_dcca_features(
            residuals=residuals,
            token_list=token_list,
            lookback_start=lookback_start,
            lookback_end=lookback_end,
        )
        return features, token_list

    elif method == "pearson":
        token_list = list(residuals.keys())
        features = extract_pearson_features(
            residuals=residuals,
            token_list=token_list,
            lookback_start=lookback_start,
            lookback_end=lookback_end,
        )
        return features, token_list

    elif method == "cointegration":
        coint_tokens = [t for t in price_data if t != CONFIG["market_index"]]
        coint_price_data = {t: price_data[t] for t in coint_tokens}
        features = extract_cointegration_features(
            price_data=coint_price_data,
            token_list=coint_tokens,
            lookback_start=lookback_start,
            lookback_end=lookback_end,
        )
        return features, coint_tokens

    return None
