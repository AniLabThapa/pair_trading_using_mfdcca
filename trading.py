import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple, Set

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, return path"""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================
# DIVERGENCE FILTER (STEP 8)
# ============================================================


def apply_divergence_filter(
    candidate_pairs: List[Tuple[str, str]],
    price_data: Dict[str, pd.DataFrame],
    lookback_days: int,
    divergence_threshold: float,
) -> List[Dict[str, Any]]:
    """
    Filter pairs based on recent PRICE divergence.

    """
    filtered_pairs = []

    for token1, token2 in candidate_pairs:
        if token1 not in price_data or token2 not in price_data:
            continue

        df1, df2 = price_data[token1], price_data[token2]

        # Use the smaller of lookback_days+1 or available days
        effective_lookback = min(lookback_days + 1, len(df1), len(df2))
        if effective_lookback < 2:
            # Need at least 2 days to compute return
            continue

        # Extract last effective_lookback prices
        prices1 = df1["close"].iloc[-effective_lookback:].values
        prices2 = df2["close"].iloc[-effective_lookback:].values

        # Calculate cumulative returns
        cum_ret1 = prices1[-1] / prices1[0] - 1.0
        cum_ret2 = prices2[-1] / prices2[0] - 1.0

        # Calculate divergence
        divergence = abs(cum_ret1 - cum_ret2)

        # Filter by divergence threshold
        if divergence < divergence_threshold:
            continue

        # Long underperformer, short outperformer
        if cum_ret1 < cum_ret2:
            long_token, short_token = token1, token2
        else:
            long_token, short_token = token2, token1

        filtered_pairs.append(
            {
                "pair": (token1, token2),
                "long_token": long_token,
                "short_token": short_token,
                "divergence": divergence,
            }
        )

    return filtered_pairs


def simulate_pair_trades(
    filtered_pairs: List[Dict[str, Any]],
    price_data: Dict[str, pd.DataFrame],
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
    transaction_cost: float = 0.001,  # 0.1% per side
) -> Dict[str, Any]:

    total_pairs_selected = len(filtered_pairs)

    if not price_data or total_pairs_selected == 0:
        return {
            "Weekly_Return": 0.0,
            "Daily_Returns": pd.Series(dtype=float),
            "Total_Pairs_Selected": total_pairs_selected,
            "Active_Pairs": set(),
        }

    trading_days = next(iter(price_data.values())).index
    week_days = trading_days[(trading_days >= week_start) & (trading_days <= week_end)]

    if len(week_days) < 2:
        return {
            "Weekly_Return": 0.0,
            "Daily_Returns": pd.Series(dtype=float),
            "Total_Pairs_Selected": total_pairs_selected,
            "Active_Pairs": set(),
        }

    n_days = len(week_days)
    daily_returns: List[float] = []
    day_traded = set()

    traded_this_week = set()

    for i in range(1, n_days):

        prev_day, curr_day = week_days[i - 1], week_days[i]
        day_ret_sum = 0.0
        day_traded = set()

        for p in filtered_pairs:

            l, s = p["long_token"], p["short_token"]

            if (
                l not in price_data
                or s not in price_data
                or prev_day not in price_data[l].index
                or curr_day not in price_data[l].index
                or prev_day not in price_data[s].index
                or curr_day not in price_data[s].index
            ):
                continue

            pair = tuple(sorted([l, s]))

            day_traded.add(pair)
            traded_this_week.add(pair)

            p_l_prev, p_l_curr = price_data[l].loc[[prev_day, curr_day], "close"]
            p_s_prev, p_s_curr = price_data[s].loc[[prev_day, curr_day], "close"]

            r_long = p_l_curr / p_l_prev - 1.0
            r_short = p_s_curr / p_s_prev - 1.0

            pair_ret = (r_long - r_short) / 2.0

            day_ret_sum += pair_ret

        valid_pairs = len(day_traded)

        portfolio_day_ret = day_ret_sum / valid_pairs if valid_pairs > 0 else 0.0

        if valid_pairs > 0:
            # Entry cost on first trading day
            if i == 1:
                portfolio_day_ret -= 2.0 * transaction_cost

            # Exit cost on last trading day
            if i == n_days - 1 and n_days > 2:
                portfolio_day_ret -= 2.0 * transaction_cost

        daily_returns.append(portfolio_day_ret)

    daily_returns_series = pd.Series(daily_returns, index=week_days[1:])

    # Weekly return now computed from cost-inclusive daily returns
    weekly_return = (
        float(np.prod(1.0 + np.array(daily_returns)) - 1.0) if daily_returns else 0.0
    )

    # ✅ No separate cost subtraction needed — already embedded in daily_returns
    return {
        "Weekly_Return": weekly_return,
        "Daily_Returns": daily_returns_series,
        "Total_Pairs_Selected": total_pairs_selected,
        "Active_Pairs": traded_this_week,
    }


# ============================================================
# EMPTY METRICS
# ============================================================


def create_empty_metrics() -> Dict[str, Any]:
    return {
        "Mean_Return": 0.0,
        "Std_Dev": 0.0,
        "Sharpe_Ratio": 0.0,
        "Sortino_Ratio": 0.0,
        "Max_Drawdown_%": 0.0,
        "Calmar_Ratio": 0.0,
        "Profit_Factor": 0.0,
        "Gross_Profit": 0.0,
        "Gross_Loss": 0.0,
        "Daily_Return_Series": pd.Series(dtype=float),
    }


def calculate_performance_metrics(
    weekly_results_list: List[Dict[str, Any]],
    result_dir: Path,
    period_name: str,
    trading_days_per_year: float = 252.0,
    risk_free_rate_annual: float = 0.0,
) -> Dict[str, Any]:
    """
    Calculate performance metrics for trading strategies or benchmarks.

    Args:
        weekly_results_list: List of weekly trading results
        result_dir: Directory to save results
        period_name: Name of evaluation period (used to detect benchmarks)
        trading_days_per_year: Trading days per year (default 252)
        risk_free_rate_annual: Annual risk-free rate (default 0.0)

    Returns:
        Dictionary of performance metrics
    """

    # 1. CONCATENATE DATA
    daily_series = [
        w["Daily_Returns"]
        for w in weekly_results_list
        if isinstance(w.get("Daily_Returns"), pd.Series)
        and not w["Daily_Returns"].empty
    ]

    if not daily_series:
        return create_empty_metrics()

    returns_series = pd.concat(daily_series).sort_index()
    returns_series = returns_series[~returns_series.index.duplicated()].dropna()
    returns = returns_series.to_numpy()
    n_days = len(returns)

    if n_days < 2:
        return create_empty_metrics()

    # 2. RISK FREE ADJUSTMENT
    rf_daily = (1 + risk_free_rate_annual) ** (1 / trading_days_per_year) - 1.0
    excess = returns - rf_daily

    # 3. PERFORMANCE MATH
    mean_daily = np.mean(returns)
    std_daily = np.std(returns, ddof=1)

    # Sharpe with Stability Check
    std_excess = np.std(excess, ddof=1)
    sharpe = (
        (np.mean(excess) / std_excess * np.sqrt(trading_days_per_year))
        if std_excess > 1e-9
        else 0.0
    )

    # Sortino (Downside Risk Only)
    downside_returns = excess[excess < 0]
    # We use the full n_days in the denominator for the downside deviation (Standard Way)
    downside_deviation = np.sqrt(np.sum(downside_returns**2) / n_days)
    sortino = (
        (np.mean(excess) / downside_deviation * np.sqrt(trading_days_per_year))
        if downside_deviation > 1e-9
        else 0.0
    )

    # 4. DRAWDOWN & CAGR
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / running_max
    max_dd = np.max(drawdowns)

    # CAGR
    cagr = (cumulative[-1] ** (trading_days_per_year / n_days)) - 1.0
    calmar = cagr / max_dd if max_dd > 0 else 0.0

    # 5. PROFIT FACTOR
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # 6. TURNOVER CALCULATION - DISTINGUISH BENCHMARKS FROM PAIR TRADING
    if period_name in ["btc_benchmark", "index_benchmark"]:
        # Buy-and-hold benchmarks have ZERO turnover
        annualized_turnover = 0.0
        logger.debug(f"{period_name}: Turnover set to 0.00% (buy-and-hold strategy)")
    else:
        # Pair trading strategies: Calculate actual turnover
        weekly_pair_sets = [set(w.get("Active_Pairs", [])) for w in weekly_results_list]

        weekly_turnovers = []
        prev_pairs = None

        for current_pairs in weekly_pair_sets:
            if prev_pairs is None:
                prev_pairs = current_pairs
                continue

            # Symmetric difference: Elements in either prev or current, but not both
            changes = len(current_pairs.symmetric_difference(prev_pairs))

            # Average size over the transition period
            avg_size = (len(current_pairs) + len(prev_pairs)) / 2.0

            # Normalizing by 2.0 * avg_size ensures 100% = full portfolio replacement
            weekly_turnover = changes / (2.0 * avg_size) if avg_size > 0 else 0.0

            weekly_turnovers.append(weekly_turnover)
            prev_pairs = current_pairs

        mean_weekly_turnover = np.mean(weekly_turnovers) if weekly_turnovers else 0.0
        annualized_turnover = mean_weekly_turnover * 52

        logger.debug(
            f"{period_name}: Calculated turnover = {annualized_turnover * 100:.2f}%"
        )

    # 7. EXPORT & RETURN
    ensure_dir(result_dir)

    return {
        "Mean_Return": mean_daily * 100,
        "Std_Dev": std_daily * 100,
        "Downside_Deviation": round(downside_deviation * 100, 6),
        "Sharpe_Ratio": round(sharpe, 4),
        "Sortino_Ratio": round(sortino, 4),
        "Max_Drawdown_%": round(max_dd * 100, 2),
        "Calmar_Ratio": round(calmar, 4),
        "Profit_Factor": round(profit_factor, 4),
        "Gross_Profit": gross_profit * 100,
        "Gross_Loss": gross_loss * 100,
        "Daily_Return_Series": returns_series,
    }
