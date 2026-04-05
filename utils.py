"""
utils.py - Utility functions
"""

import pandas as pd
import logging
import numpy as np
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> Path:
    """
    Create directory if it doesn't exist and return the path.

    Args:
        path: Path object to directory

    Returns:
        The same path object (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_trading_weeks(all_dates, start_date, end_date):
    """
    Generate canonical week start dates using ISO week calendar.

    Algorithm:
    1. Find all ISO weeks that have trading days in [start_date, end_date]
    2. For each ISO week, get its Monday
    3. Find the first trading day on or after that Monday (from FULL dataset)
    4. Only include if that trading day falls within [start_date, end_date]

    This ensures deterministic results regardless of how dates are filtered.

    Args:
        all_dates: DatetimeIndex of ALL available trading days (including lookback)
        start_date: Evaluation period start
        end_date: Evaluation period end

    Returns:
        Sorted list of pd.Timestamp representing week starts
    """
    all_dates = pd.DatetimeIndex(all_dates).sort_values().unique()

    # Get dates within evaluation period
    eval_dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)]

    if len(eval_dates) == 0:
        logger.warning(
            f"No trading dates in range {start_date.date()} to {end_date.date()}"
        )
        return []

    weeks = []
    seen_weeks = set()

    for date in eval_dates:
        # Get ISO week identifier
        iso_year, iso_week, _ = date.isocalendar()
        week_key = (iso_year, iso_week)

        if week_key in seen_weeks:
            continue
        seen_weeks.add(week_key)

        # Calculate Monday of this ISO week
        try:
            monday = pd.Timestamp.fromisocalendar(iso_year, iso_week, 1)
        except Exception as e:
            logger.warning(
                f"Could not calculate Monday for {iso_year}-W{iso_week}: {e}"
            )
            continue

        # ⭐ KEY FIX: Find first trading day >= Monday from FULL dataset
        candidates = all_dates[all_dates >= monday]

        if len(candidates) == 0:
            continue

        week_start = candidates[0]

        # Only include if week_start is within evaluation period
        if start_date <= week_start <= end_date:
            weeks.append(week_start)

    weeks = sorted(weeks)

    logger.info(
        f"Generated {len(weeks)} trading weeks: "
        f"{weeks[0].date() if weeks else 'N/A'} to {weeks[-1].date() if weeks else 'N/A'}"
    )

    return weeks
