from joblib import Parallel, delayed
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from config import CONFIG

logger = logging.getLogger(__name__)

# Global cache for full data (loaded once)
_full_data_cache: Dict[str, pd.DataFrame] = {}


def load_single_token(file_path: Path, token: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(file_path)
        df = df[["Date", "Price"]].rename(columns={"Date": "date", "Price": "close"})

        # Parse dates
        df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
        df = df.dropna(subset=["date"])

        # Set date as index
        df = df.set_index("date").sort_index()

        # Filter weekdays (business days only)
        # Monday=0, Sunday=6 - keep only Monday-Friday (0-4)
        dates = pd.DatetimeIndex(df.index.to_numpy())
        df = df[dates.dayofweek < 5]

        # Clean prices
        df["close"] = pd.to_numeric(
            df["close"].astype(str).str.replace(",", ""), errors="coerce"
        )
        df = df.dropna(subset=["close"])

        if len(df) == 0:
            logger.warning(f"{token}: No data after cleaning")
            return None

        logger.debug(f"✅ {token}: Loaded {len(df)} days")
        return df

    except Exception as e:
        logger.error(f"Failed to load {token}: {e}")
        return None


def load_all_token_data_cached(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    market_index: str,
) -> Dict[str, pd.DataFrame]:
    """
    Load cleaned token data from cache, filter by period,
    align to INDEX business days, and return common-date dataset.
    """
    global _full_data_cache

    # ------------------------------------------------------------------
    # 1. LOAD & CACHE CLEANED DATA (CSV READ HAPPENS ONLY ONCE)
    # ------------------------------------------------------------------
    if not _full_data_cache:
        logger.info("📥 Loading full data cache from CSV files...")
        tokens_to_load = list(set(CONFIG["token_names"] + [market_index]))

        results = Parallel(n_jobs=-1)(
            delayed(load_single_token)(CONFIG["data_dir"] / f"{token}.csv", token)
            for token in tokens_to_load
        )

        for token, df in zip(tokens_to_load, results):
            if df is not None:
                _full_data_cache[token] = df

        logger.info(f"✅ Cached {len(_full_data_cache)} cleaned datasets")

    if market_index not in _full_data_cache:
        logger.error(f"❌ Market index '{market_index}' missing from cache")
        return {}

    # ------------------------------------------------------------------
    # 2. FILTER INDEX TO REQUESTED PERIOD
    # ------------------------------------------------------------------
    index_data = _full_data_cache[market_index].loc[start_date:end_date]

    if index_data.empty:
        logger.error(
            f"❌ No INDEX data between {start_date.date()} and {end_date.date()}"
        )
        return {}

    index_dates = index_data.index
    logger.info(
        f"📅 INDEX range: {index_dates[0].date()} → {index_dates[-1].date()} "
        f"({len(index_dates)} business days)"
    )

    # ------------------------------------------------------------------
    # 3. ALIGN TOKENS & INTERSECT COMMON DATES (single, clean)
    # ------------------------------------------------------------------
    aligned_data = {market_index: index_data}
    aligned_tokens = []

    for token in CONFIG["token_names"]:
        if token not in _full_data_cache:
            continue

        token_data = _full_data_cache[token].loc[start_date:end_date]
        if token_data.empty:
            continue

        aligned_data[token] = token_data
        aligned_tokens.append(token)

    # True intersection for strictly common business days
    common_dates = index_data.index
    for token in aligned_tokens:
        common_dates = common_dates.intersection(aligned_data[token].index)

    if common_dates.empty:
        logger.error("❌ No common dates across all assets")
        return {}

    common_dates = common_dates.sort_values()

    # Final strictly-aligned dataset
    final_data = {token: df.loc[common_dates] for token, df in aligned_data.items()}

    logger.info(
        f"✅ Final dataset ready | Tokens: {len(final_data)} | Days: {len(common_dates)}"
    )

    return final_data


def validate_data_files(
    data_dir: Path, token_list: List[str], market_index: str
) -> Tuple[bool, List[str]]:
    """Validate data files exist"""
    missing_files = []
    all_tokens = token_list + [market_index]

    for token in all_tokens:
        file_path = data_dir / f"{token}.csv"
        if not file_path.exists():
            missing_files.append(token)

    if missing_files:
        logger.error(f"❌ Missing files: {missing_files}")
        return False, missing_files

    logger.info(f"✅ All {len(all_tokens)} data files found")
    return True, []
