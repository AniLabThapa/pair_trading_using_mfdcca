import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from statsmodels.tsa.stattools import adfuller
from typing import Dict, List, Any, cast


def compute_asset_statistics(all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    results: List[Dict[str, Any]] = []

    for ticker, df in all_data.items():
        if "close" not in df.columns:
            continue

        # Cast log_prices to Series so Pylance knows .diff() exists
        log_prices = cast(pd.Series, np.log(df["close"]))
        returns = log_prices.diff()
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

        mean = float(cast(float, returns.mean()))
        max_ = float(cast(float, returns.max()))
        min_ = float(cast(float, returns.min()))
        std = float(cast(float, returns.std(ddof=1)))
        skew = float(cast(float, returns.skew()))
        kurt = float(cast(float, returns.kurtosis()))

        # ANALYSIS FIX: Scipy/Statsmodels return types
        # Accessing by index [0] and [1] is safer for type inference
        jb_result = jarque_bera(returns.values)
        jb_stat = float(cast(float, jb_result[0]))

        adf_result = adfuller(returns.values, autolag="AIC")
        adf_stat = float(cast(float, adf_result[0]))

        results.append(
            {
                "Ticker": ticker,
                "Mean": round(mean, 4),
                "Max": round(max_, 4),
                "Min": round(min_, 4),
                "Std.": round(std, 4),
                "Skewness": round(skew, 4),
                "Kurtosis": round(kurt, 4),
                "J-B test ": round(jb_stat, 2),
                "ADF test ": round(adf_stat, 2),
            }
        )

    return pd.DataFrame(results)
