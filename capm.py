"""
CAPM
"""

import pandas as pd
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from config import CONFIG, DEVICE

logger = logging.getLogger(__name__)


def apply_capm_filter(tokens, market_index, price_data):
    """
    ✅ SIMPLIFIED: Trust data_processing gave us aligned data
    NO redundant validation - data is guaranteed aligned by load_all_token_data_cached()
    """
    if market_index not in price_data:
        logger.warning(f"Market index {market_index} not found in price_data")
        return {}

    # Define valid tokens (must exist in price_data and not be the market index)
    valid_tokens = [t for t in tokens if t in price_data and t != market_index]

    if not valid_tokens:
        logger.warning("No valid tokens found for CAPM filter")
        return {}

    # Get aligned dates from INDEX (data_processing already ensured alignment)
    common_dates = price_data[market_index].index

    logger.info(
        f"CAPM Filter: {len(valid_tokens)} tokens, {len(common_dates)} aligned days"
    )

    # Extract prices directly (no validation needed - data already clean)
    index_prices = price_data[market_index]["close"].to_numpy(dtype=np.float32)
    token_prices_list = [
        price_data[token]["close"].to_numpy(dtype=np.float32) for token in valid_tokens
    ]

    # Batch CAPM computation on GPU
    token_prices_stack = torch.tensor(
        np.vstack(token_prices_list), device=DEVICE, dtype=torch.float32
    )
    index_prices_gpu = torch.tensor(index_prices, device=DEVICE, dtype=torch.float32)

    betas, alphas, residuals_stack = compute_capm(token_prices_stack, index_prices_gpu)

    # Create results dictionary
    capm_results = {}
    residual_dates = common_dates[1:]  # Returns are 1 day shorter than prices

    for i, token in enumerate(valid_tokens):
        capm_results[token] = {
            "beta": float(betas[i].item()),
            "alpha": float(alphas[i].item()),
            "residuals": pd.Series(
                residuals_stack[i].cpu().numpy(),
                index=residual_dates,
                name=f"{token}_residuals",
            ),
            "common_days_used": len(residual_dates),
            "start_date": residual_dates[0],
            "end_date": residual_dates[-1],
        }

    logger.info(f"✅ CAPM complete: {len(capm_results)} tokens processed")

    # Log summary statistics
    if capm_results:
        beta_vals = [result["beta"] for result in capm_results.values()]
        alpha_vals = [result["alpha"] for result in capm_results.values()]
        logger.debug(f"   Beta range: [{min(beta_vals):.3f}, {max(beta_vals):.3f}]")
        logger.debug(f"   Alpha range: [{min(alpha_vals):.4f}, {max(alpha_vals):.4f}]")

    return capm_results


def compute_capm(token_prices_stack, index_prices):
    """
    Compute CAPM using simple returns

    Model: r_i - r_f = α_i + β_i(r_m - r_f) + ε_i

    Args:
        token_prices_stack: Tensor of shape (n_tokens, n_days)
        index_prices: Tensor of shape (n_days,)

    Returns:
        betas: Tensor of shape (n_tokens,)
        alphas: Tensor of shape (n_tokens,)
        residuals: Tensor of shape (n_tokens, n_days-1)
    """
    rf_annual = CONFIG.get("risk_free_rate", 0.0)
    rf_daily = rf_annual / 252

    # Calculate simple returns
    token_returns = (token_prices_stack[:, 1:] / token_prices_stack[:, :-1]) - 1
    index_returns = (index_prices[1:] / index_prices[:-1]) - 1

    # Excess returns (subtract risk-free rate)
    token_excess = token_returns - rf_daily
    market_excess = index_returns - rf_daily

    n_tokens, n_obs = token_excess.shape

    # Design matrix for OLS: [1, r_m - r_f]
    X = torch.stack(
        [torch.ones(n_obs, device=DEVICE, dtype=torch.float32), market_excess], dim=1
    )

    # Batch OLS regression for all tokens simultaneously
    X_batch = X.unsqueeze(0).expand(n_tokens, -1, -1)  # (n_tokens, n_obs, 2)
    y_batch = token_excess.unsqueeze(-1)  # (n_tokens, n_obs, 1)

    # Solve: β = (X'X)^(-1)X'y
    coeffs = torch.linalg.lstsq(X_batch, y_batch).solution.squeeze(-1)
    alphas = coeffs[:, 0]
    betas = coeffs[:, 1]

    # Calculate residuals: ε_i = (r_i - r_f) - (α_i + β_i(r_m - r_f))
    predicted = alphas.unsqueeze(1) + betas.unsqueeze(1) * market_excess.unsqueeze(0)
    residuals = token_excess - predicted

    return betas, alphas, residuals
