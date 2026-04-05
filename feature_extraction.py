"""
Unified Feature Extraction Layer
TRUSTS DATA ALIGNMENT from data_processing.py
"""

import torch
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from config import CONFIG, DEVICE
from mfdcca import process_token_pairs, extract_hurst_matrices

import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

logger = logging.getLogger(__name__)


# ============================================================================
# MFDCCA FEATURE EXTRACTION
# ============================================================================


def extract_mfdcca_features(
    residuals: Dict[str, pd.Series],
    token_list: List[str],
    q_list: List[float],
    lookback_start: pd.Timestamp,
    lookback_end: pd.Timestamp,
) -> Dict[str, Any]:
    """
    Extract MF-DCCA features from pre-filtered residuals.
    """
    logger.info(f" MFDCCA: {lookback_start.date()} to {lookback_end.date()}")

    valid_tokens = [token for token in token_list if token in residuals]

    # Run MFDCCA directly on residuals
    results = process_token_pairs(
        token_list=valid_tokens,
        residuals=residuals,
        q_list=q_list,
    )

    if not results:
        logger.warning("MFDCCA: No results generated")
        return {"has_data": False}

    # Extract matrices
    hxy_matrix, delta_H_matrix, delta_alpha_matrix = extract_hurst_matrices(
        token_list=valid_tokens, results=results, q_list=q_list
    )

    logger.info(f"✅ MFDCCA: {len(results)} pairs analyzed")

    return {
        "has_data": True,
        "hxy_matrix": hxy_matrix,
        "delta_H_matrix": delta_H_matrix,
        "delta_alpha_matrix": delta_alpha_matrix,
        "q_list": q_list,
        "num_pairs": len(results),
        "token_list": valid_tokens,
        "lookback_start": lookback_start,
        "lookback_end": lookback_end,
    }


# ============================================================================
# DCCA FEATURE EXTRACTION - FIXED
# ============================================================================
def extract_dcca_features(
    residuals: Dict[str, pd.Series],
    token_list: List[str],
    lookback_start: pd.Timestamp,
    lookback_end: pd.Timestamp,
    min_scale: int = 10,
    num_scales: int = 20,
    eps: float = 1e-12,
    device: Optional[torch.device] = None,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Extract DCCA features from pre-validated residuals.
    NOTE: Residuals are already validated and aligned by CAPM.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f" Running DCCA on {device}")
    logger.info(f"DCCA Period: {lookback_start.date()} to {lookback_end.date()}")

    # 1. Profile Creation with Mean Removal
    profiles = {}
    for token in token_list:
        # Convert to tensor
        data = torch.tensor(residuals[token].values, device=device, dtype=torch.float32)
        # Explicit Mean Removal
        centered_data = data - torch.mean(data)
        # Cumulative Sum to create the Profile (Random Walk)
        profiles[token] = torch.cumsum(centered_data, dim=0)

    logger.info(f"DCCA: Processing {len(profiles)} tokens")

    valid_tokens = list(profiles.keys())

    logger.info(f"DCCA: Processing {len(valid_tokens)} tokens")

    N = len(next(iter(profiles.values())))

    max_scale = min(int(N / 4), N - 5)
    if max_scale <= min_scale:
        logger.warning("DCCA: Invalid scale range")
        return {}

    # --------------------------------------------------
    # 2. Logarithmic scale selection
    # --------------------------------------------------
    if min_scale >= max_scale:
        min_scale = max_scale // 2

    scales = np.logspace(
        np.log10(float(min_scale)),
        np.log10(float(max_scale)),
        num=num_scales,
        dtype=int,
    )
    scales = np.unique(scales)
    scales = [int(s) for s in scales if (N - s + 1) >= 5]

    if len(scales) < 4:
        logger.warning(f"DCCA: Insufficient scales (N={N}, scales={scales})")
        return {}

    # --------------------------------------------------
    # 3. Batch detrend function
    # --------------------------------------------------
    def batch_detrend(segments: torch.Tensor) -> torch.Tensor:
        """
        Batch detrend segments using linear least squares.

        Parameters
        ----------
        segments : torch.Tensor
            Shape (n_boxes, s)

        Returns
        -------
        torch.Tensor
            Detrended segments, same shape
        """
        n_boxes, s = segments.shape

        # Create time indices for linear regression
        t = torch.arange(s, device=device, dtype=torch.float32)

        # Design matrix for linear regression: [t, 1]
        A = torch.stack([t, torch.ones_like(t)], dim=1)  # (s, 2)
        ATA = A.T @ A  # (2, 2)

        # Solve for all boxes simultaneously
        seg_reshaped = segments.unsqueeze(-1)  # (n_boxes, s, 1)
        ATb = torch.matmul(A.T.unsqueeze(0), seg_reshaped)  # (n_boxes, 2, 1)
        ATA_inv = torch.linalg.inv(ATA)  # (2, 2)
        theta = torch.matmul(ATA_inv.unsqueeze(0), ATb)  # (n_boxes, 2, 1)

        # Compute trends: trend = A @ theta
        theta_expanded = theta.squeeze(-1).unsqueeze(1)  # (n_boxes, 1, 2)
        A_expanded = A.unsqueeze(0)  # (1, s, 2)
        trend = torch.matmul(
            A_expanded, theta_expanded.transpose(1, 2)
        )  # (n_boxes, s, 1)

        # Detrend
        detrended = segments - trend.squeeze(-1)  # (n_boxes, s)

        return detrended

    # --------------------------------------------------
    # 4. Pairwise DCCA (Pure DCCA with H_xy)
    # --------------------------------------------------
    dcca_features = {}

    for i in range(len(valid_tokens)):
        for j in range(i + 1, len(valid_tokens)):
            t1, t2 = valid_tokens[i], valid_tokens[j]
            X = profiles[t1]
            Y = profiles[t2]

            F2_dcca_vals = []
            F2_dfa_x_vals = []
            F2_dfa_y_vals = []
            rho_vals = []
            used_scales = []

            for s in scales:
                n_boxes = N - s + 1

                # Extract overlapping boxes
                X_2d = X.unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
                Y_2d = Y.unsqueeze(0).unsqueeze(-1)

                seg_x = (
                    torch.nn.functional.unfold(X_2d, kernel_size=(s, 1), stride=(1, 1))
                    .squeeze(0)
                    .T
                )  # (n_boxes, s)

                seg_y = (
                    torch.nn.functional.unfold(Y_2d, kernel_size=(s, 1), stride=(1, 1))
                    .squeeze(0)
                    .T
                )  # (n_boxes, s)

                # Batch detrend
                dx = batch_detrend(seg_x)
                dy = batch_detrend(seg_y)

                # Local fluctuations (Eq. 3) - use (s-1) divisor
                f2_dfa_x = (dx**2).sum(dim=1) / (s - 1)
                f2_dfa_y = (dy**2).sum(dim=1) / (s - 1)
                f2_dcca = (dx * dy).sum(dim=1) / (s - 1)

                # Scale-averaged (Eq. 4)
                F2_x = f2_dfa_x.mean()
                F2_y = f2_dfa_y.mean()
                F2_xy = f2_dcca.mean()

                # DCCA correlation coefficient (Eq. 5)
                denom = torch.sqrt(F2_x * F2_y)
                rho = F2_xy / (denom + eps)
                rho = torch.clamp(rho, -1.0, 1.0)

                # Store results
                F2_dcca_vals.append(F2_xy.item())
                F2_dfa_x_vals.append(F2_x.item())
                F2_dfa_y_vals.append(F2_y.item())
                rho_vals.append(rho.item())
                used_scales.append(s)

            if len(used_scales) < 4:
                continue

            # --------------------------------------------------
            # 5. H_xy estimation from scaling law
            # F²_DCCA(s) ~ s^(2H_xy)
            # --------------------------------------------------
            log_s = torch.log(
                torch.tensor(used_scales, device=device, dtype=torch.float32)
            )
            F2_tensor = torch.tensor(F2_dcca_vals, device=device)

            log_F2 = torch.log(F2_tensor)

            A = torch.stack([log_s, torch.ones_like(log_s)], dim=1)
            theta = torch.linalg.lstsq(A, log_F2).solution

            H_xy = (theta[0] / 2.0).item()

            # Store pure DCCA features (no quality score, no R²)
            pair_key = tuple(sorted([t1, t2]))
            dcca_features[pair_key] = {
                "H_xy": H_xy,
                "mean_rho": float(np.mean(rho_vals)),
                "mean_abs_rho": float(np.mean(np.abs(rho_vals))),
                "rho_by_scale": dict(zip(used_scales, rho_vals)),
                "F2_dcca_by_scale": dict(zip(used_scales, F2_dcca_vals)),
                "scales_used": used_scales,
            }

    logger.info(f"✅ DCCA completed: {len(dcca_features)} pairs")
    return dcca_features


# ============================================================================
# PEARSON FEATURE EXTRACTION
# ============================================================================
def extract_pearson_features(
    residuals: Dict[str, pd.Series],
    token_list: List[str],
    lookback_start: pd.Timestamp,
    lookback_end: pd.Timestamp,
) -> Dict[str, Any]:

    logger.info(f"📈 Pearson: {lookback_start.date()} to {lookback_end.date()}")

    # ✅ SIMPLIFIED - token_list matches residuals.keys()
    res_stack = torch.stack(
        [
            torch.tensor(residuals[token].values, device=DEVICE, dtype=torch.float32)
            for token in token_list
        ]
    )

    logger.info(f"Pearson: Processing {len(token_list)} tokens")

    try:
        corr_matrix = torch.corrcoef(res_stack)
        logger.info(f"✅ Pearson: {len(token_list)} tokens, {res_stack.shape[1]} days")

        return {
            "has_data": True,
            "correlation_matrix": corr_matrix,
            "token_list": token_list,
            "n_observations": res_stack.shape[1],
        }
    except Exception as e:
        logger.error(f"Pearson failed: {e}")
        return {"has_data": False}


# ============================================================================
# COINTEGRATION FEATURE EXTRACTION
# ============================================================================
# feature_extraction.py - extract_cointegration_features()


def extract_cointegration_features(
    price_data: Dict[str, pd.DataFrame],
    token_list: List[str],
    lookback_start: pd.Timestamp,
    lookback_end: pd.Timestamp,
) -> Dict[Tuple[str, str], Dict[str, Any]]:

    logger.info("🔗 Cointegration: Engle-Granger test...")

    clean_data = {
        token: np.log(
            price_data[token].loc[lookback_start:lookback_end, "close"].values
        )
        for token in token_list
        if token in price_data
    }

    n = len(clean_data)
    if n < 2:
        logger.warning(f"Cointegration: Insufficient tokens ({n} < 2)")
        return {}

    valid_tokens = list(clean_data.keys())
    n_pairs = n * (n - 1) // 2
    features = {}
    valid_pairs = 0

    # --- WITHIN THE NESTED LOOP ---
    for i in range(n):
        for j in range(i + 1, n):
            t_original_1, t_original_2 = valid_tokens[i], valid_tokens[j]
            y, x = clean_data[t_original_1], clean_data[t_original_2]

            try:
                # Keep your existing logic for testing both directions
                t1_stat, p1, crit1 = coint(y, x, autolag="BIC")
                t2_stat, p2, crit2 = coint(x, y, autolag="BIC")

                if np.isnan(p1) or np.isnan(p2):
                    continue

                # Selection of stronger p-value
                if p1 < p2:
                    pvalue, t_stat, crit = p1, t1_stat, crit1
                else:
                    pvalue, t_stat, crit = p2, t2_stat, crit2

                if np.isnan(pvalue) or np.isinf(pvalue):
                    continue

                # ✅ THE FIX: Create a canonical (alphabetical) key
                pair_key = tuple(sorted([t_original_1, t_original_2]))

                features[pair_key] = {
                    "pvalue": float(pvalue),
                    "t_stat": float(t_stat),
                    "crit_1pct": float(crit[0]),
                    "crit_5pct": float(crit[1]),
                    "crit_10pct": float(crit[2]),
                    "n_obs": len(y),
                }
                valid_pairs += 1

            except Exception:
                continue

    logger.info(f"✅ Cointegration: {valid_pairs}/{n_pairs} valid tests")
    return features
