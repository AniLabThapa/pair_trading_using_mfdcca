"""
Pair Selection Module
=================================================
"""

import logging
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional
from config import DEVICE

logger = logging.getLogger(__name__)


def ensure_gpu_tensor(data: Any) -> Optional[torch.Tensor]:
    """Convert data to GPU tensor with validation"""
    if data is None:
        return None

    try:
        if isinstance(data, torch.Tensor):
            return data.to(DEVICE)
        elif isinstance(data, np.ndarray):
            return torch.tensor(data, device=DEVICE, dtype=torch.float32)
        else:
            return torch.tensor(data, device=DEVICE, dtype=torch.float32)
    except Exception as e:
        logger.error(f"Cannot convert to GPU tensor: {type(data)} - {e}")
        return None


def select_pairs_mfdcca(
    features: Dict[str, Any],
    pair_hxy_threshold: float,
    threshold_h: float,
    threshold_alpha: float,
) -> List[Tuple[str, str]]:
    """

    Select pairs using MF-DCCA criteria:
    - H_xy(2) < threshold (mean-reversion)
    - ΔH < threshold (limited multifractality)
    - Δα < threshold (limited singularity spread)
    """

    if not features.get("has_data", False):
        logger.warning("MF-DCCA: No features available")
        return []

    # ✅ Always use token_list from features
    token_list = features.get("token_list")
    if token_list is None or len(token_list) < 2:
        logger.warning("MF-DCCA: No token_list in features")
        return []

    # Extract matrices
    hxy_matrix = ensure_gpu_tensor(features.get("hxy_matrix"))
    delta_H_matrix = ensure_gpu_tensor(features.get("delta_H_matrix"))
    delta_alpha_matrix = ensure_gpu_tensor(features.get("delta_alpha_matrix"))

    # ✅ FIX 1: Add null safety checks
    if hxy_matrix is None or delta_H_matrix is None or delta_alpha_matrix is None:
        logger.warning("MF-DCCA: Missing matrices")
        return []

    n_tokens = len(token_list)

    # ✅ FIX 2: Verify matrix dimensions
    if hxy_matrix.shape != (n_tokens, n_tokens):
        logger.error(
            f"MF-DCCA: Matrix size mismatch! "
            f"token_list={n_tokens}, matrix={hxy_matrix.shape}"
        )
        return []

    # ✅ FIX 3: Convert thresholds to GPU tensors ONCE
    pair_hxy_threshold_gpu = torch.tensor(
        pair_hxy_threshold, device=DEVICE, dtype=torch.float32
    )
    threshold_h_gpu = torch.tensor(threshold_h, device=DEVICE, dtype=torch.float32)
    threshold_alpha_gpu = torch.tensor(
        threshold_alpha, device=DEVICE, dtype=torch.float32
    )

    # Get upper-triangular indices
    i_idx, j_idx = torch.triu_indices(n_tokens, n_tokens, offset=1, device=DEVICE)

    # ✅ FIX 4: Apply thresholds with GPU tensors (now safe - matrices verified non-null)
    valid_mask = (
        ~torch.isnan(hxy_matrix[i_idx, j_idx])
        & ~torch.isnan(delta_H_matrix[i_idx, j_idx])
        & ~torch.isnan(delta_alpha_matrix[i_idx, j_idx])
        & (hxy_matrix[i_idx, j_idx] < pair_hxy_threshold_gpu)
        & (delta_H_matrix[i_idx, j_idx] < threshold_h_gpu)
        & (delta_alpha_matrix[i_idx, j_idx] < threshold_alpha_gpu)
    )

    valid_i = i_idx[valid_mask]
    valid_j = j_idx[valid_mask]

    # Convert indices to token pairs
    selected_pairs = [
        (token_list[i], token_list[j])
        for i, j in zip(valid_i.cpu().tolist(), valid_j.cpu().tolist())
    ]

    logger.info(f"✅ MF-DCCA: {len(selected_pairs)} pairs selected")
    logger.info(
        f"   Thresholds: H_xy<{pair_hxy_threshold:.2f}, "
        f"ΔH<{threshold_h:.2f}, Δα<{threshold_alpha:.2f}"
    )

    return selected_pairs


def select_pairs_dcca(
    features: Dict[Tuple[str, str], Dict[str, Any]],
    pair_hxy_threshold: float,
    min_mean_abs_rho: float = 0.5,
    token_list: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """
    Pure DCCA pair selection based only on H_xy.
    """

    if not features:
        logger.warning("DCCA: No features available")
        return []

    token_filter = set(token_list) if token_list else None
    selected_pairs = []

    for (t1, t2), feat in features.items():

        if token_filter and (t1 not in token_filter or t2 not in token_filter):
            continue

        H_xy = feat.get("H_xy")
        mean_abs_rho = feat.get("mean_abs_rho", 0)

        # 🔹 Safety check
        if H_xy is None:
            continue

        if not np.isfinite(H_xy):
            continue

        # Mean-reverting condition
        # Pure DCCA criteria: Mean-reverting + Strong correlation
        if H_xy < pair_hxy_threshold and mean_abs_rho >= min_mean_abs_rho:
            selected_pairs.append(tuple(sorted([t1, t2])))

    logger.info(f"✅ DCCA: {len(selected_pairs)} pairs selected")
    logger.info(
        f"   Criteria: H_xy < {pair_hxy_threshold:.2f}, |ρ| ≥ {min_mean_abs_rho:.2f}"
    )

    return selected_pairs


def select_pairs_pearson(
    features: Dict[str, Any],
    rho_threshold: float,
) -> List[Tuple[str, str]]:
    """
    Select pairs using Pearson correlation
    """
    corr_matrix = ensure_gpu_tensor(features.get("correlation_matrix"))
    token_list = features.get("token_list", [])

    if corr_matrix is None or not token_list:
        logger.warning("Pearson: Missing data")
        return []

    n = len(token_list)

    if corr_matrix.shape != (n, n):
        logger.error(
            f"Pearson: Matrix size mismatch! "
            f"token_list={n}, matrix={corr_matrix.shape}"
        )
        return []

    # ✅ FIX: Convert threshold to tensor
    rho_threshold_gpu = torch.tensor(rho_threshold, device=DEVICE, dtype=torch.float32)

    # Upper triangular mask
    mask = torch.triu(torch.ones(n, n, device=DEVICE), diagonal=1).bool()

    # Extract values
    corr_values = corr_matrix[mask]

    # Filter by |ρ| > threshold
    selected_mask = torch.abs(corr_values) > rho_threshold_gpu

    # Get indices
    pair_idx = torch.nonzero(mask, as_tuple=False)
    selected_idx = pair_idx[selected_mask]

    # Convert to pairs
    if len(selected_idx) > 0:
        pairs_cpu = selected_idx.cpu().numpy()
        selected_pairs = [
            tuple(sorted([token_list[i], token_list[j]])) for i, j in pairs_cpu
        ]
    else:
        selected_pairs = []

    logger.info(f"✅ Pearson: {len(selected_pairs)} pairs selected")
    logger.info(f"   Threshold: |ρ| > {rho_threshold:.2f}")

    return selected_pairs


def select_pairs_cointegration(
    features: Dict[Tuple[str, str], Dict[str, Any]],
    pval_threshold: float,
    token_list: List[str],
) -> List[Tuple[str, str]]:
    """
    Select pairs using Engle-Granger cointegration test
    """
    if not features:
        logger.warning("Cointegration: No features available")
        return []
    allowed_tokens = set(token_list)
    selected_pairs = []

    for (t1, t2), feat in features.items():
        if t1 not in allowed_tokens or t2 not in allowed_tokens:
            continue

        pvalue = feat.get("pvalue")
        if pvalue is None:
            continue

        if pvalue < pval_threshold:
            selected_pairs.append(tuple(sorted([t1, t2])))

    logger.info(f"✅ Cointegration: {len(selected_pairs)} pairs selected")
    logger.info(f"   Threshold: p < {pval_threshold:.3f}")

    return selected_pairs
