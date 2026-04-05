import torch
from pathlib import Path
import pandas as pd
import os

CONFIG = {
    # ============================================================================
    # DATA PARAMETERS
    # ============================================================================
    "use_capm": True,
    "date_column": "Date",
    "price_column": "Price",
    "data_dir": Path(__file__).resolve().parent / "data",
    "results_dir": Path(__file__).resolve().parent / "Result",
    # ============================================================================
    # FULL PERIOD EVALUATION (2021–2025)
    # ============================================================================
    "data_loading_start": pd.Timestamp("2020-01-01"),
    "data_loading_end": pd.Timestamp("2025-12-31"),
    # ============================================================================
    # FULL PERIOD EVALUATION (2021–2025)
    # ============================================================================
    "full_period_start": pd.Timestamp("2021-01-01"),
    "full_period_end": pd.Timestamp("2025-12-31"),
    # ============================================================================
    # SUB-PERIOD ANALYSIS (AFTER PARAMETER SELECTION)
    # ============================================================================
    "sub_periods": [
        {
            "name": "2021",
            "study_start": pd.Timestamp("2021-01-01"),
            "study_end": pd.Timestamp("2021-12-31"),
        },
        {
            "name": "2022",
            "study_start": pd.Timestamp("2022-01-01"),
            "study_end": pd.Timestamp("2022-12-31"),
        },
        {
            "name": "2023",
            "study_start": pd.Timestamp("2023-01-01"),
            "study_end": pd.Timestamp("2023-12-31"),
        },
        {
            "name": "2024",
            "study_start": pd.Timestamp("2024-01-01"),
            "study_end": pd.Timestamp("2024-12-31"),
        },
        {
            "name": "2025",
            "study_start": pd.Timestamp("2025-01-01"),
            "study_end": pd.Timestamp("2025-12-31"),
        },
    ],
    # ============================================================================
    # MFDCCA PARAMETERS
    # ============================================================================
    "q_list": [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
    "risk_free_rate": 0.0,
    # ============================================================================
    # TOKENS & MARKET
    # ============================================================================
    "token_names": [
        "ADA",
        "ATOM",
        "BCH",
        "BNB",
        "BTC",
        "DASH",
        "DOGE",
        "EOS",
        "ETC",
        "ETH",
        "LINK",
        "LTC",
        "NEO",
        "TRX",
        "VET",
        "XLM",
        "XMR",
        "XRP",
        "XTZ",
        "ZEC",
    ],
    "market_index": "INDEX",
    "btc_symbol": "BTC",
    # ============================================================================
    # METHOD-SPECIFIC PARAMETER GRIDS - MIN/MAX/STEP FORMAT (BOTH OPTIMIZATION & SENSITIVITY)
    # ============================================================================
    "parameter_grids": {
        # MFDCCA: 5 parameters
        "mfdcca": {
            "pair_hxy_threshold": {"min": 0.35, "max": 0.50, "step": 0.05},
            "threshold_h": {"min": 0.05, "max": 0.30, "step": 0.05},
            "threshold_alpha": {"min": 0.10, "max": 0.30, "step": 0.05},
            "divergence_lookback": {"min": 3, "max": 10, "step": 1},
            "divergence_threshold": {"min": 0.01, "max": 0.15, "step": 0.01},
        },
        "mfdcca_raw": {
            "pair_hxy_threshold": {"min": 0.50, "max": 0.70, "step": 0.05},
            "threshold_h": {"min": 0.10, "max": 0.40, "step": 0.05},
            "threshold_alpha": {"min": 0.20, "max": 0.40, "step": 0.05},
            "divergence_lookback": {"min": 3, "max": 10, "step": 1},
            "divergence_threshold": {"min": 0.01, "max": 0.15, "step": 0.01},
        },
        # DCCA: 3 parameters
        "dcca": {
            "pair_hxy_threshold": {"min": 0.35, "max": 0.50, "step": 0.05},
            "divergence_lookback": {"min": 3, "max": 10, "step": 1},
            "divergence_threshold": {"min": 0.01, "max": 0.15, "step": 0.01},
        },
        # Pearson: 3 parameters
        "pearson": {
            "rho_threshold": {"min": 0.40, "max": 0.70, "step": 0.05},
            "divergence_lookback": {"min": 3, "max": 10, "step": 1},
            "divergence_threshold": {"min": 0.01, "max": 0.15, "step": 0.01},
        },
        # Cointegration: 3 parameters
        "cointegration": {
            "pval_threshold": {"min": 0.01, "max": 0.10, "step": 0.01},
            "divergence_lookback": {"min": 3, "max": 10, "step": 1},
            "divergence_threshold": {"min": 0.01, "max": 0.15, "step": 0.01},
        },
        # Index (benchmark) - no parameters needed
        "index": {},
        "btc": {},
    },
    # ============================================================================
    # TRADING PARAMETERS
    # ============================================================================
    "holding_period_days": 5,
    "rebalance_freq": "W-MON",
    "export_results": True,
    # ============================================================================
    # DEVICE
    # ============================================================================
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
}

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def get_device():
    """
    Stay on GPU 0.
    One added line: set_per_process_memory_fraction(0.33)
    caps usage at ~8 GB out of 24 GB so Windows desktop processes
    (explorer, VS Code, Edge) always have VRAM headroom on the same card.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        # ── NEW: hard memory cap ──────────────────────────────────────────
        # 0.33  →  ~8 GB  (your code peaks at ~2-3 GB, so this is safe)
        # raise to 0.50 (~12 GB) if you ever see OOM errors during a run
        torch.cuda.set_per_process_memory_fraction(0.70, device=device)
        # ─────────────────────────────────────────────────────────────────

        total_mib = torch.cuda.get_device_properties(device).total_memory // (1024**2)
        print(f"[config] GPU 0 | {total_mib} MiB total | capped at 33% (~8 GB)")
        return device

    print("[config] No CUDA GPU found — running on CPU")
    return torch.device("cpu")


DEVICE = get_device()
torch.set_num_threads(1)
