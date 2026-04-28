"""
First Week Visualization Module
================================
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Tuple, Optional
from matplotlib.ticker import MaxNLocator

from config import CONFIG, DEVICE
from data_processing import load_all_token_data_cached
from utils import generate_trading_weeks
from capm import apply_capm_filter, compute_capm
from mfdcca import process_token_pairs, extract_hurst_matrices
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["savefig.dpi"] = 300

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_first_week_lookback(all_data: dict):
    market_dates = all_data[CONFIG["market_index"]].index
    all_dates = pd.DatetimeIndex(market_dates)

    weekly_starts = generate_trading_weeks(
        all_dates,
        CONFIG["full_period_start"],
        CONFIG["full_period_end"],
    )

    first_week_start = weekly_starts[0]
    prev_dates = all_dates[all_dates < first_week_start]

    lookback_start = prev_dates[-250]
    lookback_end = prev_dates[-1]

    return first_week_start, lookback_start, lookback_end


def plot_first_week_capm_scatter(all_data: Optional[dict] = None) -> bool:
    logger.info("\n" + "=" * 80)
    logger.info("CAPM SCATTER PLOTS")
    logger.info("=" * 80)

    try:
        if all_data is None:
            DATA_START = pd.Timestamp("2020-01-01")
            all_data = load_all_token_data_cached(
                start_date=DATA_START,
                end_date=CONFIG["full_period_end"],
                market_index=CONFIG["market_index"],
            )

        if not all_data:
            logger.error("Failed to load data")
            return False

        first_week_start, lookback_start, lookback_end = _get_first_week_lookback(
            all_data
        )

        logger.info(f"First week starts: {first_week_start.date()}")
        logger.info(f"Lookback period: {lookback_start.date()} → {lookback_end.date()}")

        lookback_data = {}
        for token, df in all_data.items():
            mask = (df.index >= lookback_start) & (df.index <= lookback_end)
            window_data = df[mask]
            if len(window_data) >= 245:
                lookback_data[token] = window_data.copy()

        market_index = CONFIG["market_index"]

        if market_index not in lookback_data:
            logger.error("Market index not in lookback data")
            return False

        index_prices = lookback_data[market_index]["close"].to_numpy(dtype=np.float32)
        tokens = [
            t for t in CONFIG["token_names"] if t in lookback_data and t != market_index
        ]

        if len(tokens) < 20:
            logger.warning(f"Only {len(tokens)}/20 tokens available")

        tokens = tokens[:20]

        token_prices_list = [
            lookback_data[token]["close"].to_numpy(dtype=np.float32) for token in tokens
        ]

        token_prices_stack = torch.tensor(
            np.vstack(token_prices_list), device=DEVICE, dtype=torch.float32
        )
        index_prices_gpu = torch.tensor(
            index_prices, device=DEVICE, dtype=torch.float32
        )

        betas, alphas, residuals_stack = compute_capm(
            token_prices_stack, index_prices_gpu
        )

        token_returns = (token_prices_stack[:, 1:] / token_prices_stack[:, :-1]) - 1
        index_returns = (index_prices_gpu[1:] / index_prices_gpu[:-1]) - 1

        token_returns_np = token_returns.cpu().numpy()
        index_returns_np = index_returns.cpu().numpy()
        betas_np = betas.cpu().numpy()
        alphas_np = alphas.cpu().numpy()

        fig, axes = plt.subplots(4, 5, figsize=(20, 14))

        for idx, token in enumerate(tokens):
            row = idx // 5
            col = idx % 5
            ax = axes[row, col]

            token_ret = token_returns_np[idx]
            index_ret = index_returns_np

            ax.scatter(index_ret, token_ret, alpha=0.3, s=15, c="gray")

            beta = betas_np[idx]
            alpha = alphas_np[idx]

            x_range = np.array([index_ret.min(), index_ret.max()])
            y_fitted = alpha + beta * x_range

            ax.plot(x_range, y_fitted, "r-", linewidth=2, label="CAPM fit")

            # Clean title
            ax.set_title(token, fontsize=14)

            # Alpha / Beta annotation box (same style as version 2)
            textstr = rf"$\beta$: {beta:.4f}" "\n" rf"$\alpha$: {alpha:.6f}"
            ax.text(
                0.05,
                0.95,
                textstr,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.85,
                ),
            )

            if col == 0:
                ax.set_ylabel("Asset Return", fontsize=14)
            if row == 3:
                ax.set_xlabel("Market Return", fontsize=14)

            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=12)

        plt.tight_layout()

        # ✅ FIXED: Same pattern as statistics
        figures_dir = ensure_dir(Path(CONFIG["results_dir"]) / "figures")

        output_path = figures_dir / "first_week_capm_scatter_all_tokens.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"✅ Saved: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to generate CAPM scatter plots: {e}", exc_info=True)
        return False


###############################################################
# RAW VS RESIDUAL
###############################################################
def plot_first_week_raw_vs_residual(all_data=None):

    if all_data is None:
        all_data = load_all_token_data_cached(
            start_date=pd.Timestamp("2020-01-01"),
            end_date=CONFIG["full_period_end"],
            market_index=CONFIG["market_index"],
        )

    first_week_start, lookback_start, lookback_end = _get_first_week_lookback(all_data)

    lookback_data = {}

    for token, df in all_data.items():
        mask = (df.index >= lookback_start) & (df.index <= lookback_end)

        if len(df[mask]) >= 245:
            lookback_data[token] = df[mask]

    capm_results = apply_capm_filter(
        tokens=list(lookback_data.keys()),
        market_index=CONFIG["market_index"],
        price_data=lookback_data,
    )

    tokens = [
        t
        for t in CONFIG["token_names"]
        if t in capm_results and t != CONFIG["market_index"]
    ][:20]

    fig, axes = plt.subplots(10, 4, figsize=(16, 18), sharex="col")

    # Column Titles
    axes[0, 0].set_title("Raw Returns", fontsize=14)
    axes[0, 1].set_title("CAPM Residuals", fontsize=14)
    axes[0, 2].set_title("Raw Returns", fontsize=14)
    axes[0, 3].set_title("CAPM Residuals", fontsize=14)

    for i in range(0, len(tokens) - 1, 2):

        row = i // 2

        token_a = tokens[i]
        token_b = tokens[i + 1]

        raw_a = lookback_data[token_a]["close"].pct_change().dropna()
        raw_b = lookback_data[token_b]["close"].pct_change().dropna()

        res_a = capm_results[token_a]["residuals"]
        res_b = capm_results[token_b]["residuals"]

        # Plot Token A
        axes[row, 0].plot(
            raw_a.index,
            raw_a.values,
            linewidth=0.8,
            color="steelblue",
        )

        axes[row, 1].plot(
            res_a.index,
            res_a.values,
            linewidth=0.8,
            color="darkorange",
        )

        # Plot Token B
        axes[row, 2].plot(
            raw_b.index,
            raw_b.values,
            linewidth=0.8,
            color="steelblue",
        )

        axes[row, 3].plot(
            res_b.index,
            res_b.values,
            linewidth=0.8,
            color="darkorange",
        )

        # Symmetric scaling Token A
        pair_max_a = max(
            abs(raw_a.min()),
            abs(raw_a.max()),
            abs(res_a.min()),
            abs(res_a.max()),
        )

        axes[row, 0].set_ylim(-pair_max_a, pair_max_a)
        axes[row, 1].set_ylim(-pair_max_a, pair_max_a)
        for ax in [axes[row, 0], axes[row, 1]]:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        # Symmetric scaling Token B
        pair_max_b = max(
            abs(raw_b.min()),
            abs(raw_b.max()),
            abs(res_b.min()),
            abs(res_b.max()),
        )

        axes[row, 2].set_ylim(-pair_max_b, pair_max_b)
        axes[row, 3].set_ylim(-pair_max_b, pair_max_b)

        for ax in [axes[row, 2], axes[row, 3]]:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        # Zero reference lines
        for ax in [axes[row, 0], axes[row, 1], axes[row, 2], axes[row, 3]]:
            ax.axhline(
                0,
                color="red",
                linestyle="--",
                linewidth=0.8,
                alpha=0.6,
            )

            ax.grid(True, alpha=0.3)

        # Token labels
        axes[row, 0].set_ylabel(
            token_a,
            fontsize=14,
        )

        axes[row, 2].set_ylabel(
            token_b,
            fontsize=14,
        )

    # Bottom X labels
    for ax in axes[-1, :]:
        ax.set_xlabel("Date", fontsize=14)

    # Tick formatting
    for ax in axes.flat:
        ax.tick_params(
            axis="x",
            rotation=45,
            labelsize=12,
        )

        ax.tick_params(
            axis="y",
            labelsize=12,
        )

    plt.tight_layout()

    figures_dir = ensure_dir(Path(CONFIG["results_dir"]) / "figures")

    plt.savefig(
        figures_dir / "first_week_raw_vs_residual_all_tokens.png",
        dpi=300,
        bbox_inches="tight",
        pil_kwargs={"dpi": (300, 300)},
        format="png",
    )

    plt.close()


def compute_and_save_delta_matrices(
    all_data: Optional[dict] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    logger.info("\n" + "=" * 80)
    logger.info("COMPUTING DELTA H AND DELTA ALPHA MATRICES")
    logger.info("=" * 80)

    try:
        # ============================
        # 1) Load data
        # ============================
        if all_data is None:
            DATA_START = pd.Timestamp("2020-01-01")
            all_data = load_all_token_data_cached(
                start_date=DATA_START,
                end_date=CONFIG["full_period_end"],
                market_index=CONFIG["market_index"],
            )

        if not all_data:
            logger.error("Failed to load data")
            return None, None

        first_week_start, lookback_start, lookback_end = _get_first_week_lookback(
            all_data
        )

        logger.info(f"First week starts: {first_week_start.date()}")
        logger.info(f"Lookback period: {lookback_start.date()} → {lookback_end.date()}")

        # ============================
        # 2) Build lookback window data
        # ============================
        lookback_data = {}
        for token, df in all_data.items():
            mask = (df.index >= lookback_start) & (df.index <= lookback_end)
            window_data = df.loc[mask]
            if len(window_data) >= 245:
                lookback_data[token] = window_data.copy()

        # ============================
        # 3) CAPM filtering
        # ============================
        capm_results = apply_capm_filter(
            tokens=list(lookback_data.keys()),
            market_index=CONFIG["market_index"],
            price_data=lookback_data,
        )

        if not capm_results:
            logger.error("CAPM filter returned no results")
            return None, None

        residuals = {
            token: capm_results[token]["residuals"]
            for token in capm_results
            if "residuals" in capm_results[token]
        }

        for token in residuals:
            residuals[token] = residuals[token].dropna().sort_index()

        # ============================
        # 4) Select valid tokens
        # ============================
        valid_tokens = [
            token
            for token in CONFIG["token_names"]
            if token in residuals and token != CONFIG["market_index"]
        ]

        if len(valid_tokens) < 2:
            logger.error("Need at least 2 tokens with residuals")
            return None, None

        logger.info(f"Valid tokens for MF-DCCA: {len(valid_tokens)}")

        # ============================
        # 5) MF-DCCA computation
        # ============================
        q_list = CONFIG["q_list"]
        logger.info("Running MF-DCCA analysis...")

        results = process_token_pairs(
            token_list=valid_tokens,
            residuals=residuals,
            q_list=q_list,
        )

        if not results:
            logger.error("No valid MF-DCCA results")
            return None, None

        logger.info(f"✅ MF-DCCA computed for {len(results)} pairs")

        # ============================
        # 6) Extract matrices
        # ============================
        hxy_matrix, delta_H_matrix, delta_alpha_matrix = extract_hurst_matrices(
            token_list=valid_tokens,
            results=results,
            q_list=q_list,
        )

        delta_H_np = delta_H_matrix.cpu().numpy()
        delta_alpha_np = delta_alpha_matrix.cpu().numpy()

        delta_H_df = pd.DataFrame(delta_H_np, index=valid_tokens, columns=valid_tokens)
        delta_alpha_df = pd.DataFrame(
            delta_alpha_np, index=valid_tokens, columns=valid_tokens
        )

        # ============================
        # 7) Lower triangular matrix only
        # ============================
        mask = np.tril(np.ones(delta_H_df.shape, dtype=bool), k=-1)
        delta_H_df_lower = delta_H_df.where(mask)
        delta_alpha_df_lower = delta_alpha_df.where(mask)

        # ============================
        # 8) Round to 3 decimals
        # ============================
        delta_H_df_lower = delta_H_df_lower.round(3)
        delta_alpha_df_lower = delta_alpha_df_lower.round(3)

        # ============================
        # 9) GLOBAL COLOR SCALE
        # ============================
        global_min = min(delta_H_df_lower.min().min(), delta_alpha_df_lower.min().min())

        global_max = max(delta_H_df_lower.max().max(), delta_alpha_df_lower.max().max())

        # ============================
        # 10) Save as png heatmaps
        # ============================
        import matplotlib.pyplot as plt
        import seaborn as sns

        figures_dir = ensure_dir(Path(CONFIG["results_dir"]) / "figures")

        # Save Delta H as png
        plt.figure(figsize=(14, 12), dpi=300)

        sns.heatmap(
            delta_H_df_lower,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=global_min,
            vmax=global_max,
            cbar_kws={"label": "ΔH"},
            linewidths=1,
            linecolor="gray",
            square=True,
            annot_kws={"size": 12},
            mask=delta_H_df_lower.isna(),
        )

        plt.xlabel("Assets", fontsize=14)
        plt.ylabel("Assets", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        # delta_H heatmap
        plt.savefig(
            figures_dir / "delta_H.png",
            dpi=300,
            bbox_inches="tight",
            pil_kwargs={"dpi": (300, 300)},
            format="png",
        )
        plt.close()
        logger.info("Saved ΔH matrix (PDF)")

        # Save Delta Alpha as png
        plt.figure(figsize=(14, 12), dpi=300)

        sns.heatmap(
            delta_alpha_df_lower,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=global_min,
            vmax=global_max,
            cbar_kws={"label": "Δα"},
            linewidths=1,
            linecolor="gray",
            square=True,
            annot_kws={"size": 12},
            mask=delta_alpha_df_lower.isna(),
        )
        plt.xlabel("Assets", fontsize=14)
        plt.ylabel("Assets", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        plt.savefig(
            figures_dir / "delta_alpha.png",
            dpi=300,
            bbox_inches="tight",
            pil_kwargs={"dpi": (300, 300)},
            format="png",
        )
        plt.close()
        logger.info("Saved Δα matrix (PDF)")

        return delta_H_df_lower, delta_alpha_df_lower

    except Exception as e:
        logger.error(f"Failed to compute delta matrices: {e}", exc_info=True)
        return None, None
