"""
Sensitivity Analysis Module - FIXED VERSION
================================================
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union

from config import CONFIG

logger = logging.getLogger(__name__)


def get_parameter_values(param_spec: Dict[str, float]) -> List[Union[int, float]]:
    """
    Convert parameter specification to list of values using NumPy.
    Automatically handles integer vs float parameters.

    Args:
        param_spec: Dict with "min", "max", "step"

    Returns:
        List of test values (int or float based on input)
    """
    required_keys = ["min", "max", "step"]
    if not all(key in param_spec for key in required_keys):
        raise ValueError(
            f"Invalid param_spec {param_spec}. " f"Must have keys: {required_keys}"
        )

    # ✅ FIX: Detect integer parameters
    is_int_param = all(isinstance(param_spec[k], int) for k in ["min", "max", "step"])

    values = np.arange(
        start=param_spec["min"],
        stop=param_spec["max"] + param_spec["step"] / 2,
        step=param_spec["step"],
    )

    values = np.round(values, decimals=10)

    # ✅ Return appropriate type
    if is_int_param:
        return [int(v) for v in values]
    else:
        return values.tolist()


def get_sensitivity_ranges(method: str) -> Dict[str, List[Union[int, float]]]:
    """
    Get sensitivity test ranges from CONFIG parameter grid.
    """
    if "parameter_grids" not in CONFIG:
        raise ValueError("CONFIG missing 'parameter_grids'")

    if method not in CONFIG["parameter_grids"]:
        raise ValueError(f"No parameter grid for method '{method}'")

    param_specs = CONFIG["parameter_grids"][method]
    sensitivity_ranges = {}

    for param_name, spec in param_specs.items():
        values = get_parameter_values(spec)
        sensitivity_ranges[param_name] = values

        logger.info(
            f"  {param_name}: {len(values)} values "
            f"from {min(values)} to {max(values)} (type: {type(values[0]).__name__})"
        )

    return sensitivity_ranges


def run_sensitivity_analysis(
    method: str,
    best_params: Dict[str, Any],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    use_cache: bool,
):
    """
    Run sensitivity analysis by varying each parameter individually.

    Args:
        method: Trading method (must be 'mfdcca')
        best_params: Optimal parameters from grid search
    """
    if method != "mfdcca":
        logger.info(f"Sensitivity analysis only for MFDCCA, skipping {method}")
        return

    logger.info(f"\n{'='*80}")
    logger.info(f"SENSITIVITY ANALYSIS: {method.upper()}")
    logger.info(f"Period: {start_date.date()} → {end_date.date()}")

    logger.info(f"{'='*80}\n")

    # Setup output directory
    results_dir = Path(CONFIG["results_dir"]) / "full_period" / method / "sensitivity"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Get parameter ranges
    sensitivity_ranges = get_sensitivity_ranges(method)

    # Extract base parameters

    param_names = list(sensitivity_ranges.keys())
    base_params = {k: best_params[k] for k in param_names if k in best_params}

    logger.info(f"Base Parameters (from optimization):")
    for k, v in base_params.items():
        logger.info(f"  {k}: {v} (type: {type(v).__name__})")
    logger.info("")

    # Test each parameter
    for param_name, test_values in sensitivity_ranges.items():
        logger.info(f"\n{'─'*60}")
        logger.info(f"Testing: {param_name}")
        logger.info(f"Values: {test_values}")
        logger.info(f"{'─'*60}")

        results = []

        for value in test_values:
            # Create test parameters (vary only this parameter)
            test_params = base_params.copy()
            test_params[param_name] = value

            logger.info(
                f"  Testing {param_name}={value} (type: {type(value).__name__})"
            )

            try:
                from simulation import run_full_period_evaluation

                # Run evaluation
                result = run_full_period_evaluation(
                    method=method,
                    params=test_params,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=use_cache,
                    return_weekly_data=False,
                )

                # Handle Union return type
                if isinstance(result, tuple):
                    metrics, _ = result
                else:
                    metrics = result

                # Store result
                result_dict = {
                    param_name: value,
                    "Sharpe_Ratio": metrics.get("Sharpe_Ratio", 0.0),
                    "Sortino_Ratio": metrics.get("Sortino_Ratio", 0.0),
                    "Max_Drawdown_%": metrics.get("Max_Drawdown_%", 0.0),
                    "Profit_Factor": metrics.get("Profit_Factor", 0.0),
                    "Calmar_Ratio": metrics.get("Calmar_Ratio", 0.0),
                }
                results.append(result_dict)

                logger.info(f"    → Sharpe: {metrics.get('Sharpe_Ratio', 0):.4f}")

            except Exception as e:
                logger.error(f"    ✗ Evaluation failed: {e}")
                # Store zero metrics for failed case
                result_dict = {
                    param_name: value,
                    "Sharpe_Ratio": 0.0,
                    "Sortino_Ratio": 0.0,
                    "Max_Drawdown_%": 0.0,
                    "Profit_Factor": 0.0,
                    "Calmar_Ratio": 0.0,
                }
                results.append(result_dict)

        # Save results for this parameter
        if results:
            df = pd.DataFrame(results)
            csv_path = results_dir / f"sensitivity_{param_name}.csv"
            df.to_csv(csv_path, index=False)

            logger.info(f"\n  ✅ Saved: {csv_path.name}")
            logger.info(
                f"     Best Sharpe: {df['Sharpe_Ratio'].max():.4f} "
                f"at {param_name}={df.loc[df['Sharpe_Ratio'].idxmax(), param_name]}"
            )

    logger.info(f"\n{'='*80}")
    logger.info(f"✅ Sensitivity analysis complete")
    logger.info(f"Results saved to: {results_dir}")
    logger.info(f"{'='*80}\n")
