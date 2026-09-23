# Multifractal Cross-Correlation Framework for Cryptocurrency Pair Trading

A research implementation of a **multifractal cross-correlation (MFDCCA) framework for cryptocurrency pair trading**, combined with **single-index market-model (CAPM) filtering** to isolate asset-specific return dynamics before identifying statistically related cryptocurrency pairs.

This repository contains the implementation, data-processing pipeline, MFDCCA analysis, pair-selection procedure, trading simulation, statistical analysis, sensitivity analysis, and experimental results associated with the research project:

> **A Multifractal Cross-Correlation Framework for Cryptocurrency Pair Trading with CAPM Filtering**

---

## Overview

Cryptocurrency returns are influenced by both **market-wide movements** and **asset-specific dynamics**. Directly measuring the relationship between raw cryptocurrency returns can therefore capture common market movements rather than the underlying relationship between individual assets.

This project addresses this issue using a two-stage framework:

1. **Market filtering**
   A single-index market model is used to remove the common market component from individual cryptocurrency returns.

2. **Multifractal dependence analysis**
   MFDCCA is then applied to the filtered residual returns to measure nonlinear and scale-dependent cross-correlations between cryptocurrency pairs.

The resulting dependence measures are used to identify candidate pairs for a **market-neutral statistical arbitrage strategy**.

### Research Pipeline

```text
Cryptocurrency Prices
        │
        ▼
   Return Series
        │
        ▼
Single-Index Market Model
        │
        ▼
Market-Filtered Residual Returns
        │
        ▼
       MFDCCA
        │
        ├── Generalized Hurst Exponent
        ├── Multifractal Spectrum
        ├── Cross-Correlation Measures
        └── Dependence Differences
        │
        ▼
     Pair Selection
        │
        ▼
 Divergence / Convergence Signals
        │
        ▼
  Long–Short Trading
        │
        ▼
 Performance Evaluation
        │
        ├── Return
        ├── Sharpe Ratio
        ├── Sortino Ratio
        ├── Maximum Drawdown
        └── Profit Factor
```

---

## Key Features

* Cryptocurrency return preprocessing
* Single-index market-model filtering
* Market-adjusted residual return construction
* Multifractal detrended cross-correlation analysis (MFDCCA)
* Scale-dependent dependence analysis
* Generalized Hurst exponent estimation
* Multifractal spectrum analysis
* Cryptocurrency pair selection
* Divergence-based pair filtering
* Long–short statistical arbitrage simulation
* Transaction-cost modelling
* Performance evaluation
* Sensitivity analysis
* Statistical analysis and visualization
* Reproducible research workflow

---

## Methodology

### 1. Market Filtering

For each cryptocurrency, the return series is first modeled against a cryptocurrency market index:

$$
r_{i,t} = \alpha_i + \beta_i r_{m,t} + \epsilon_{i,t}
$$

where:

* \(r_{i,t}\) = return of cryptocurrency \(i\)
* \(r_{m,t}\) = market-index return
* \(\alpha_i\) = asset-specific intercept
* \(\beta_i\) = sensitivity to the market
* \(\epsilon_{i,t}\) = market-filtered residual return

The residual component is then used for the subsequent dependence analysis.

This helps distinguish **common market-driven movements** from **asset-specific dynamics**.

---

### 2. Multifractal Detrended Cross-Correlation Analysis

MFDCCA is applied to pairs of market-filtered cryptocurrency returns.

Unlike conventional correlation measures, MFDCCA allows the analysis of:

* nonlinear dependence,
* dependence across different time scales,
* persistence and anti-persistence,
* heterogeneous fluctuation behaviour,
* multifractal cross-correlation structure.

The framework evaluates the dependence structure across different fluctuation orders \(q\) and time scales \(s\).

---

### 3. Pair Selection

Candidate cryptocurrency pairs are selected according to their estimated multifractal cross-correlation characteristics.

The framework considers measures such as:

* generalized Hurst exponents,
* multifractal characteristics,
* cross-correlation differences,
* scale-dependent dependence,
* divergence between the dependence structures of the two assets.

These characteristics are used to identify pairs that exhibit sufficiently related but dynamically different behaviour suitable for relative-value trading.

---

### 4. Statistical Arbitrage Strategy

The selected pairs are used in a long–short trading framework.

When the relative relationship between the two assets deviates from the expected relationship:

* the relatively stronger asset can be **shorted**,
* the relatively weaker asset can be **bought/longed**.

The positions are subsequently closed or adjusted when the relationship converges toward the expected level.

The strategy therefore seeks to capture **relative mispricing** rather than simply predicting the overall direction of the cryptocurrency market.

---

## Data

The empirical implementation uses cryptocurrency market data together with a cryptocurrency market index.

The cryptocurrency universe includes:

```text
ADA   ATOM   BCH   BNB   BTC
DASH  DOGE   EOS   ETC   ETH
LINK  LTC    NEO   TRX   VET
XLM   XMR    XRP   XTZ   ZEC
```

The main sample covers:

```text
2021-01-01 to 2025-12-31
```

with additional historical observations used where required for rolling-window calculations.

A rolling-window framework is used to estimate the dependence structure through time.

### Market Proxy

The cryptocurrency market is represented by the:

**S&P Cryptocurrency LargeCap Index**

rather than using Bitcoin as the market proxy.

---

## Repository Structure

```text
pair_trading_using_mfdcca/
│
├── data/
│   └── Market and cryptocurrency data
│
├── Result/
│   └── Experimental results and outputs
│
├── capm.py
│   └── Market-model estimation and residual filtering
│
├── config.py
│   └── Experiment and strategy configuration
│
├── data_processing.py
│   └── Data loading, cleaning, alignment and return processing
│
├── feature_extraction.py
│   └── Feature construction for the trading framework
│
├── first_week_visualizations.py
│   └── Initial exploratory visualizations
│
├── main.py
│   └── Main execution pipeline
│
├── mfdcca.py
│   └── MFDCCA implementation
│
├── pair_selection.py
│   └── Cryptocurrency pair-selection procedure
│
├── precompute.py
│   └── Pre-computation of intermediate quantities
│
├── sensitivity_analysis.py
│   └── Parameter sensitivity and robustness analysis
│
├── simulation.py
│   └── Trading-strategy simulation
│
├── stats_analysis.py
│   └── Statistical and performance analysis
│
├── trading.py
│   └── Trading signal and portfolio logic
│
├── utils.py
│   └── Utility functions
│
└── README.md
```

---

## Main Components

| File                      | Description                                                            |
| ------------------------- | ---------------------------------------------------------------------- |
| `capm.py`                 | Estimates the single-index market model and generates residual returns |
| `mfdcca.py`               | Implements the multifractal detrended cross-correlation analysis       |
| `pair_selection.py`       | Selects candidate cryptocurrency pairs                                 |
| `trading.py`              | Generates trading positions and manages long–short logic               |
| `simulation.py`           | Simulates the trading strategy                                         |
| `data_processing.py`      | Processes prices and returns                                           |
| `feature_extraction.py`   | Extracts features used by the framework                                |
| `stats_analysis.py`       | Calculates statistical and portfolio-performance measures              |
| `sensitivity_analysis.py` | Tests robustness to parameter changes                                  |
| `precompute.py`           | Pre-computes intermediate MFDCCA quantities                            |
| `config.py`               | Stores model and experiment parameters                                 |
| `main.py`                 | Runs the complete research pipeline                                    |

---

## Strategy Parameters

The main experimental configuration includes parameters for:

* rolling-window length,
* MFDCCA fluctuation orders,
* dependence thresholds,
* pair-selection criteria,
* trading thresholds,
* number of selected pairs,
* transaction costs,
* risk-free rate.

Representative baseline parameters include:

```text
Rolling window:       250 observations
Divergence threshold: 0.09
Hurst threshold:      0.40
ΔH threshold:         0.25
Δα threshold:         0.30
Number of pairs:      3
Transaction cost:     0.001 per asset per side
Risk-free rate:       3%
```

These parameters can be modified through the configuration and analysis scripts.

---

## Performance Evaluation

The strategy is evaluated using several portfolio-level metrics:

### Return

Measures the cumulative profitability of the strategy.

### Sharpe Ratio

Measures risk-adjusted performance relative to return volatility.

### Sortino Ratio

Similar to the Sharpe ratio but focuses on downside volatility.

### Maximum Drawdown

Measures the largest peak-to-trough decline in portfolio value.

### Profit Factor

Measures the ratio between gross profits and gross losses.

Multiple analyses are performed to examine the robustness of the strategy under different parameter configurations.

---

## Reproducibility

The repository is structured to support reproducible research.

The main workflow can be executed through:

```bash
python main.py
```

Depending on the experiment, individual components can also be executed separately.

For example:

```bash
python capm.py
python mfdcca.py
python pair_selection.py
python simulation.py
python stats_analysis.py
python sensitivity_analysis.py
```

---

## Research Contribution

The main methodological idea is to combine:

```text
Market Filtering
       +
Multifractal Dependence
       +
Pair Selection
       +
Market-Neutral Trading
```

Instead of applying dependence analysis directly to raw cryptocurrency returns, the framework first removes the common market component and then examines the remaining asset-specific dynamics.

This provides a framework for studying whether **nonlinear and multiscale dependence characteristics** can improve cryptocurrency pair selection for statistical arbitrage.

---

## Results

The repository contains the experimental outputs used to evaluate the proposed framework.

The reported analysis includes:

* strategy performance,
* benchmark comparison,
* risk-adjusted performance,
* drawdown analysis,
* sensitivity analysis,
* statistical analysis,
* visualizations of multifractal dependence characteristics.

The detailed results are available in the [`Result`](./Result) directory.

---

## Related Publication

This repository accompanies the following research article:

**Thapa, A., & Cho, P. (2026).**
*A Multifractal Cross-Correlation Framework for Cryptocurrency Pair Trading with CAPM Filtering.*
**Fractal and Fractional, 10(8), 530.**

[![Paper](https://img.shields.io/badge/Paper-MDPI-blue)](https://www.mdpi.com/2504-3110/10/8/530)

---

## Citation

If you use this repository or the methodology in academic research, please cite:

```bibtex
@article{thapa2026multifractal,
  title={A Multifractal Cross-Correlation Framework for Cryptocurrency Pair Trading with CAPM Filtering},
  author={Thapa, Anil and Cho, Poongjin},
  journal={Fractal and Fractional},
  volume={10},
  number={8},
  pages=
```
