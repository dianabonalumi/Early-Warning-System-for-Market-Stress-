# Portfolio Replica Strategy: Multi-Asset Index Replication with Liquid Futures

Replication of a multi-asset target index using a basket of liquid futures, comparing linear models, a Kalman filter and a neural network under realistic portfolio constraints.

> Course project, *Fintech* — Politecnico di Milano (Business Case 3).


## Problem

Hedge fund indices are hard to invest in directly: they are partly opaque and their returns are smoothed. The goal is to replicate a target with transparent, liquid futures, and to understand how closely this is possible.

The raw benchmarks are not good replication targets on their own:

- **HFRXGL** has almost flat long-run returns and fat-tailed losses;
- **MXWO / MXWD** are nearly identical (redundant) and have drawdowns of about 58%;
- **LEGATRUU** (bonds) is too easy to replicate with a few bond futures.

For this reason a **"Monster Index"** is built as an equal-weight combination of HFRXGL, MXWO and LEGATRUU (MXWD is excluded as redundant). The weight choice is checked against four tilted alternatives.

## Data

The notebook expects `Dataset3_PortfolioReplicaStrategy.xlsx`: weekly prices for 15 series (4 indices and the futures universe).
The dataset comes from the course material and is **not included** in this repository.

## Approach

1. **Preprocessing and data quality**: missing values, duplicates, return ranges, index/futures split.
2. **Exploratory analysis**: risk/return, drawdowns, rolling correlations, VIF among futures, QQ plots, autocorrelation, Ljung-Box test, volatility clustering.
3. **Feature selection** among the futures (done inside the rolling procedure in the neural network, to avoid look-ahead).
4. **Linear models**: OLS, Ridge, Lasso, Elastic Net, with cross-validated penalty, rolling-window and rebalancing-frequency sensitivity, extended grid search and transaction costs.
5. **Kalman filter**: state-space model of the weights (Ornstein-Uhlenbeck vs random walk), EM parameter estimation, a volatility-recalibrated improved version, and a projection step that enforces the constraints at every date.
6. **Neural network**: a model that outputs portfolio weights (not return forecasts), trained with a tracking-error loss plus penalties on gross exposure and an L1 penalty on weights, with a rolling backtest and grid search.

All models use the same futures universe, the same transaction cost function (5 bps per unit of turnover) and are evaluated on the same out-of-sample weekly returns. Constraints follow a UCITS-style setup (gross exposure limit of 200% and VaR-based scaling).

## Main findings

- The futures basket reproduces the **direction, volatility level and drawdown profile** of the target (including the 2008 crisis and COVID), but week-by-week co-movement is not fully captured: a residual **tracking error of about 3.4–3.5% annualised** remains in the best configurations. The notebook attributes this to the smoothing of the hedge fund component, which liquid futures cannot recover.
- **Linear models** (Elastic Net ranked best overall) are better when the mandate is to minimise tracking error: sparse, interpretable weights and low turnover.
- The **Improved Kalman filter** ranked first on Sharpe ratio and information ratio and gives smoother weight paths, with higher maximum drawdown (9.28% vs 7.36–8.50% for linear models).
- The **neural network did not improve** on the simpler models: when the residual error is structural, extra flexibility mostly fits noise.

## How to run

The notebook was written for Google Colab and loads the data from Google Drive.

1. Put `Dataset3_PortfolioReplicaStrategy.xlsx` in a folder you can access.
2. Edit `FILE_PATH` (and the `drive.mount` cell if you do not use Colab).
3. Install the dependencies and run the notebook top to bottom. The neural network grid search is the slowest part and saves intermediate results (`completed_configs.csv`).

```bash
pip install pandas numpy scipy scikit-learn statsmodels cvxpy tensorflow \
            matplotlib seaborn psutil openpyxl
```

## Repository contents

- `Business_Case3_ordered_final.ipynb`: full analysis, models and conclusions.

## Limitations

- Results depend on the constructed Monster Index, whose weights are a modelling choice.
- Transaction costs are modelled with a single flat rate.
- Past out-of-sample performance is not a guarantee of future results.
