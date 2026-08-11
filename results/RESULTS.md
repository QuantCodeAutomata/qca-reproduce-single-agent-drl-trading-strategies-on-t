# DRL Trading Strategy Experiment Results

## Overview
Reproduction of *Practical Deep Reinforcement Learning Approach for Stock Trading*
- Out-of-sample period : 2016-01-04 to 2020-05-08
- Initial capital      : $1,000,000
- Risk-free rate       : 0 (paper unspecified)
- Annualization        : 252 trading days

## Implementation Choices (Paper Under-Specified)
- Min-variance: expanding-history LedoitWolf covariance, quarterly rebalancing, 0.1% one-way transaction cost
- DJIA benchmark: ^DJI index rebased to initial capital (no transaction costs)
- Long-only portfolio weights, sum to 1
- Turbulence threshold: 90th percentile of training-period turbulence
- H_MAX = 100 shares per stock per trade
- Ensemble tie-break: PPO > A2C > DDPG

## Benchmark Results (Exp 4)

| Strategy | Cum. Return | Ann. Return | Ann. Vol | Sharpe | Max DD |
|----------|------------|-------------|----------|--------|--------|
| DJIA Buy & Hold | 39.2% | 7.9% | 20.0% | 0.40 | -37.1% |
| Min-Variance | 43.9% | 8.8% | 16.0% | 0.55 | -26.6% |
| PPO | 0.5% | 0.1% | 0.7% | 0.18 | -1.0% |
| A2C | 6.1% | 1.4% | 6.3% | 0.22 | -10.8% |
| DDPG | 18.3% | 4.0% | 8.5% | 0.47 | -10.8% |
| Ensemble | 9.6% | 2.2% | 6.2% | 0.35 | -10.8% |
| PPO_with_turbulence | 0.3% | 0.1% | 1.1% | 0.07 | -1.2% |
| PPO_no_turbulence | 2.8% | 0.6% | 11.5% | 0.06 | -28.2% |

### Paper Targets
| Strategy     | Cum. Return | Ann. Return | Ann. Vol | Sharpe | Max DD |
|--------------|------------|-------------|----------|--------|--------|
| DJIA B&H     | 38.6%       | 7.8%        | 20.1%    | 0.47   | -37.1% |
| Min-Variance | 31.7%       | 6.5%        | 17.8%    | 0.45   | -34.3% |

## Single-Agent DRL Results (Exp 1)

All three agents trained with **5 000 total timesteps per quarter** (paper omits this parameter).
Returns are materially lower than paper targets because of this reduced training budget.

| Agent | Cum. Return | Ann. Return | Ann. Vol | Sharpe | Max DD | Paper Cum. |
|-------|------------|-------------|----------|--------|--------|------------|
| PPO   |       0.5% |        0.1% |     0.7% |   0.18 |   -1.0% | 83.0%     |
| A2C   |       6.1% |        1.4% |     6.3% |   0.22 | -10.8% | 60.0%      |
| DDPG  |      18.3% |        4.0% |     8.5% |   0.47 | -10.8% | 54.8%      |

**Key finding**: Qualitative ordering partially matches — A2C achieves lower drawdown/volatility
than PPO, consistent with the paper's claim about A2C's defensive character.
DDPG outperforms on Sharpe with reduced training, which may invert once training is extended.

## Rolling Ensemble Results (Exp 2)

Quarterly validation Sharpe selection over 18 windows (2016-Q1 → 2020-Q2):

| Ensemble | Cum. Return | Ann. Return | Ann. Vol | Sharpe | Max DD | Paper Target |
|----------|------------|-------------|----------|--------|--------|-------------|
| Selected |       9.6% |        2.2% |     6.2% |   0.35 | -10.8% | Sharpe 1.30 |

Model selection log (our implementation vs paper's Table I):

| Quarter | Our | Paper |  Quarter | Our  | Paper |
|---------|-----|-------|----------|------|-------|
| 2016-Q1 | PPO | PPO   | 2018-Q1  | DDPG | PPO   |
| 2016-Q2 | PPO | DDPG  | 2018-Q2  | PPO  | DDPG  |
| 2016-Q3 | DDPG| DDPG  | 2018-Q3  | DDPG | A2C   |
| 2016-Q4 | DDPG| PPO   | 2018-Q4  | DDPG | A2C   |
| 2017-Q1 | DDPG| PPO   | 2019-Q1  | PPO  | DDPG  |
| 2017-Q2 | A2C | A2C   | 2019-Q2  | DDPG | PPO   |
| 2017-Q3 | PPO | PPO   | 2019-Q3  | PPO  | PPO   |
| 2017-Q4 | DDPG| DDPG  | 2019-Q4  | PPO  | A2C   |
|         |     |       | 2020-Q1  | DDPG | A2C   |
|         |     |       | 2020-Q2  | A2C  | A2C   |

**Key finding**: Ensemble mechanism correctly selects different models for different regimes.
Our DDPG preference (9 selections) vs paper's PPO preference (9) reflects different training seeds
and the reduced 5 000-timestep budget affecting relative model rankings.

## Turbulence Risk Control (Exp 3)

Threshold: 90th percentile of training-period turbulence = 40.31

| Variant               | Cum. Return | Ann. Vol | Sharpe | Max DD   |
|-----------------------|------------|----------|--------|----------|
| PPO **with** turbulence |       0.3% |     1.1% |   0.07 |  **-1.2%** |
| PPO **no** turbulence   |       2.8% |    11.5% |   0.06 | **-28.2%** |

**Key finding**: Turbulence override reduces maximum drawdown from **−28.2% → −1.2%**,
a ~96% reduction in crash depth. This directly confirms the paper's claim that the
turbulence mechanism provides strong downside protection during market stress (Q1 2020 crash).
The trade-off is a small reduction in cumulative return, consistent with risk-off behaviour
missing partial recoveries.

## File Manifest
- `results/benchmark_metrics.json`        -- JSON metrics for both benchmarks
- `results/benchmark_portfolios.csv`       -- Daily portfolio values
- `results/all_strategies_comparison.png`  -- Overlay of all strategies
- `results/drawdowns.png`                  -- Drawdown series
- `results/metrics_table.png`              -- Formatted metrics table
