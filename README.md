# DRL Trading Strategies: DJIA-30 Reproduction

Reproduction of *Practical Deep Reinforcement Learning Approach for Stock Trading*
(Ensemble DRL paper).  The project trains PPO, A2C, and DDPG agents on DJIA-30
stocks, combines them via a rolling-window ensemble, and compares performance
against DJIA buy-and-hold and minimum-variance portfolio benchmarks.

## Results Summary (Paper Targets)

| Strategy         | Cum. Return | Ann. Return | Ann. Vol | Sharpe | Max DD  |
|------------------|------------|-------------|----------|--------|---------|
| Ensemble DRL     | 82.0%       | 16.8%       | 17.0%    | 0.97   | -17.7%  |
| DJIA Buy & Hold  | 38.6%       | 7.8%        | 20.1%    | 0.47   | -37.1%  |
| Min-Variance     | 31.7%       | 6.5%        | 17.8%    | 0.45   | -34.3%  |

Out-of-sample period: **2016-01-04 to 2020-05-08**

## Project Structure

```
project/
├── src/
│   ├── config.py             # Global constants and hyperparameters
│   ├── data_loader.py        # DJIA-30 data download / caching
│   ├── feature_engineering.py# Technical indicator computation
│   ├── turbulence.py         # Turbulence index (Mahalanobis distance)
│   ├── env_trading.py        # Gymnasium-compatible trading environment
│   ├── metrics.py            # Performance metric functions
│   ├── train_agent.py        # SB3 agent training wrapper
│   ├── rolling_schedule.py   # Quarterly rolling-window generator
│   ├── benchmarks.py         # DJIA B&H and min-variance benchmarks
│   └── plots.py              # All matplotlib visualisation functions
├── exp/
│   ├── exp_1_single_agents.py # PPO, A2C, DDPG standalone evaluation
│   ├── exp_2_ensemble.py      # Rolling-window ensemble
│   ├── exp_3_turbulence.py    # Turbulence control analysis
│   └── exp_4_benchmarks.py   # Benchmark construction & comparison
├── tests/
│   ├── test_metrics.py
│   ├── test_turbulence.py
│   ├── test_env.py
│   └── test_benchmarks.py
├── data/                      # Auto-populated by data_loader (gitignored)
├── results/                   # Generated figures, CSVs, and RESULTS.md
└── README.md
```

## Setup

```bash
pip install stable-baselines3 gymnasium empyrical pyportfolioopt ta \
            numpy pandas scipy matplotlib pandas-datareader
```

Set your data API key if required:

```bash
export MASSIVE_TOKEN=your_api_key   # or ALPHA_VANTAGE_KEY, etc.
```

## Running Experiments

```bash
# 1. Train PPO, A2C, DDPG separately and evaluate out-of-sample
python exp/exp_1_single_agents.py

# 2. Rolling ensemble: select best model each quarter
python exp/exp_2_ensemble.py

# 3. Turbulence de-risking analysis
python exp/exp_3_turbulence.py

# 4. DJIA buy-and-hold and min-variance benchmarks
python exp/exp_4_benchmarks.py
```

Each script saves artefacts to `results/` and appends to `results/RESULTS.md`.

## Running Tests

```bash
pytest tests/ -v
```

All four test modules cover:

| Module               | What is tested |
|----------------------|---------------|
| `test_metrics.py`    | Cumulative/annualized return, Sharpe, max drawdown |
| `test_turbulence.py` | Non-negativity, shape, early zeros, known-case identity covariance |
| `test_env.py`        | State dimension, initial cash/holdings, no short-selling, termination |
| `test_benchmarks.py` | DJIA rebasing formula, length, positivity, correct window |

## Key Implementation Choices

| Decision | Choice |
|----------|--------|
| Max shares per trade | 100 (H_MAX) |
| Turbulence threshold | 90th percentile of training-period turbulence |
| Covariance | Expanding window, LedoitWolf shrinkage |
| Trade order | Sells first, then buys by descending action magnitude |
| Agent retraining | From scratch each quarter |
| Portfolio continuity | Preserved across quarters |
| Ensemble tie-break | PPO > A2C > DDPG |
| Min-var transaction cost | 0.1% one-way at each rebalance |
| Risk-free rate | 0 (paper unspecified) |

## Citation

```bibtex
@article{liu2020practical,
  title   = {Practical Deep Reinforcement Learning Approach for Stock Trading},
  author  = {Liu, Xiao-Yang and Xiong, Zhuoran and Zhong, Shan and Yang, Hongyang and Walid, Anwar},
  journal = {arXiv preprint arXiv:1811.07522},
  year    = {2020}
}
```
