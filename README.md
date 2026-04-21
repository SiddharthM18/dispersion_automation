# dispersion_meta

A meta-optimization layer for dispersion trading strategies. Instead of manually picking which optimizer configuration to run each day, this package learns from historical outcomes which configurations work best under which market conditions and automatically proposes the best candidates.

## Problem

A dispersion trading desk runs a portfolio optimizer ([dispersion_optimization](https://github.com/SiddharthM18/dispersion_optimization)) daily to select option positions. The optimizer accepts a configuration dict that controls the objective function (maximize mean PnL, maximize Sharpe, minimize drawdown, etc.), weight bounds, and cardinality constraints. Different configurations perform better in different market regimes — a max-Sharpe config may excel in low-volatility, high-correlation environments while a min-drawdown config protects capital when VIX spikes.

The question this package answers: **given today's market features, which optimizer configuration family should we run?**

## Approach

The package uses a **contextual Thompson sampling bandit** with Bayesian linear regression per arm:

- **5 arms** correspond to the 5 objective families: `max_mean`, `max_min`, `max_sharpe`, `min_drawdown`, `composite`
- **Context** is a 7-dimensional daily feature vector: VIX level, VIX 20d change, average pairwise correlation, recent dispersion PnL, skew steepness, term structure slope, and earnings density
- **Reward** is `meta_score` — a forward-looking Sharpe ratio computed from realized T+5 returns, optionally penalized for portfolio turnover
- Each arm maintains a Bayesian linear model mapping features to expected meta_score, with a conjugate normal prior updated analytically (no MCMC, no gradient descent — just matrix algebra on 8x8 matrices)
- **Thompson sampling** draws from each arm's posterior to balance exploration and exploitation

The bandit is **per-product** (5 product types: corridor variance swaps, variance swaps, vol swaps, gamma swaps, delta-neutral gamma swaps), because the feature-to-reward relationship is product-dependent. This gives 25 tiny posterior models total.

### Two-tier search

The bandit handles family selection (the hard, regime-dependent decision). Within a selected family, a specific configuration is sampled uniformly from a discrete grid of ~12-36 hyperparameter combinations per family (~99 total). This separation is deliberate: family selection benefits from learning, while inner hyperparameters are well-behaved enough for grid coverage.

### Daily output

Each day, the bandit proposes **4 configurations per product** (20 total):
- **1 best** — the Thompson-sampled top family
- **2 alt** — the next two ranked families
- **1 explore** — the family with highest posterior uncertainty (not already selected), ensuring continued learning

## Data layer

All historical data is stored as Hive-partitioned Parquet files with atomic writes, schema validation, and walk-forward filtering:

| Table | Key | Contents |
|-------|-----|----------|
| `daily_features` | (date, product) | 7 market features per product per day |
| `proposals` | (date, product, config_hash) | Proposed configs, weights, solver stats, Thompson sampling diagnostics |
| `outcomes` | (date, product, config_hash) | Forward 5-day PnL, Sharpe, meta_score |
| `decisions` | (date, product, config_hash) | Accept/reject/modified audit log (append-only) |

PnL matrices are stored separately as one Parquet file per (date, product, config_hash) for full reproducibility.

## Daily workflow

```python
from dispersion_meta.propose import propose_today
from dispersion_meta.record_outcome import record_outcomes
from dispersion_meta.record_decision import record_decision
from datetime import date

# 1. Propose — bandit selects configs, optimizer solves, proposals written to disk
result = propose_today(
    today=date(2026, 4, 16),
    features={"vix_level": 18.5, "vix_20d_change": -0.02, ...},
    pnl_matrices={"corridor_var": pnl_matrix, ...},  # (T, N) numpy arrays
    column_names=tickers,
)

# 2. After T+5 — record forward realized outcomes
record_outcomes(
    propose_date=date(2026, 4, 16),
    eval_date=date(2026, 4, 23),
    forward_returns={"corridor_var": fwd_returns, ...},  # (5, N) numpy arrays
    trailing_vol={"corridor_var": 0.012, ...},
    column_names=tickers,
)

# 3. Log your decision
record_decision(
    propose_date=date(2026, 4, 16),
    config_hash="abc123...",
    product="corridor_var",
    decision="accepted",
)
```

## Cold start

On day 1 with no historical data, the bandit's prior (`mu_0 = 0`, `Lambda_0 = alpha * I`) makes all families look equally uncertain. Thompson samples from this prior are approximately random — the bandit explores uniformly until it accumulates enough observations to form preferences. No special-casing is needed; the math handles it naturally.

## Package structure

```
src/dispersion_meta/
  schemas.py          # Schema definitions, validation, product/family constants
  paths.py            # Data root, partition paths, atomic writes
  io.py               # Read/write for all four tables + PnL matrices
  hashing.py          # Deterministic config hashing (SHA-256, canonical JSON)
  training_table.py   # Walk-forward joins for bandit training
  meta_score.py       # Meta-score formula (research / continuous modes)
  synthetic.py        # Synthetic data generator with ground-truth oracle
  config_space.py     # Discrete config grid (5 families, ~99 configs)
  bandit.py           # Per-product Bayesian linear regression + Thompson sampling
  propose.py          # Daily proposal orchestration
  record_outcome.py   # T+5 forward outcome computation
  record_decision.py  # Accept/reject/modified decision logging
```

## Install

```bash
pip install -e ".[dev]"
```

The optimizer backend must be installed separately:

```bash
pip install git+https://github.com/SiddharthM18/dispersion_optimization.git
```

## Test

```bash
pytest
```

156 tests covering schema validation, IO round-trips, bandit posterior math, oracle recovery, proposal orchestration with retry logic, and optimizer config compatibility.

## Quick start

See `examples/walkthrough.ipynb` for a full end-to-end demo using synthetic data.
