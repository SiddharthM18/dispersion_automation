# dispersion_meta

Meta-optimization layer for dispersion trading. Combines a data layer (Hive-partitioned
Parquet storage with schema validation) and a learning layer (contextual Thompson
sampling bandit) to propose, evaluate, and track optimizer configurations across
multiple product types.

## Package structure

```
src/dispersion_meta/
  schemas.py          # Schema definitions, validation, product/family constants
  paths.py            # Data root, partition paths, atomic writes
  io.py               # Read/write for features, proposals, outcomes, decisions
  hashing.py          # Deterministic config hashing
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

The package depends on
[dispersion_optimization](https://github.com/SiddharthM18/dispersion_optimization)
for the optimizer backend:

```bash
pip install git+https://github.com/SiddharthM18/dispersion_optimization.git
```

## Test

```bash
pytest
```

## Quick start

See `examples/walkthrough.ipynb` for a full end-to-end demo using synthetic data.
