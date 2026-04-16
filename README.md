# dispersion_meta

Data layer for dispersion trading meta-optimization. Persists a structured history of
(proposed config, market features, realized forward outcome, accept/reject decision) tuples
for consumption by a contextual bandit (built separately).

## Install

```bash
pip install -e ".[dev]"
```

## Test

```bash
pytest
```
