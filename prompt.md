# Dispersion Meta-Optimizer: Data Layer

## Project context

I'm an exotics structured products trader at a major bank, building a meta-optimization system that sits on top of my existing dispersion trading optimizer (separate repo: SiddharthM18/dispersion_optimization on GitHub). My day job involves derivatives, volatility, correlation, and structured products; the dispersion optimizer is a personal research project I use to systematically explore basket configurations.

The existing optimizer takes a YAML/dict config (objective family + hyperparameters + constraints) and a precomputed (T, N) PnL matrix, and returns optimal weights via cvxpy convex optimization. It supports four objective families: max_sharpe, max_mean, min_drawdown, composite_recency.

The vision for THIS new project: I never have to manually pick the optimizer's config each day. Instead, the meta-system proposes 3-4 configs daily based on current market regime, I approve/reject them, and over time a contextual bandit learns which configs work in which regimes. Eventually it becomes a passive system that generates better backtests every day with minimal manual input.

This session: build the DATA LAYER ONLY. The bandit itself is deferred to a future session. The data layer is the substrate the bandit will consume.

## Background and rationale (read this carefully — it explains the WHY behind every design choice)

### The problem this solves

I currently pick optimizer configs by hand based on market intuition. This doesn't scale and isn't reproducible. The opposite extreme — running a global hyperparameter sweep daily — overfits brutally because the optimizer's "objective" is computed on in-sample backtest PnL, which it trivially maximizes by construction. A max_sharpe config will always have a great in-sample Sharpe; that tells you nothing about whether it'll work going forward.

The right framing is two-tier:

- **Optimizer level (existing repo):** given a config, find the best weights. Convex problem, deterministic, well-defined.
- **Meta level (this project):** given today's market state, find the best config. Non-convex, learned over time from realized forward outcomes.

The data layer being built this session is the persistence substrate for the meta level: it logs every (config, market state at propose time, realized forward outcome, my decision) tuple so that a future bandit can learn the mapping.

### The meta-objective and why it's what it is

The bandit's training target — the scalar that tells it "this config was good" — is computed by a single function `compute_meta_score()` in a dedicated `meta_score.py` module. The function supports two modes, selected via `MetaScoreConfig.mode`:

**`research` mode (DEFAULT):**
```
meta_score = forward_5d_sharpe
           = forward_5d_mean_return / forward_realized_vol_21d
```
No turnover penalty. This matches the actual workflow: I cherry-pick which proposals to act on, so there's no continuous traded book against which "turnover" has economic meaning. The meta_score measures *config quality as a research signal* — which configs would have generated the most interesting forward risk-adjusted P&L.

**`continuous` mode (opt-in, for future use if workflow changes):**
```
meta_score = forward_5d_sharpe - lambda * L1_turnover(weights, weights_prev_best_proposed)
```
Penalizes turnover vs the previous day's proposed 'best' basket. Use this if you commit to acting on every 'best' proposal. Lambda starts at 0.3 — pinned by asking "how much Sharpe am I willing to give up to avoid a full basket turnover?" Lambda is tuned MANUALLY, not learned.

In both modes, `forward_5d_sharpe = forward_5d_mean_return / forward_realized_vol_21d`. The 21-day realized vol denominator (not 5-day) is deliberate: a 5-day vol estimate has ~4 degrees of freedom and is wildly noisy, so the ratio's variance gets dominated by denominator noise. Using a stabilized denominator preserves the signal in the numerator.

The 5-day forward window was chosen because that's a realistic rebalancing cadence. With daily proposals and 5-day forward windows, overlapping evaluation gives ~50 roughly-independent reward signals per year — enough for a contextual bandit, not enough for full RL.

**Why the meta_score lives in its own module, not inline:** The formula is the most important single decision in the project — it defines what the bandit optimizes. Putting it in one place means the formula can be changed by editing one file, and changing it doesn't require touching synthetic data generation, outcome recording scripts, or anything else. The function returns ALL components (sharpe, turnover, mode, lambda) alongside the final score, so the outcomes row stores inputs not just outputs — which means meta_scores can be recomputed under a new formula without re-running anything.

**Alternatives considered and rejected:**

- **Raw forward PnL:** rejected — would tolerate ugly drawdowns to chase carry. No vol normalization means scores aren't comparable across regimes.
- **Replication quality vs a benchmark:** rejected — we're not benchmark-tracking; we're trying to extract realized dispersion P&L.
- **CVaR / tail-risk-penalized:** plausible but harder to estimate from 5-day windows.
- **`accepted_only` mode (turnover vs previous *accepted* basket):** rejected — requires every proposal to have a logged decision, which silently breaks the outcome pipeline if the user forgets to log decisions for a few days. The two modes we kept (`research`, `continuous`) both fail loudly or not at all; `accepted_only` failed silently.

### What the meta-score is actually optimizing (be honest about this)

The meta_score is a proxy for forward economic value, not for forward payoff directly, and not for in-sample backtest quality at all. To be precise:

- It is **NOT** optimizing for the best dispersion *backtest*. In-sample backtest stats live in the proposals table; forward stats live in the outcomes table; they are deliberately separated so the bandit cannot accidentally train on in-sample numbers. A max_sharpe config has a great in-sample Sharpe by construction; this tells you nothing.
- It is **NOT** exactly optimizing for the best dispersion *payoff*. A config that prints +1% with low vol is preferred over one that prints +1.5% with higher vol, because of the Sharpe normalization. In `continuous` mode, a high-payoff high-turnover config is also penalized.
- It IS optimizing for "realized 5-day risk-adjusted return of the held basket" (research mode), or "realized 5-day risk-adjusted return net of trading costs" (continuous mode).

In the user's actual workflow (cherry-picking proposals as research signals), `research` mode is honest: the bandit observes counterfactual returns of baskets that may never have been held, and the score it optimizes is "would this config have generated valuable signal." This is the right reward function for an idea-generation tool. It would NOT be the right reward function if the system were autonomously executing trades; that's what `continuous` mode is for.

### Counterfactual rewards and the cherry-pick workflow (important subtlety)

In the `research` workflow, the bandit's training data consists of (config, features, forward outcome) tuples where the forward outcome is the realized P&L of a basket that was never actually traded. This is fine — counterfactual rewards are how off-policy bandits work — but it means:

- The bandit is learning "which configs are good ideas," not "which configs are good to execute." These coincide most of the time but not always.
- The accept/reject decision becomes a more important second-order training signal (for the future imitation layer), because it captures "did the user find this proposal worth acting on" — which is the real economic outcome.
- Don't add execution-cost terms to the meta_score in `research` mode; they're literally zero in counterfactual.

### Why a contextual bandit (deferred to next session, but design must accommodate it)

We considered three modeling choices and explicitly rejected the other two:

- **Pure Bayesian optimization (Optuna et al.):** answers "what's the best static config across this historical window?" Useful for offline initialization but degrades as regime drifts. Markets are non-stationary; what worked in Q2 2024 vol regime is wrong in a Q4 vol-spike regime.
- **Full reinforcement learning:** sample-starved. ~50 independent reward signals per year × 10 years = 500 episodes, far below what RL needs to learn a meaningful policy. Also overcomplicated for the problem.
- **Contextual bandit (chosen):** answers "what's the best config GIVEN today's market state?" Tractable with the data we have, handles non-stationarity via the context vector, and the math (Bayesian linear regression per arm + Thompson sampling) is textbook robust.

Two-tier search structure within the bandit: outer tier picks objective FAMILY (4 discrete arms), inner tier tunes hyperparameters within that family (continuous, smaller per-arm BO). Treat them differently because family is a regime-level choice; hyperparameters within a family are smoother.

### Non-stationarity is the central risk

Markets change. The mapping from `config → realized performance` is not stationary. This shapes multiple design decisions:

- Why we need a context vector at all (not just static historical optimization)
- Why decisions are append-only (regime preferences may shift; old decisions shouldn't be overwritten)
- Why the bandit (next session) will use exponentially-decayed priors or recency-weighted training data
- Why we log feature snapshots PER PROPOSAL — the snapshot is the canonical record of what regime the proposal was made in

If anything in the data layer assumes stationarity (e.g., "just keep the latest config and run it"), it's wrong.

### The p-hacking / multiple-testing trap

If the system generates many configs and reports the best ones, we'll find configs with great backtest stats that are pure noise. Mitigations baked into the design:

- **Walk-forward filtering:** the `as_of` parameter on `build_training_table()` enforces strict before-date filtering. The bandit must NEVER see data from after its decision point. This is load-bearing — without it the entire system is a sophisticated overfitting machine.
- **Separation of in-sample stats from forward outcomes:** the proposals row stores in-sample stats (computed by the optimizer at propose time), the outcomes row stores forward stats (computed T+5). They live in different tables to make it impossible to accidentally train on in-sample stats.
- **Append-only log:** every proposal ever made is preserved, including ones that turned out badly, so we can compute deflated Sharpe and other multiple-testing-corrected metrics later.

### Why "propose, don't replace"

The system proposes 3-4 configs per day; I review and accept/reject. This is a deliberate choice, not laziness:

- I am the safety layer. The system doesn't need to be robust against its own worst outputs because I catch them.
- Multiple proposals per day enables the bandit to do real exploration without committing capital to bad configs.
- My accept/reject decisions become a SECOND training signal — encoding judgment that pure forward-PnL doesn't capture (e.g., "I rejected this because earnings season is next week and I don't trust the backtest through it"). A future "imitation layer" will train a classifier on these decisions to pre-filter proposals before I see them. The decisions table is built to support this from day one.
- "Propose multiple" structure: 1 'best' (highest expected meta-score), 2 'alt' (second-best per family), 1 'explore' (high posterior variance pick). The `proposal_type` column encodes this.

### Operational context

This is a desk tool I will actually use. Implications:

- I want raw command-line access to inspect data. Hence Parquet files I can `head` directly via `duckdb -c "SELECT * FROM 'path.parquet' LIMIT 5"`, not opaque DB blobs.
- One-file-per-config-hash for PnL matrices means I can `ls` a day's directory and immediately see which configs ran.
- File names map 1:1 to logical objects. No UUIDs, no timestamps-in-filenames. Re-running day T overwrites the same files (idempotent).
- I'm sophisticated enough to want clear errors at the boundary, not silent coercion. Hence strict validation on writes.
- The system runs on manual trigger. No daemons, no cron required. I run scripts when I want to.
- Eventually I'll want to share datasets with collaborators or sync to a cloud bucket. Hence Parquet (portable) and relative paths (portable) rather than absolute paths or DuckDB files.

### Phased roadmap (this session is step 1)

The full project plan, for context only — DO NOT build steps 2-5 in this session:

1. **Logging substrate + walk-forward harness (THIS SESSION).** All persistence, schemas, atomic writes, joins, synthetic data. The data the bandit will consume.
2. **Per-family BO baseline (next session).** Optuna over fixed config skeleton, walk-forward eval. Establishes a baseline the contextual bandit must beat.
3. **Regime feature pipeline integration.** Wire to my real upstream feature pipeline (which I have separately).
4. **Contextual Thompson sampling bandit.** The actual learning system. Bayesian linear regression per family, Thompson sampling for arm selection.
5. **Imitation layer.** Classifier on accept/reject decisions to pre-filter proposals.

The data layer must SUPPORT all of these without anticipating them in implementation. Specifically: don't add bandit-related fields to schemas (e.g., "regret"), don't precompute things the bandit will need, don't add convenience methods that lock in a bandit architecture. Stay at the substrate level.

## What you're building

A Python package `dispersion_meta` that:

1. Persists a structured history of (proposed config, market features at propose time, realized forward outcome, my accept/reject decision) tuples to disk
2. Provides clean read APIs for the bandit (next session) to consume
3. Includes a synthetic data generator with a known ground-truth oracle so the bandit can be developed and verified before real data accumulates

## Architectural decisions (already made — implement these, don't relitigate)

**Stack:** Python 3.11+, Polars (not pandas), DuckDB (for ad-hoc analytical queries later), Parquet as source of truth (NOT a DuckDB database file). Numpy for matrix math.

Stack rationale: Parquet over DuckDB-file because append-only-per-day matches Parquet's strengths and avoids in-place mutation risk. Polars over pandas because (a) zero-copy interop with Parquet via Arrow, (b) lazy evaluation with predicate pushdown, (c) cleaner List/Struct dtype support for the weights/column_names list columns. DuckDB present as a query engine (not source of truth) for ad-hoc CLI queries the user will run later.

**Storage topology:** Parquet files on disk, organized in Hive-style partitioned directories (`year=YYYY/month=MM/`). DuckDB and Polars both auto-detect this layout for predicate pushdown. Append-only; no in-place mutation. The system is "manual trigger" — no daemons, no cron required (the user runs scripts when they want to).

**Forward evaluation window:** 5 trading days. Meta-score formulation as described in the rationale section above.

**Four logical tables:**

- `daily_features` — one row per trading day, the context vector
- `proposals` — 3-4 rows per day, written at propose time, never updated
- `outcomes` — one row per proposal, written T+5 trading days later, joined on (date, config_hash)
- `decisions` — append-only audit log of accept/reject events; latest by timestamp wins per (date, config_hash)

**PnL matrices stored separately from proposal rows:** Each (T, N) PnL matrix is its own Parquet file in a per-day subdirectory: `pnl_matrices/year=2026/month=04/2026-04-13/{config_hash}.parquet`. The proposal row stores a relative path pointer. We chose this over a list-of-structs single-file layout because: (1) loading single matrices is the most common operation and the 2D access is natural; (2) command-line introspection is easier; (3) orphan cleanup is straightforward; (4) the list-of-nested-lists Polars pattern is awkward in practice. The downsides (atomicity, file count) are mitigated with atomic writes and an orphan sweep step.

**Atomic writes everywhere:** Write to `{path}.tmp`, then `os.replace` to final path. Crash-safe. Single helper in `_io_common.py`.

**Orphan sweep on proposals write:** After successfully writing today's proposals.parquet, scan the day's pnl_matrices subdirectory and delete any matrix file whose hash isn't in today's proposals row. Also delete any leftover `.tmp` files. This makes re-running a day idempotent and self-healing.

**Config hashing:** SHA-256 truncated to 16 hex chars (64 bits = ~1e-12 collision probability at a million configs). Canonicalize before hashing: sort dict keys, drop None values (treat as absent), round floats to 10 decimals, raise on non-canonicalizable types like bytes / numpy arrays / sets. CRITICAL: check `bool` before `int` in the type dispatch because `isinstance(True, int)` is True in Python. Include a schema version in the hash payload so old hashes stay stable when config schema is extended.

**Schema versioning:** Each table has a versioned schema constant (`PROPOSALS_V1`, etc.), a current alias (`PROPOSALS = PROPOSALS_V1`), and stores `_table_schema_version` as an Int32 column in every row. Writers are STRICT (reject extra/missing columns); readers are LENIENT (tolerate extra columns for forward compat).

**Validation:** Every writer calls `validate_for_write(df, schema, version)` immediately before writing — this injects the version column, casts to declared dtypes, asserts no missing/extra columns, and raises with column-specific error messages. Callers MUST NOT set `_table_schema_version` themselves; that's owned by the schema layer.

**All paths through one module:** Every filesystem path is constructed via functions in `paths.py`. No string concatenation of paths anywhere else. Includes a `set_data_root()` for redirecting to temp dirs in tests.

**Decisions are append-only with full history:** A given (date, config_hash) may have multiple decision rows if I change my mind. Two read APIs: `read_decisions_raw` (full audit log) and `read_decisions_latest` (collapsed to one row per key by latest `decided_at_utc`). All datetimes must be timezone-aware UTC; raise on naive datetimes.

## Detailed schemas

```python
# daily_features (one row per trading day)
{
    "date": Date,
    "vix_level": Float64,
    "vix_20d_change": Float64,
    "avg_pairwise_corr": Float64,        # avg pairwise corr of index constituents
    "dispersion_pnl_20d": Float64,       # trailing 20d dispersion P/L
    "skew_steepness": Float64,           # ATM skew slope
    "term_structure_slope": Float64,     # vol term structure slope
    "earnings_density_21d": Float64,     # fraction of basket reporting in next 21d
    "_table_schema_version": Int32,
}

# proposals (3-4 rows per day; immutable once written)
{
    "date": Date,
    "config_hash": Utf8,                 # 16-char hex
    "config_json": Utf8,                 # canonical JSON, exact string that was hashed
    "config_schema_version": Int32,      # version of CONFIG schema, NOT this table
    "family": Utf8,                      # 'max_sharpe'|'max_mean'|'min_drawdown'|'composite_recency'
    "proposal_type": Utf8,               # 'best'|'alt'|'explore'
    "thompson_sample_value": Float64,    # value the bandit drew from posterior
    "posterior_mean": Float64,
    "posterior_std": Float64,
    "weights": List(Float64),            # (N,) aligned with column_names
    "column_names": List(Utf8),          # (N,) tickers
    "n_names": Int32,                    # count of nonzero weights
    "in_sample_sharpe": Float64,
    "in_sample_mean_pnl": Float64,
    "in_sample_max_dd": Float64,
    "solver_status": Utf8,               # 'optimal'|'infeasible'|etc
    "solve_time_seconds": Float64,
    "pnl_matrix_path": Utf8,             # relative path under DATA_ROOT
    "_table_schema_version": Int32,
}

# outcomes (one row per proposal, written T+5)
{
    "date": Date,                        # propose date (the join key)
    "config_hash": Utf8,
    "eval_date": Date,                   # propose_date + 5 trading days
    "forward_window_days": Int32,        # 5, stored for future-proofing
    "forward_5d_pnl": Float64,
    "forward_5d_mean_return": Float64,
    "forward_realized_vol_21d": Float64, # 21d trailing for stable denom
    "forward_sharpe": Float64,           # mean / vol; computed in BOTH modes
    "turnover_vs_prev_best": Float64,    # nullable; null in 'research' mode, populated in 'continuous'
    "meta_score": Float64,               # mode-dependent; see meta_score.py
    "meta_score_mode": Utf8,             # 'research' | 'continuous'
    "lambda_used": Float64,              # turnover penalty coef; 0.0 in research mode
    "_table_schema_version": Int32,
}

# decisions (append-only)
{
    "date": Date,                        # propose date
    "config_hash": Utf8,
    "decision": Utf8,                    # 'accepted'|'rejected'|'modified'
    "decided_at_utc": Datetime(us, UTC), # MUST be tz-aware UTC
    "notes": Utf8,                       # nullable
    "modified_config_hash": Utf8,        # nullable; required iff decision='modified'
    "_table_schema_version": Int32,
}
```

## Repo structure to create

```
dispersion_meta/
├── pyproject.toml                     # Python 3.11+, deps: polars, numpy, duckdb, pytest
├── README.md                          # brief description + usage
├── .gitignore                         # ignore data/, __pycache__/, .pytest_cache/, etc.
├── src/dispersion_meta/
│   ├── __init__.py
│   ├── schemas.py                     # Polars schemas + validate_for_write
│   ├── hashing.py                     # config hashing
│   ├── paths.py                       # all filesystem path construction
│   ├── _io_common.py                  # atomic_write_parquet, read_parquet_if_exists
│   ├── io_features.py
│   ├── io_proposals.py                # includes Proposal dataclass + orphan sweep
│   ├── io_outcomes.py
│   ├── io_decisions.py
│   ├── meta_score.py                  # MetaScoreConfig + compute_meta_score
│   ├── training_table.py              # joins for the bandit (next session)
│   └── synthetic.py                   # fake data + MetaScoreOracle
├── tests/
│   ├── test_hashing.py
│   ├── test_schemas.py
│   ├── test_io_features.py
│   ├── test_io_proposals.py           # cover orphan sweep, atomic writes
│   ├── test_io_outcomes.py
│   ├── test_io_decisions.py
│   ├── test_meta_score.py
│   ├── test_training_table.py
│   └── test_synthetic.py              # incl. oracle learnability sanity check
└── data/                              # gitignored
```

## Synthetic data + ground-truth oracle

`synthetic.py` must include a `MetaScoreOracle` class that defines the true relationship between (features, family) → expected meta_score, with these regime preferences encoded as linear coefficient vectors per family:

- `max_sharpe`: prefers moderate VIX (negative coef on vix_level), high pairwise corr (large positive), avoids earnings noise (negative on earnings_density)
- `max_mean`: strong positive coef on dispersion_pnl_20d (momentum); weak elsewhere
- `min_drawdown`: positive coef on vix_level and vix_20d_change (likes high/rising vol regimes); negative on pairwise corr
- `composite_recency`: large positive coefs on skew_steepness and term_structure_slope

Sample meta_scores by adding Gaussian noise (std ~ 0.4) to oracle expected values. The bandit (next session) must NEVER see the oracle directly — it only sees noisy outcomes. Include a test that fits per-family linear regression on synthetic data and verifies the largest-magnitude oracle coefficient is recovered with the correct sign for each family.

This learnability check matters: it's the contract that says "synthetic data is fit-for-purpose for bandit development." If the test fails, the synthetic data is too noisy or the oracle is mis-specified, and the bandit session will be wasted debugging a phantom problem.

## meta_score.py specification

This module is the single source of truth for the meta-score formula. Both synthetic data generation and (eventually) real outcome recording must call `compute_meta_score()` rather than computing the score inline.

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class MetaScoreConfig:
    mode: Literal["research", "continuous"] = "research"
    lambda_turnover: float = 0.0  # ignored in 'research' mode

    def __post_init__(self):
        if self.mode == "continuous" and self.lambda_turnover <= 0:
            raise ValueError(
                "continuous mode requires lambda_turnover > 0; "
                "use research mode if you don't want a turnover penalty"
            )
        if self.mode == "research" and self.lambda_turnover != 0.0:
            raise ValueError(
                "research mode ignores lambda_turnover; set it to 0.0 explicitly"
            )

DEFAULT_META_SCORE_CONFIG = MetaScoreConfig(mode="research", lambda_turnover=0.0)

def compute_meta_score(
    *,
    config: MetaScoreConfig,
    forward_5d_mean_return: float,
    forward_realized_vol_21d: float,
    weights: list[float],
    prev_best_weights: list[float] | None,
) -> dict:
    """
    Compute meta_score and all its components.

    Returns a dict with keys ready to merge into an outcomes row:
      forward_sharpe, turnover_vs_prev_best (nullable),
      meta_score, meta_score_mode, lambda_used.

    In 'research' mode: meta_score = forward_sharpe; turnover is null.
    In 'continuous' mode: meta_score = forward_sharpe - lambda * L1_turnover.
        If prev_best_weights is None (no prior 'best'), turnover is computed
        vs a flat-zero baseline (i.e., L1 norm of current weights). This is
        a reasonable cold-start convention; document it in the docstring.

    Returns are always all-fields-present dicts (with explicit None for
    nullable fields) so callers can predictably merge into the outcomes row.
    """
```

Tests must cover: research mode produces null turnover, continuous mode with no prior gives sane cold-start, lambda=0 in continuous mode raises, lambda!=0 in research mode raises, the components dict has every field the outcomes schema expects.



1. **`pyproject.toml`, `.gitignore`, `README.md`, `__init__.py`** — repo skeleton, `git init`, initial commit.
2. **`schemas.py`** — all four schemas + `validate_for_write` + `assert_schema_compatible`. Tests cover: happy path per table, missing columns raise, extra columns raise, caller pre-setting version raises, forward-compat extra columns tolerated on read.
3. **`hashing.py`** — canonicalize + hash + `hash_and_serialize` convenience. Tests cover: key order invariance, float drift, None equivalence, bool-vs-int distinction, schema version in hash, bytes/numpy-array raise, numpy scalars hash same as Python scalars (because they subclass).
4. **`paths.py`** + **`_io_common.py`** — all path functions, atomic_write helper. Test atomic_write leaves no .tmp files on success.
5. **`io_features.py`** — simplest table, establishes the pattern. Tests cover: round-trip, missing date returns None, idempotent re-write, predicate-pushdown lazy scan across multiple year/month partitions.
6. **`io_outcomes.py`** — like features but multi-row writes; empty list raises.
7. **`io_decisions.py`** — append-only with raw vs latest readers; validation on decision string, modified+hash coupling, naive datetime rejection.
8. **`io_proposals.py`** — `Proposal` dataclass with `__post_init__` validation (weights/columns length, matrix 2D, matrix cols match column_names, proposal_type valid), three-step write (matrices → row → orphan sweep), `read_pnl_matrix` by (date,hash) and by relative path, `delete_day` for cleanup. Tests cover: round-trip, orphan sweep removes hashes not in current proposals, .tmp leftover cleanup, all the validation cases.
9. **`meta_score.py`** — `MetaScoreConfig` + `compute_meta_score`. Single source of truth for the meta-score formula; consumed by `synthetic.py` and (later) by the real outcome-recording script. Tests as specified above.
10. **`training_table.py`** — `build_training_table` (inner joins, walk-forward `as_of` filter), `build_full_table` (left joins, includes pending), `pending_proposals`, `latest_best_weights`. Tests use synthetic data.
11. **`synthetic.py`** — features generator, `MetaScoreOracle`, `synth_proposal`, `synth_day_proposals`, `synth_outcome_rows` (MUST call `compute_meta_score` rather than computing the score inline), `populate_synthetic_history`. The oracle defines the *true expected meta_score* under `research` mode; `populate_synthetic_history` accepts an optional `meta_score_config` parameter (default research). Test: 60 days of synthetic data → training table has 240 rows, all 4 families present, walk-forward filter works, oracle coefficient recovery sanity check passes. Also test that switching to `continuous` mode produces different meta_scores while leaving forward_sharpe identical.

After each step: run pytest, commit if green. Use semantic commit messages (`feat: add proposals io with orphan sweep`).

## What's explicitly OUT of scope this session

- The bandit itself (Thompson sampling, Bayesian linear regression per family). Defer entirely. `training_table.py` is the boundary — it produces what the bandit will consume.
- Wiring to the real `dispersion_optimization` repo. Synthetic data only this session.
- CLI scripts (`propose_today.py`, `record_outcome.py`, etc.). Defer.
- DuckDB-specific code. Polars handles everything we need; DuckDB is for ad-hoc CLI queries the user will write later.
- Production logging, retries, error recovery beyond what's described.
- Anticipating bandit needs in the schemas. Don't add fields like "regret," "exploration_bonus," or "arm_id" — they belong in the bandit module, not the substrate.

## Style and conventions

- Python 3.11+ syntax: `dict | None`, `list[str]`, `from __future__ import annotations` at top of modules
- Polars (not pandas) everywhere; numpy only for matrix math
- Type hints on all public functions
- Docstrings explain the WHY, not just the what — especially for non-obvious design choices (e.g., why orphan sweep is step 3, why bool comes before int in canonicalize)
- No print statements in library code; tests can print
- Import order: stdlib, third-party, local. Use `from . import paths, schemas` style for intra-package imports.

## Final check before you start

Confirm you understand the design before writing any code. Specifically:

1. Why is `_table_schema_version` injected by the schema layer rather than set by callers?
2. Why does the proposals write happen in three steps in this order: matrices first, row second, sweep third?
3. Why are decisions append-only rather than overwriting?
4. Why is the meta-score's denominator a 21-day realized vol rather than the 5-day forward vol?
5. Why is the system "propose, don't replace" rather than fully autonomous?
6. Why does the data layer log a feature snapshot per proposal, when features are the same across all proposals on a given day? (Hint: think about the bandit's training-time vs propose-time view of the world, and what happens if features get recomputed/revised later.)
7. Why does `compute_meta_score` return a dict of all components (forward_sharpe, turnover, mode, lambda) rather than just the scalar score? (Hint: think about what happens when the user wants to recompute meta_scores under a different mode or lambda value six months from now.)

Then proceed with step 1.