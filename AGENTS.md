# AGENTS.md — mlops-signal-pipeline

## Purpose & Context
This repository is a submission for an MLOps Engineering Internship Task 0 assessment. It implements a production-style **batch** signal pipeline over BTC/USD 1-minute OHLCV data (10,000 rows) with:
- **Reproducibility** via config + seed
- **Observability** via structured metrics (`metrics.json`) and detailed logs (`run.log`)
- **Deployment readiness** via Docker (`docker build` + `docker run`)

The goal is a deterministic, validation-heavy batch job that can run on any clean machine.

## Project Structure (must remain stable)
```
mlops-signal-pipeline/
├── run.py              # CLI batch job (fixed CLI signature)
├── eda.py              # Standalone EDA (no imports from run.py)
├── config.yaml         # seed/window/version
├── data.csv            # Deterministic synthetic BTC/USD OHLCV, 10,000 rows
├── requirements.txt    # Pinned runtime deps
├── Dockerfile          # Builds a runnable image that prints metrics.json
├── README.md           # Usage & explanation
├── AGENTS.md           # This file (invariants, contributor guide)
├── metrics.json        # Sample successful output (committed)
└── run.log             # Sample successful logs (committed)
```

## Key Invariants (do not break)
- **CLI signature is fixed**:
  - `python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log`
  - All four flags are required; no hardcoded paths.
- **Metrics schema must not change**:
  - Success metrics keys/types must match exactly:
    - `version` (str), `rows_processed` (int), `metric` (str), `value` (float rounded to 4),
      `latency_ms` (int), `seed` (int), `status` (str = "success")
  - Error metrics keys/types must match exactly:
    - `version` (str), `status` (str = "error"), `error_message` (str)
  - Metrics file must be written in both success and error cases.
- **Determinism**:
  - `numpy.random.seed(seed)` is called immediately after config validation.
  - Running the same command twice with the same `config.yaml` must produce the same `value`.
- **Docker must work**:
  - `docker build -t mlops-task .` and `docker run --rm mlops-task` must succeed.
  - Container prints final metrics JSON to stdout and exits with code 0 on success.

## Config Schema (validation rules)
`config.yaml` must contain:
- `seed`: non-negative `int`
- `window`: positive `int`
- `version`: non-empty `str`

Validation failures are treated as job errors:
- Missing file → `FileNotFoundError`
- Invalid YAML structure → `ValueError`
- Missing keys or invalid types/ranges → `ValueError`

## Signal Logic (must remain consistent)
Rolling mean and signal definition:

```python
df["rolling_mean"] = df["close"].rolling(window=window, min_periods=1).mean()
df["signal"] = (df["close"] > df["rolling_mean"]).astype(int)
```

Notes:
- `min_periods=1` ensures the first `window-1` rows use partial windows; no NaNs are produced.

## `run.py` Responsibilities
- Parse args (argparse) and configure logging **before any other work**
- Load/validate config and set seed
- Load/validate dataset with clean handling of:
  - Missing input file
  - Invalid CSV format
  - Empty file (0 rows)
  - Missing `close` column
  - Close coercion: `pd.to_numeric(errors="coerce")`, warn+drop NaNs
- Compute rolling mean and generate signal
- Compute metrics (rows_processed, signal_rate, latency_ms)
- Write metrics JSON on both success and error; create parent dirs as needed
- Print final JSON to stdout on success (stderr on error)
- Exit code 0 on success, 1 on error

## Logging Requirements (order matters)
Log entries required (in order):
1. `=== Job start ===` with input/config/output paths
2. Config loaded + validated
3. Seed set
4. Rows loaded
5. Rolling mean computed
6. Signal generated (1s vs 0s)
7. Metrics summary (rows, signal_rate, latency_ms)
8. Metrics written to path
9. `=== Job end — status=success/error ===`

## `eda.py` Responsibilities
Standalone (does not import `run.py`). Must:
- Load `data.csv` with timestamp parsing, set index, sort ascending
- Print descriptive stats, date range, memory usage
- Print missing value audit
- Print outlier detection (IQR for all numeric, z-score |z|>3 for close)
- Compute rolling mean preview (window=5) and signal; print `value_counts`
- Save four charts to `./eda_output/` using matplotlib `Agg` backend (no `plt.show()`)
- Write `preprocessed_data.csv` with:
  - float64 OHLCV, `log_return`, `rolling_mean_5`, `signal`

## Dependencies
| Package    | Purpose |
|------------|---------|
| numpy      | Determinism + math |
| pandas     | CSV I/O + rolling features |
| pyyaml     | Config parsing |
| matplotlib | EDA charts (Agg backend) |

## Docker Notes
- Base image: `python:3.9-slim`
- Image installs `requirements.txt`, copies `data.csv`, `config.yaml`, `run.py`
- Entrypoint runs `run.py` and `cat metrics.json` to print final JSON to stdout

## Modifying Signal Logic
If you change the signal logic:
- Keep the CLI, config schema, logging format/order, and metrics schema unchanged.
- Add new config keys only if you also update validation and docs.
- Ensure determinism (same inputs/config → same outputs).

## Adding New Metrics
To add metrics without breaking the rubric:
- Keep the existing success keys **exactly** as-is.
- Add optional additional keys only if downstream consumers won’t fail.
- Ensure error schema remains stable.
- Update README and AGENTS invariants if you add keys.

## Pre-commit Testing Checklist
Before pushing:
- Run `python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log` (exit 0)
- Run the same command again; verify `value` is identical (determinism)
- Run `python eda.py`; verify 4 PNGs exist in `eda_output/` and `preprocessed_data.csv` is written
- Run error-case: `python run.py --input nonexistent.csv --config config.yaml --output err_metrics.json --log-file err.log` (exit 1) and error metrics JSON written
- Run Docker: `docker build -t mlops-task . && docker run --rm mlops-task` (exit 0)

