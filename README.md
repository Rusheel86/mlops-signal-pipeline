# mlops-signal-pipeline (Task 0)

This repo implements the **Task 0 technical assessment**: a minimal MLOps-style **batch** job in Python that is:
- **Reproducible** (config + seed)
- **Observable** (structured metrics + detailed logs)
- **Deployment-ready** (Dockerized, one-command run)

## What it does
- Loads `config.yaml` (`seed`, `window`, `version`)
- Reads `data.csv` (BTC/USD 1-minute OHLCV; uses `close`)
- Computes a rolling mean of `close` (window from config)
- Generates a binary signal:
  - `signal = 1` if `close > rolling_mean`
  - else `signal = 0`
- Writes `metrics.json` and `run.log`

## Required CLI (exact)
This program is designed to be run exactly like this (no hardcoded paths):

```bash
python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
```

## Local run
Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
```

Determinism check (run twice; `value` must match):

```bash
python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
python run.py --input data.csv --config config.yaml --output metrics2.json --log-file run2.log
```

## Output: `metrics.json` (schema must match)
Success output keys:

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4991,
  "latency_ms": 54,
  "seed": 42,
  "status": "success"
}
```

Error output keys (metrics file is written even on failure; exit code is non-zero):

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Description of what went wrong"
}
```

## Logging: `run.log`
`run.py` uses Python logging and includes:
- job start/end + status
- config validation + seed set
- rows loaded
- rolling mean + signal generation
- metrics summary
- full exception trace on failures

## Config (`config.yaml`)
Required keys:
- `seed`: non-negative int
- `window`: positive int
- `version`: non-empty string

## Docker
Build and run (the evaluator runs exactly these):

```bash
docker build -t mlops-task .
docker run --rm mlops-task
```

Container behavior:
- includes `data.csv` and `config.yaml`
- writes `metrics.json` and `run.log`
- prints final `metrics.json` to stdout
- exits with code 0 on success (non-zero on failure)

## EDA (optional helper)
`eda.py` is a standalone script (does not import `run.py`) that:
- prints stats / missing audit / outlier checks
- saves 4 charts to `eda_output/`
- writes `preprocessed_data.csv`

Run:

```bash
python eda.py
```

## Note: `preflight-ml` GitHub Action (optional)
I also built **preflight-ml**, a GitHub Action for running fast “pre-flight” checks on **PyTorch** training pipelines to catch silent failures (NaNs/Infs, leakage, shape mismatch) before expensive training runs.

- Action page: `https://github.com/marketplace/actions/preflight-ml`
- In this repo it’s wired as a **manual** workflow (`.github/workflows/preflight-ml.yml`) because this Task 0 project is a batch signal job (not a training loop).
- Why it matters: it’s the same MLOps principle as this project’s strict validation + metrics—**fail fast** and produce clear diagnostics before you waste compute/time.

## Files included
- `run.py`
- `config.yaml`
- `data.csv`
- `requirements.txt`
- `Dockerfile`
- `metrics.json` (sample successful output)
- `run.log` (sample successful log)

