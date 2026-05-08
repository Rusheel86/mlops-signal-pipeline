import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_metrics(path: Path, payload: dict) -> None:
    _ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch signal pipeline")
    p.add_argument("--input", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--log-file", required=True)
    return p.parse_args(argv)


def _setup_logging(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("mlops-signal-pipeline")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _load_and_validate_config(cfg_path: Path) -> dict:
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Invalid config structure: expected a YAML mapping/object")

    missing = [k for k in ("seed", "window", "version") if k not in cfg]
    if missing:
        raise ValueError(f"Invalid config structure: missing keys {missing}")

    seed = cfg["seed"]
    window = cfg["window"]
    version = cfg["version"]

    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Invalid config: seed must be a non-negative int")
    if not isinstance(window, int) or window <= 0:
        raise ValueError("Invalid config: window must be a positive int")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("Invalid config: version must be a non-empty str")

    return {"seed": seed, "window": window, "version": version.strip()}


def _load_and_validate_dataset(input_path: Path, logger: logging.Logger) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Invalid CSV format: {e}") from e

    if df.shape[0] == 0:
        raise ValueError("Empty input file: zero rows")

    if "close" not in df.columns:
        raise ValueError(f"Missing required column 'close'. Columns present: {list(df.columns)}")

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    nan_before = int(df["close"].isna().sum())
    if nan_before > 0:
        logger.warning("close coercion produced %s NaN rows; dropping", nan_before)
        df = df.dropna(subset=["close"]).copy()

    if df.shape[0] == 0:
        raise ValueError("All rows dropped after close coercion; no valid close values remain")

    return df


def main(argv: list[str] | None = None) -> int:
    t_start = time.perf_counter()
    args = _parse_args(argv)

    input_path = Path(args.input)
    config_path = Path(args.config)
    output_path = Path(args.output)
    log_file_path = Path(args.log_file)

    logger = _setup_logging(log_file_path)
    logger.info(
        "=== Job start === input=%s config=%s output=%s log_file=%s",
        input_path,
        config_path,
        output_path,
        log_file_path,
    )

    version_for_error = "v1"
    try:
        cfg = _load_and_validate_config(config_path)
        version_for_error = cfg["version"]
        logger.info(
            "Config loaded - seed=%s window=%s version=%s",
            cfg["seed"],
            cfg["window"],
            cfg["version"],
        )

        np.random.seed(cfg["seed"])
        logger.info("Seed set")

        df = _load_and_validate_dataset(input_path, logger)
        logger.info("Dataset loaded - %s rows, %s columns", int(df.shape[0]), int(df.shape[1]))

        window = int(cfg["window"])
        df["rolling_mean"] = df["close"].rolling(window=window, min_periods=1).mean()
        logger.info("Rolling mean computed")
        logger.debug("Rolling mean sample (first 8 rows):\n%s", df.head(8).to_string(index=False))

        df["signal"] = (df["close"] > df["rolling_mean"]).astype(int)
        ones = int((df["signal"] == 1).sum())
        zeros = int((df["signal"] == 0).sum())
        logger.info("Signal generated - ones=%s zeros=%s", ones, zeros)

        rows_processed = int(len(df))
        signal_rate = round(float(df["signal"].mean()), 4)
        latency_ms = int((time.perf_counter() - t_start) * 1000)

        logger.info(
            "Metrics summary - rows_processed=%s signal_rate=%.4f latency_ms=%s",
            rows_processed,
            signal_rate,
            latency_ms,
        )

        metrics = {
            "version": cfg["version"],
            "rows_processed": rows_processed,
            "metric": "signal_rate",
            "value": signal_rate,
            "latency_ms": latency_ms,
            "seed": cfg["seed"],
            "status": "success",
        }
        _write_metrics(output_path, metrics)
        logger.info("Metrics written to %s", output_path)
        logger.info("=== Job end - status=success ===")
        print(json.dumps(metrics, indent=2), file=sys.stdout)
        return 0
    except Exception as e:  # noqa: BLE001
        logger.exception("Job failed")
        err_metrics = {
            "version": version_for_error,
            "status": "error",
            "error_message": str(e),
        }
        try:
            _write_metrics(output_path, err_metrics)
            logger.info("Metrics written to %s", output_path)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to write error metrics to %s", output_path)
        logger.info("=== Job end - status=error ===")
        print(json.dumps(err_metrics, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
