from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    for c in ["open", "high", "low", "close", "volume_btc", "volume_usd"]:
        df[c] = df[c].astype("float64")
    return df


def _print_missing_audit(df: pd.DataFrame) -> None:
    na_counts = df.isna().sum()
    total_missing = int(na_counts.sum())
    if total_missing == 0:
        print("No missing values found")
        return
    na_pct = (na_counts / len(df) * 100.0).round(4)
    print("Missing value audit (count, %):")
    for col in df.columns:
        print(f"- {col}: {int(na_counts[col])} ({float(na_pct[col])}%)")
    print(f"Total missing cells: {total_missing}")


def _print_outliers(df: pd.DataFrame) -> None:
    print("Outlier detection (IQR method):")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        q1 = float(df[col].quantile(0.25))
        q3 = float(df[col].quantile(0.75))
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        count = int(((df[col] < lo) | (df[col] > hi)).sum())
        print(f"- {col}: {count}")

    close = df["close"]
    z = (close - close.mean()) / close.std(ddof=0)
    z_count = int((z.abs() > 3).sum())
    print(f"Outlier detection (Z-score on close, |z|>3): {z_count}")


def _normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = float(sigma)
    if sigma == 0.0:
        return np.zeros_like(x)
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def main() -> int:
    print("Loading data.csv")
    df = _load_data(Path("data.csv"))
    print("Loaded dataset")

    print("Descriptive statistics")
    print(f"shape: {df.shape}")
    print("dtypes:")
    print(df.dtypes)
    print("describe:")
    print(df.describe())

    first_ts = df.index.min()
    last_ts = df.index.max()
    duration = last_ts - first_ts
    print(f"date_range: {first_ts} -> {last_ts} (duration={duration})")

    mem_bytes = int(df.memory_usage(deep=True).sum())
    print(f"memory_usage_bytes: {mem_bytes}")

    print("Missing value audit")
    _print_missing_audit(df)

    print("Outlier detection")
    _print_outliers(df)

    print("Rolling signal preview")
    df["rolling_mean_5"] = df["close"].rolling(5, min_periods=1).mean()
    df["signal"] = (df["close"] > df["rolling_mean_5"]).astype(int)
    print(df["signal"].value_counts())

    out_dir = Path("eda_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving charts to {out_dir}{os.sep}")

    print("Chart 1: close price with rolling means")
    rm5 = df["close"].rolling(5, min_periods=1).mean()
    rm20 = df["close"].rolling(20, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["close"], label="close", linewidth=1.0)
    ax.plot(df.index, rm5, label="rolling_mean_5", linestyle="--", linewidth=1.0)
    ax.plot(df.index, rm20, label="rolling_mean_20", linestyle="--", linewidth=1.0)
    ax.set_title("BTC/USD Close Price with Rolling Means")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "01_close_price.png", dpi=150)
    plt.close(fig)

    print("Chart 2: OHLC first 200 bars")
    sample = df.head(200)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sample.index, sample["open"], label="open", alpha=0.7)
    ax.plot(sample.index, sample["high"], label="high", alpha=0.7)
    ax.plot(sample.index, sample["low"], label="low", alpha=0.7)
    ax.plot(sample.index, sample["close"], label="close", alpha=0.7)
    ax.set_title("OHLC — First 200 Bars")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "02_ohlc_sample.png", dpi=150)
    plt.close(fig)

    print("Chart 3: volume over time")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.bar(df.index, df["volume_btc"], alpha=0.5, width=0.0008)
    ax1.set_title("Trading Volume Over Time")
    ax1.set_ylabel("volume_btc")
    ax2.plot(df.index, df["volume_usd"], linewidth=1.0)
    ax2.set_ylabel("volume_usd")
    fig.tight_layout()
    fig.savefig(out_dir / "03_volume.png", dpi=150)
    plt.close(fig)

    print("Chart 4: log return distribution")
    log_returns = np.log(df["close"] / df["close"].shift(1)).dropna()
    mu = float(log_returns.mean())
    sigma = float(log_returns.std(ddof=0))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(log_returns.values, bins=100, density=True, alpha=0.7, label="log_returns")
    xs = np.linspace(float(log_returns.min()), float(log_returns.max()), 400)
    ax.plot(xs, _normal_pdf(xs, mu, sigma), label="Normal fit", linewidth=1.5)
    ax.set_title("Log Return Distribution")
    ax.legend()
    ax.text(
        0.02,
        0.95,
        f"mean={mu:.6f}\nstd={sigma:.6f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    fig.tight_layout()
    fig.savefig(out_dir / "04_returns_distribution.png", dpi=150)
    plt.close(fig)

    print("Writing preprocessed_data.csv")
    out = df.copy()
    out["log_return"] = np.log(out["close"] / out["close"].shift(1))
    out["rolling_mean_5"] = out["close"].rolling(5, min_periods=1).mean()
    out["signal"] = (out["close"] > out["rolling_mean_5"]).astype(int)
    out = out.sort_index()
    out.to_csv("preprocessed_data.csv", index=True)
    print(f"preprocessed_data.csv written - {len(out)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

