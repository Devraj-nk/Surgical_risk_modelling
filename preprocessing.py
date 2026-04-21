"""Basic preprocessing + EDA for the surgical risk dataset.

This script is meant for quick, repeatable dataset sanity checks:
- Loads the CSV
- Coerces numeric columns
- Sorts by episode/time
- Simple within-episode forward/back fill for signal columns
- Fallback median imputation
- Generates a few baseline plots and a summary JSON

It is intentionally lightweight and avoids heavy dependencies.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = "risk_labeled_dataset_30s.updated.csv"
DEFAULT_OUTDIR = "preprocessing_out"


FEATURE_COLS = [
    "HeartRate(1/min)",
    "MeanArterialPressure(mmHg)",
    "OxygenSaturation",
    "RespirationRate(1/min)",
    "CardiacOutput(L/min)",
    "EndTidalCarbonDioxideFraction",
    "TidalVolume(mL)",
    "SystolicArterialPressure(mmHg)",
    "DiastolicArterialPressure(mmHg)",
    "CentralVenousPressure(mmHg)",
    "CarbonDioxideSaturation",
    "TotalAlveolarVentilation(L/min)",
    "OxygenConsumptionRate(mL/min)",
    "CarbonDioxideProductionRate(mL/min)",
]

TARGET_COLS = [
    "hypoxia_next_30s",
    "apnea_next_30s",
    "hypoventilation_next_30s",
    "low_tidal_volume_next_30s",
    "hypercapnia_next_30s",
    "high_etco2_next_30s",
    "respiratory_compromise_next_30s",
]

EXTRA_INPUT_COLS = ["anesthesia", "oxygen", "blood_loss"]

GROUP_COL = "episode_id"
TIME_COL = "Time(s)"


@dataclass(frozen=True)
class Summary:
    n_rows: int
    n_cols: int
    n_episodes: int
    time_step_seconds_median: float | None
    missing_fraction: dict[str, float]
    targets_pos_rate: dict[str, float]


def _ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _count_missing(df: pd.DataFrame, cols: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in cols:
        if c in df.columns:
            out[c] = int(df[c].isna().sum())
    return out


def _median_dt_seconds(df: pd.DataFrame) -> float | None:
    if GROUP_COL not in df.columns or TIME_COL not in df.columns:
        return None
    dt = df.sort_values([GROUP_COL, TIME_COL]).groupby(GROUP_COL)[TIME_COL].diff()
    dt = pd.to_numeric(dt, errors="coerce").dropna()
    if len(dt) == 0:
        return None
    return float(dt.median())


def _simple_impute(df: pd.DataFrame, *, signal_cols: list[str]) -> pd.DataFrame:
    # Sort first so ffill/bfill is meaningful.
    if GROUP_COL in df.columns and TIME_COL in df.columns:
        df = df.sort_values([GROUP_COL, TIME_COL]).reset_index(drop=True)

    # Episode-wise ffill/bfill for continuous signals.
    existing_signal_cols = [c for c in signal_cols if c in df.columns]
    if existing_signal_cols and GROUP_COL in df.columns:
        df[existing_signal_cols] = (
            df.groupby(GROUP_COL, sort=False)[existing_signal_cols]
            .apply(lambda g: g.ffill().bfill())
            .reset_index(level=0, drop=True)
        )

    # Global median fallback.
    medians = df[existing_signal_cols].median(numeric_only=True)
    df[existing_signal_cols] = df[existing_signal_cols].fillna(medians)

    return df


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")  # ensure non-interactive
        import matplotlib.pyplot as plt

        return plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with 'pip install matplotlib'."
        ) from e


def _plot_missingness(df: pd.DataFrame, outdir: Path, *, top_n: int = 25) -> None:
    plt = _import_matplotlib()

    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing.head(top_n)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.barh(missing.index[::-1], missing.values[::-1])
    ax.set_xlabel("Missing fraction")
    ax.set_title(f"Top {len(missing)} columns by missingness")
    fig.tight_layout()
    fig.savefig(outdir / "missingness_top.png", dpi=150)
    plt.close(fig)


def _plot_target_rates(df: pd.DataFrame, outdir: Path) -> None:
    plt = _import_matplotlib()

    existing = [c for c in TARGET_COLS if c in df.columns]
    if not existing:
        return

    rates = {c: float(pd.to_numeric(df[c], errors="coerce").fillna(0).mean()) for c in existing}
    items = sorted(rates.items(), key=lambda kv: kv[1], reverse=True)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar([k for k, _ in items], [v for _, v in items])
    ax.set_ylabel("Positive rate")
    ax.set_title("Target label prevalence")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "target_prevalence.png", dpi=150)
    plt.close(fig)


def _plot_feature_hists(df: pd.DataFrame, outdir: Path) -> None:
    plt = _import_matplotlib()

    cols = [c for c in (FEATURE_COLS + EXTRA_INPUT_COLS) if c in df.columns]
    if not cols:
        return

    # Keep it readable: 4 columns grid
    n = len(cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(4 * ncols, 2.8 * nrows))

    for i, c in enumerate(cols, start=1):
        ax = fig.add_subplot(nrows, ncols, i)
        s = pd.to_numeric(df[c], errors="coerce")
        s = s.dropna()
        if len(s) == 0:
            ax.set_title(c)
            ax.text(0.5, 0.5, "all NA", ha="center", va="center")
            ax.set_axis_off()
            continue

        # Use robust bounds so outliers don't squash the plot.
        lo, hi = np.nanquantile(s.to_numpy(), [0.01, 0.99])
        s_clip = s.clip(lo, hi)
        ax.hist(s_clip, bins=40)
        ax.set_title(c, fontsize=9)

    fig.suptitle("Feature distributions (clipped to 1–99th percentile)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(outdir / "feature_histograms.png", dpi=150)
    plt.close(fig)


def _plot_corr(df: pd.DataFrame, outdir: Path) -> None:
    plt = _import_matplotlib()

    cols = [c for c in (FEATURE_COLS + EXTRA_INPUT_COLS) if c in df.columns]
    if len(cols) < 2:
        return

    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr(numeric_only=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticklabels(cols, fontsize=7)
    ax.set_title("Correlation heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outdir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def _plot_example_episode(df: pd.DataFrame, outdir: Path) -> None:
    plt = _import_matplotlib()

    if GROUP_COL not in df.columns or TIME_COL not in df.columns:
        return

    # Pick the episode with most rows.
    eid = int(df[GROUP_COL].value_counts().idxmax())
    g = df[df[GROUP_COL] == eid].sort_values(TIME_COL)

    cols = [c for c in ["HeartRate(1/min)", "MeanArterialPressure(mmHg)", "OxygenSaturation"] if c in g.columns]
    if not cols:
        return

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    t = pd.to_numeric(g[TIME_COL], errors="coerce")
    for c in cols:
        y = pd.to_numeric(g[c], errors="coerce")
        ax.plot(t, y, label=c, linewidth=1.0)

    ax.set_xlabel("Time (s)")
    ax.set_title(f"Example episode {eid}: key vitals")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "example_episode_timeseries.png", dpi=150)
    plt.close(fig)


def run(input_csv: Path, outdir: Path, *, write_cleaned: bool) -> None:
    _ensure_outdir(outdir)

    print(f"[preprocessing] input: {input_csv}")
    print(f"[preprocessing] outdir: {outdir}")

    df = pd.read_csv(input_csv)

    print(f"[preprocessing] loaded rows={len(df):,} cols={len(df.columns):,}")

    # Track missingness for key signals before preprocessing.
    key_cols = [TIME_COL, GROUP_COL] + FEATURE_COLS + EXTRA_INPUT_COLS
    missing_before = _count_missing(df, key_cols)

    # Coerce numerics for known columns (keep everything else untouched).
    _coerce_numeric(df, [TIME_COL, GROUP_COL] + FEATURE_COLS + EXTRA_INPUT_COLS + TARGET_COLS)

    # Sort + impute signals.
    df = _simple_impute(df, signal_cols=FEATURE_COLS + EXTRA_INPUT_COLS)

    missing_after = _count_missing(df, key_cols)
    mb = sum(missing_before.values())
    ma = sum(missing_after.values())
    print(f"[preprocessing] key-signal missing values: {mb:,} -> {ma:,} (after imputation)")

    # Summary
    missing_fraction = df.isna().mean().sort_values(ascending=False).to_dict()
    targets_pos_rate = {
        c: float(pd.to_numeric(df[c], errors="coerce").fillna(0).mean())
        for c in TARGET_COLS
        if c in df.columns
    }

    summary = Summary(
        n_rows=int(len(df)),
        n_cols=int(len(df.columns)),
        n_episodes=int(df[GROUP_COL].nunique()) if GROUP_COL in df.columns else 0,
        time_step_seconds_median=_median_dt_seconds(df),
        missing_fraction={k: float(v) for k, v in missing_fraction.items()},
        targets_pos_rate=targets_pos_rate,
    )

    _save_json(outdir / "summary.json", payload={"summary": summary.__dict__})
    print(f"[preprocessing] wrote: {outdir / 'summary.json'}")

    # Plots
    _plot_missingness(df, outdir)
    _plot_target_rates(df, outdir)
    _plot_feature_hists(df, outdir)
    _plot_corr(df, outdir)
    _plot_example_episode(df, outdir)

    # Report expected plot files (some may be skipped if prerequisites are missing).
    for name in [
        "missingness_top.png",
        "target_prevalence.png",
        "feature_histograms.png",
        "correlation_heatmap.png",
        "example_episode_timeseries.png",
    ]:
        p = outdir / name
        if p.exists():
            print(f"[preprocessing] wrote: {p}")

    if write_cleaned:
        cleaned_path = outdir / (input_csv.stem + ".cleaned.csv")
        df.to_csv(cleaned_path, index=False)
        print(f"[preprocessing] wrote: {cleaned_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Basic preprocessing + EDA for the surgical risk dataset")
    p.add_argument("--input", type=str, default=DEFAULT_INPUT, help=f"Input CSV (default: {DEFAULT_INPUT})")
    p.add_argument(
        "--outdir",
        type=str,
        default=DEFAULT_OUTDIR,
        help=f"Output directory for plots + summary (default: {DEFAULT_OUTDIR})",
    )
    p.add_argument(
        "--skip-cleaned",
        action="store_true",
        help="Skip writing the cleaned CSV (default writes it).",
    )

    args = p.parse_args()
    run(Path(args.input), Path(args.outdir), write_cleaned=not bool(args.skip_cleaned))


if __name__ == "__main__":
    main()
