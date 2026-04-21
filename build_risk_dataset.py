import pandas as pd
import numpy as np

SOURCE_CSV = "final_numeric_physiological_dataset.csv"
OUTPUT_CSV = "risk_labeled_dataset_30s.csv"

HORIZON_S = 30.0

# Evidence-backed thresholds based on this dataset's observed distributions
THRESHOLDS = {
    "hypoxia_spo2_lt": 0.90,
    "severe_hypoxia_spo2_lt": 0.85,
    "apnea_rr_lt": 0.1,
    "hypovent_alvvent_lt": 0.1,
    "low_tidal_volume_ml_lt": 200.0,
    "hypercapnia_co2sat_gt": 0.10,
    "severe_hypercapnia_co2sat_gt": 0.15,
    "high_etco2frac_gt": 0.05,
}

LABEL_TO_SCENARIO = {
    0: "anesthesia",
    1: "oxygen_tank",
    2: "oxygen_wall",
    3: "valve_leak",
    4: "valve_obstruction",
    5: "ventilator_failure",
}


def _add_episode_id(df: pd.DataFrame) -> pd.DataFrame:
    # New episode when time resets (dataset is concatenated episodes where Time(s) restarts near 0.02)
    new_ep = (df["Time(s)"].lt(df["Time(s)"].shift(fill_value=df["Time(s)"].iloc[0]))) | (df.index == 0)
    df = df.copy()
    df["episode_id"] = new_ep.cumsum() - 1
    return df


def _forward_rolling_any(s: pd.Series, steps: int) -> pd.Series:
    # Whether the event occurs at any point in the next horizon (including now)
    return (
        s.astype("int8")
        .iloc[::-1]
        .rolling(window=steps + 1, min_periods=1)
        .max()
        .iloc[::-1]
        .astype(bool)
    )


def _forward_rolling_min(s: pd.Series, steps: int) -> pd.Series:
    return s.iloc[::-1].rolling(window=steps + 1, min_periods=1).min().iloc[::-1]


def main() -> None:
    df = pd.read_csv(SOURCE_CSV)

    df = _add_episode_id(df)
    df["scenario"] = df["label"].map(LABEL_TO_SCENARIO)

    dt = df.groupby("episode_id")["Time(s)"].diff().median()
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"Invalid dt computed from Time(s): {dt}")

    steps = int(round(HORIZON_S / float(dt)))

    # Current-state event flags
    df["hypoxia_now"] = df["OxygenSaturation"] < THRESHOLDS["hypoxia_spo2_lt"]
    df["severe_hypoxia_now"] = df["OxygenSaturation"] < THRESHOLDS["severe_hypoxia_spo2_lt"]
    df["apnea_now"] = df["RespirationRate(1/min)"] < THRESHOLDS["apnea_rr_lt"]
    df["hypoventilation_now"] = df["TotalAlveolarVentilation(L/min)"] < THRESHOLDS["hypovent_alvvent_lt"]
    df["low_tidal_volume_now"] = df["TidalVolume(mL)"] < THRESHOLDS["low_tidal_volume_ml_lt"]
    df["hypercapnia_now"] = df["CarbonDioxideSaturation"] > THRESHOLDS["hypercapnia_co2sat_gt"]
    df["severe_hypercapnia_now"] = df["CarbonDioxideSaturation"] > THRESHOLDS["severe_hypercapnia_co2sat_gt"]
    df["high_etco2_now"] = df["EndTidalCarbonDioxideFraction"] > THRESHOLDS["high_etco2frac_gt"]

    now_cols = [
        "hypoxia_now",
        "severe_hypoxia_now",
        "apnea_now",
        "hypoventilation_now",
        "low_tidal_volume_now",
        "hypercapnia_now",
        "severe_hypercapnia_now",
        "high_etco2_now",
    ]

    # Forward-looking labels for prediction
    suffix = f"next_{int(HORIZON_S)}s"
    for c in now_cols:
        df[c.replace("_now", f"_{suffix}")] = df.groupby("episode_id")[c].transform(lambda s: _forward_rolling_any(s, steps))

    # Composite respiratory compromise (covers oxygenation + ventilation + CO2 retention proxies)
    df[f"respiratory_compromise_{suffix}"] = df[
        [
            f"hypoxia_{suffix}",
            f"apnea_{suffix}",
            f"hypoventilation_{suffix}",
            f"low_tidal_volume_{suffix}",
            f"hypercapnia_{suffix}",
            f"high_etco2_{suffix}",
        ]
    ].any(axis=1)

    # Severity targets (useful if you later want regression or calibration)
    df[f"spo2_min_{suffix}"] = df.groupby("episode_id")["OxygenSaturation"].transform(lambda s: _forward_rolling_min(s, steps))
    df[f"map_min_{suffix}"] = df.groupby("episode_id")["MeanArterialPressure(mmHg)"].transform(lambda s: _forward_rolling_min(s, steps))

    df.to_csv(OUTPUT_CSV, index=False)

    risk_cols = [
        f"hypoxia_{suffix}",
        f"severe_hypoxia_{suffix}",
        f"apnea_{suffix}",
        f"hypoventilation_{suffix}",
        f"low_tidal_volume_{suffix}",
        f"hypercapnia_{suffix}",
        f"severe_hypercapnia_{suffix}",
        f"high_etco2_{suffix}",
        f"respiratory_compromise_{suffix}",
    ]
    pos_rates = (df[risk_cols].mean() * 100).round(2)

    spo2_min = df[f"spo2_min_{suffix}"]
    map_min = df[f"map_min_{suffix}"]

    print(f"Wrote {OUTPUT_CSV}")
    print(
        f"Rows: {len(df)}  Cols: {df.shape[1]}  Episodes: {df['episode_id'].nunique()}  dt_median: {dt:.5f}s  horizon_steps: {steps}"
    )
    print("Positive rates (% of rows):")
    print(pos_rates.to_string())
    print("Severity stats (min within horizon):")
    print(
        pd.DataFrame(
            {
                f"spo2_min_{suffix}": spo2_min.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).round(4),
                f"map_min_{suffix}": map_min.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).round(4),
            }
        ).to_string()
    )


if __name__ == "__main__":
    main()
