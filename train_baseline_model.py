import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATASET_CSV = "risk_labeled_dataset_30s.csv"
MODEL_OUT = "baseline_logreg_ovr.joblib"
METRICS_OUT = "baseline_metrics.json"

# Use only signals available at inference time (no scenario/episode/label leakage, no derived now/next flags)
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

# Evidence-backed risk targets from the derived dataset
TARGET_COLS = [
    "hypoxia_next_30s",
    "apnea_next_30s",
    "hypoventilation_next_30s",
    "low_tidal_volume_next_30s",
    "hypercapnia_next_30s",
    "high_etco2_next_30s",
    "respiratory_compromise_next_30s",
]

GROUP_COL = "episode_id"
TIME_COL = "Time(s)"

# Forward-looking labels look ahead 30s, so we purge a full horizon at the train/test boundary.
HORIZON_S = 30.0
TRAIN_FRACTION = 0.7

# Raw data is sampled at ~50Hz; downsample to reduce training time for this baseline.
DOWNSAMPLE_STEP = 5  # keep every 5th row within each episode (~10Hz)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    # roc_auc_score errors if only one class is present
    if np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def _to_python(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main() -> None:
    df = pd.read_csv(DATASET_CSV)
    n_rows_raw = int(len(df))

    # Deterministic ordering and lightweight downsampling (per-episode)
    df = df.sort_values([GROUP_COL, TIME_COL]).reset_index(drop=True)
    if DOWNSAMPLE_STEP > 1:
        keep = (df.groupby(GROUP_COL).cumcount() % DOWNSAMPLE_STEP) == 0
        df = df.loc[keep].reset_index(drop=True)

    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    missing_targets = [c for c in TARGET_COLS if c not in df.columns]
    if missing_features or missing_targets:
        raise ValueError({"missing_features": missing_features, "missing_targets": missing_targets})

    X = df[FEATURE_COLS]
    y = df[TARGET_COLS].astype(int)

    # Baseline model: standardize features + one-vs-rest logistic regression
    # class_weight='balanced' helps with label imbalance
    base_lr = LogisticRegression(
        solver="saga",
        penalty="l2",
        max_iter=2000,
        class_weight="balanced",
        n_jobs=None,
        random_state=42,
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), FEATURE_COLS)],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", OneVsRestClassifier(base_lr)),
        ]
    )

    # Time-based split within each episode (streaming-like): train on early timeline,
    # test on later timeline, with a purge gap of HORIZON_S to prevent label leakage
    # across the split boundary (because labels look ahead into the future).
    if TIME_COL not in df.columns:
        raise ValueError(f"Missing required column for time split: {TIME_COL}")

    train_mask = np.zeros(len(df), dtype=bool)
    test_mask = np.zeros(len(df), dtype=bool)
    split_info: list[dict] = []

    for eid, g in df.groupby(GROUP_COL, sort=True):
        tmax = float(g[TIME_COL].max())
        split_t = tmax * TRAIN_FRACTION
        train_end_t = split_t - HORIZON_S

        idx = g.index.to_numpy()
        t = g[TIME_COL].to_numpy(dtype=float)

        tr = t <= train_end_t
        te = t >= split_t

        train_mask[idx] = tr
        test_mask[idx] = te

        split_info.append(
            {
                "episode_id": int(eid),
                "t_max_s": tmax,
                "split_t_s": split_t,
                "train_end_t_s": train_end_t,
                "n_train": int(tr.sum()),
                "n_test": int(te.sum()),
            }
        )

    X_train, X_test = X.iloc[train_mask], X.iloc[test_mask]
    y_train, y_test = y.iloc[train_mask], y.iloc[test_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError({"n_train": int(len(X_train)), "n_test": int(len(X_test)), "split_info": split_info})

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)  # shape: (n_samples, n_targets)
    y_pred = (y_proba >= 0.5).astype(int)

    per_target: dict[str, dict[str, float | None]] = {}
    for j, target in enumerate(TARGET_COLS):
        yt = y_test[target].to_numpy()
        ys = y_proba[:, j]
        yp = y_pred[:, j]

        per_target[target] = {
            "pos_rate_train": float(y_train[target].to_numpy().mean()),
            "pos_rate_test": float(yt.mean()),
            "roc_auc": _safe_auc(yt, ys),
            "avg_precision": _safe_ap(yt, ys),
            "precision@0.5": float(precision_score(yt, yp, zero_division=0)),
            "recall@0.5": float(recall_score(yt, yp, zero_division=0)),
            "f1@0.5": float(f1_score(yt, yp, zero_division=0)),
        }

    # Save the model trained on the training split used for evaluation
    dump(model, MODEL_OUT)

    out = {
        "dataset": {
            "path": str(Path(DATASET_CSV).resolve()),
            "n_rows_raw": n_rows_raw,
            "n_rows_used": int(len(df)),
            "downsample_step": int(DOWNSAMPLE_STEP),
            "n_episodes": int(df[GROUP_COL].nunique()),
            "feature_cols": FEATURE_COLS,
            "target_cols": TARGET_COLS,
        },
        "evaluation": {
            "scheme": "WithinEpisodeTimeSplitWithPurge",
            "group_col": GROUP_COL,
            "time_col": TIME_COL,
            "train_fraction": TRAIN_FRACTION,
            "horizon_s": HORIZON_S,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "split_info": split_info,
            "per_target": per_target,
        },
        "model": {
            "type": "Pipeline(StandardScaler -> OneVsRest(LogisticRegression))",
            "threshold": 0.5,
            "artifact": str(Path(MODEL_OUT).resolve()),
        },
    }

    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=_to_python)

    print(f"Saved model: {MODEL_OUT}")
    print(f"Saved metrics: {METRICS_OUT}")
    print("Test metrics (time-split within episode, purged):")
    for target in TARGET_COLS:
        m = per_target[target]
        print(
            f"- {target}: AUC={m['roc_auc']}, AP={m['avg_precision']}, F1@0.5={m['f1@0.5']}, pos_rate_test={m['pos_rate_test']:.4f}"
        )


if __name__ == "__main__":
    main()
