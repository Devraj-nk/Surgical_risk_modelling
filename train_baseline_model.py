import argparse
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

PROJECT_ROOT = Path(__file__).resolve().parent

# Prefer the cleaned updated dataset if it exists.
DATASET_CSV_DEFAULT = (
    PROJECT_ROOT / "preprocessing_out" / "risk_labeled_dataset_30s.updated.cleaned.csv"
)
if not DATASET_CSV_DEFAULT.exists():
    DATASET_CSV_DEFAULT = PROJECT_ROOT / "risk_labeled_dataset_30s.csv"

MODEL_OUT_DEFAULT = PROJECT_ROOT / "baseline_logreg_ovr.joblib"
METRICS_OUT_DEFAULT = PROJECT_ROOT / "baseline_metrics.json"

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
TRAIN_FRACTION = 0.6
VAL_FRACTION = 0.2

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


def _subset_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # For multilabel classification, sklearn's accuracy_score is subset accuracy.
    return float((y_true == y_pred).all(axis=1).mean())


def _hamming_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true != y_pred).mean())


def _split_masks(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Create train/val/test masks within each episode using time fractions.

    We use purge gaps of HORIZON_S between segments because labels look ahead.
    """

    train_mask = np.zeros(len(df), dtype=bool)
    val_mask = np.zeros(len(df), dtype=bool)
    test_mask = np.zeros(len(df), dtype=bool)
    split_info: list[dict] = []

    for eid, g in df.groupby(GROUP_COL, sort=True):
        tmax = float(g[TIME_COL].max())
        split1 = tmax * TRAIN_FRACTION
        split2 = tmax * (TRAIN_FRACTION + VAL_FRACTION)

        train_end_t = split1 - HORIZON_S
        val_start_t = split1
        val_end_t = split2 - HORIZON_S
        test_start_t = split2

        idx = g.index.to_numpy()
        t = g[TIME_COL].to_numpy(dtype=float)

        tr = t <= train_end_t
        va = (t >= val_start_t) & (t <= val_end_t)
        te = t >= test_start_t

        train_mask[idx] = tr
        val_mask[idx] = va
        test_mask[idx] = te

        split_info.append(
            {
                "episode_id": int(eid),
                "t_max_s": tmax,
                "split1_t_s": split1,
                "split2_t_s": split2,
                "train_end_t_s": train_end_t,
                "val_start_t_s": val_start_t,
                "val_end_t_s": val_end_t,
                "test_start_t_s": test_start_t,
                "n_train": int(tr.sum()),
                "n_val": int(va.sum()),
                "n_test": int(te.sum()),
            }
        )

    return train_mask, val_mask, test_mask, split_info


def _evaluate_split(
    *,
    name: str,
    y_true: pd.DataFrame,
    y_proba: np.ndarray,
    threshold: float,
) -> dict:
    y_true_np = y_true.to_numpy(dtype=int)
    y_pred = (y_proba >= threshold).astype(int)

    # Mean per-label accuracy: average over targets of (correct predictions).
    mean_label_accuracy = float((y_true_np == y_pred).mean(axis=0).mean())

    per_target: dict[str, dict[str, float | None]] = {}
    aucs: list[float] = []
    aps: list[float] = []

    for j, target in enumerate(TARGET_COLS):
        yt = y_true_np[:, j]
        ys = y_proba[:, j]
        yp = y_pred[:, j]

        auc = _safe_auc(yt, ys)
        ap = _safe_ap(yt, ys)
        if auc is not None:
            aucs.append(float(auc))
        if ap is not None:
            aps.append(float(ap))

        per_target[target] = {
            "pos_rate": float(yt.mean()),
            "roc_auc": auc,
            "avg_precision": ap,
            "precision@0.5": float(precision_score(yt, yp, zero_division=0)),
            "recall@0.5": float(recall_score(yt, yp, zero_division=0)),
            "f1@0.5": float(f1_score(yt, yp, zero_division=0)),
        }

    out = {
        "name": name,
        "n": int(len(y_true)),
        "subset_accuracy": _subset_accuracy(y_true_np, y_pred),
        "mean_label_accuracy": mean_label_accuracy,
        "hamming_loss": _hamming_loss(y_true_np, y_pred),
        "macro_f1@0.5": float(f1_score(y_true_np, y_pred, average="macro", zero_division=0)),
        "micro_f1@0.5": float(f1_score(y_true_np, y_pred, average="micro", zero_division=0)),
        "mean_roc_auc": float(np.mean(aucs)) if aucs else None,
        "mean_avg_precision": float(np.mean(aps)) if aps else None,
        "per_target": per_target,
    }
    return out


def _to_python(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline OVR logistic regression risk model")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DATASET_CSV_DEFAULT),
        help="Path to dataset CSV (default prefers preprocessing_out/*.cleaned.csv if present).",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default=str(MODEL_OUT_DEFAULT),
        help="Output path for joblib model artifact.",
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default=str(METRICS_OUT_DEFAULT),
        help="Output path for metrics JSON.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold used for printing thresholded metrics (default 0.5).",
    )
    args = parser.parse_args()

    dataset_csv = Path(args.dataset)
    model_out = Path(args.model_out)
    metrics_out = Path(args.metrics_out)
    threshold = float(args.threshold)

    df = pd.read_csv(dataset_csv)
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

    # Train/val/test time-based split within each episode (streaming-like), with purge gaps.
    if TIME_COL not in df.columns:
        raise ValueError(f"Missing required column for time split: {TIME_COL}")

    train_mask, val_mask, test_mask, split_info = _split_masks(df)

    X_train, y_train = X.iloc[train_mask], y.iloc[train_mask]
    X_val, y_val = X.iloc[val_mask], y.iloc[val_mask]
    X_test, y_test = X.iloc[test_mask], y.iloc[test_mask]

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError(
            {
                "n_train": int(len(X_train)),
                "n_val": int(len(X_val)),
                "n_test": int(len(X_test)),
                "split_info": split_info,
            }
        )

    model.fit(X_train, y_train)

    # Evaluate splits
    y_proba_train = model.predict_proba(X_train)
    y_proba_val = model.predict_proba(X_val)
    y_proba_test = model.predict_proba(X_test)

    metrics_train = _evaluate_split(name="train", y_true=y_train, y_proba=y_proba_train, threshold=threshold)
    metrics_val = _evaluate_split(name="validation", y_true=y_val, y_proba=y_proba_val, threshold=threshold)
    metrics_test = _evaluate_split(name="test", y_true=y_test, y_proba=y_proba_test, threshold=threshold)

    # Save the model trained on the training split used for evaluation
    model_out.parent.mkdir(parents=True, exist_ok=True)
    dump(model, str(model_out))

    out = {
        "dataset": {
            "path": str(dataset_csv.resolve()),
            "n_rows_raw": n_rows_raw,
            "n_rows_used": int(len(df)),
            "downsample_step": int(DOWNSAMPLE_STEP),
            "n_episodes": int(df[GROUP_COL].nunique()),
            "feature_cols": FEATURE_COLS,
            "target_cols": TARGET_COLS,
        },
        "evaluation": {
            "scheme": "WithinEpisodeTrainValTestTimeSplitWithPurge",
            "group_col": GROUP_COL,
            "time_col": TIME_COL,
            "train_fraction": TRAIN_FRACTION,
            "val_fraction": VAL_FRACTION,
            "horizon_s": HORIZON_S,
            "threshold": threshold,
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
            "split_info": split_info,
            "splits": {
                "train": metrics_train,
                "val": metrics_val,
                "test": metrics_test,
            },
        },
        "model": {
            "type": "Pipeline(StandardScaler -> OneVsRest(LogisticRegression))",
            "artifact": str(model_out.resolve()),
        },
    }

    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=_to_python)

    print(f"Dataset: {dataset_csv.resolve()}")
    print(f"Saved model: {model_out.resolve()}")
    print(f"Saved metrics: {metrics_out.resolve()}")
    print(
        "\nNote: R² is a regression metric; this is multilabel classification, so we report subset_accuracy/F1/AUC/AP instead."
    )

    def _print_split(m: dict) -> None:
        print(
            f"[{m['name']}] n={m['n']:,}  accuracy(subset)={m['subset_accuracy']:.4f}  accuracy(mean_label)={m['mean_label_accuracy']:.4f}  micro_F1={m['micro_f1@0.5']:.4f}  "
            f"macro_F1={m['macro_f1@0.5']:.4f}  hamming={m['hamming_loss']:.4f}  mean_AUC={m['mean_roc_auc']}  mean_AP={m['mean_avg_precision']}"
        )

    print("\nSplit summary metrics:")
    _print_split(metrics_train)
    _print_split(metrics_val)
    _print_split(metrics_test)

    print("\nPer-target test metrics:")
    for target in TARGET_COLS:
        mt = metrics_test["per_target"][target]
        print(
            f"- {target}: AUC={mt['roc_auc']}, AP={mt['avg_precision']}, F1@0.5={mt['f1@0.5']}, pos_rate={mt['pos_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
