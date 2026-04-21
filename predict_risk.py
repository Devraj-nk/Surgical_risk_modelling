import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import load


MODEL_DEFAULT = "baseline_logreg_ovr.joblib"

# Must match training exactly
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


def _coerce_float(value: Any, *, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid numeric value for '{name}': {value!r}") from e


def _load_features_from_json(path: Path) -> dict[str, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "features" in raw and isinstance(raw["features"], dict):
        raw = raw["features"]

    if not isinstance(raw, dict):
        raise ValueError("JSON must be an object with keys equal to feature names (or {features: {...}}).")

    missing = [c for c in FEATURE_COLS if c not in raw]
    if missing:
        raise ValueError({"missing_features": missing})

    return {c: _coerce_float(raw[c], name=c) for c in FEATURE_COLS}


def _load_features_from_values(values: list[float]) -> dict[str, float]:
    if len(values) != len(FEATURE_COLS):
        raise ValueError(
            f"Expected {len(FEATURE_COLS)} numeric values (one per feature), got {len(values)}."
        )
    return {c: _coerce_float(v, name=c) for c, v in zip(FEATURE_COLS, values, strict=True)}


def _load_features_interactive() -> dict[str, float]:
    print("Enter the current physiological values (same units as the dataset).")
    print("Press Ctrl+C to abort.\n")

    out: dict[str, float] = {}
    for col in FEATURE_COLS:
        while True:
            s = input(f"{col}: ").strip()
            if not s:
                print("  Value required.")
                continue
            try:
                out[col] = _coerce_float(s, name=col)
                break
            except ValueError as e:
                print(f"  {e}")

    return out


def _predict(model, features: dict[str, float]) -> dict[str, float]:
    X = pd.DataFrame([features], columns=FEATURE_COLS)

    # For OneVsRestClassifier, predict_proba returns (n_samples, n_targets)
    proba = model.predict_proba(X)
    if hasattr(proba, "shape") and proba.shape[0] == 1:
        probs = proba[0]
    else:
        # Fallback: attempt to coerce to a 1d array-like
        probs = list(proba)[0]

    if len(probs) != len(TARGET_COLS):
        raise RuntimeError(
            f"Unexpected predict_proba shape: got {len(probs)} probabilities for {len(TARGET_COLS)} targets."
        )

    return {t: float(p) for t, p in zip(TARGET_COLS, probs, strict=True)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load the baseline risk model and predict next-30s risks from a single physiological snapshot."
    )
    parser.add_argument(
        "--model",
        default=MODEL_DEFAULT,
        help=f"Path to joblib model artifact (default: {MODEL_DEFAULT}).",
    )

    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--json",
        type=str,
        help="Path to a JSON file with feature keys (or {features: {...}}).",
    )
    src.add_argument(
        "--values",
        type=float,
        nargs="+",
        help="Provide all 14 feature values in order (see --print-schema).",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for printing a boolean flag per risk (default: 0.5).",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--print-schema",
        action="store_true",
        help="Print the expected feature order/names and exit.",
    )

    args = parser.parse_args()

    if args.print_schema:
        print("Expected features (in order):")
        for i, c in enumerate(FEATURE_COLS, start=1):
            print(f"{i:2d}. {c}")
        return

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = load(model_path)

    if args.json:
        features = _load_features_from_json(Path(args.json))
    elif args.values is not None:
        features = _load_features_from_values(args.values)
    else:
        features = _load_features_interactive()

    probs = _predict(model, features)

    threshold = float(args.threshold)
    flags = {k: bool(v >= threshold) for k, v in probs.items()}

    if args.format == "json":
        print(
            json.dumps(
                {
                    "features": features,
                    "risk_probabilities": probs,
                    "risk_flags": flags,
                    "threshold": threshold,
                },
                indent=2,
            )
        )
        return

    print("\nRisk probabilities (next 30s):")
    for k in TARGET_COLS:
        p = probs[k]
        flag = "YES" if flags[k] else "no"
        print(f"- {k}: {p:.4f}  (>= {threshold:g}? {flag})")


if __name__ == "__main__":
    main()
