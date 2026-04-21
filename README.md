# Surgical Risk Modelling (BioGears + Baseline ML + 3D UI)

This project combines:

- **BioGears-generated physiology time-series data** (CSV)
- A **baseline ML model** that predicts short-horizon respiratory risk events from a single physiological snapshot
- A lightweight **3D “digital twin” UI** (Three.js + GLB) with realtime monitor-style graphs

---

## Repo contents

- `final_numeric_physiological_dataset.csv`
  - Numeric physiology time-series exported from BioGears simulations.
  - Contains multiple scenarios/episodes concatenated into a single file (time resets between episodes).
- `preprocessing.ipynb`
  - Notebook used for exploratory preprocessing/inspection.
- `build_risk_dataset.py`
  - Creates a supervised learning dataset with forward-looking risk labels.
- `risk_labeled_dataset_30s.csv`
  - Output of `build_risk_dataset.py` (generated file).
- `train_baseline_model.py`
  - Trains a baseline multi-label classifier and writes artifacts.
- `baseline_logreg_ovr.joblib`, `baseline_metrics.json`
  - Trained model artifact + evaluation summary.
- `predict_risk.py`
  - CLI for predicting risk probabilities from a single physiological snapshot.
- `ui/server.py`
  - Minimal Python server that hosts the UI.
- `ui/static/`
  - Frontend (HTML/CSS/JS) + `.glb` heart assets.

---

## Prerequisites

- Python 3.9+ (recommended: 3.10+)
- pip


---

## Clone and set up

```bash
git clone <your-repo-url>
cd surgical_risk_modelling

python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn joblib
---

## Run the 3D UI

The UI is served by a small Python HTTP server.

```bash
python ui/server.py
```

Then open:

- http://127.0.0.1:8000

### UI inputs

The UI exposes **three high-level controls**:

- **Anesthesia drugs (%)**
- **Oxygen supply (%)**
- **Blood loss (mL)**

### UI outputs

The UI displays:

- Realtime monitor-style graphs for:
  - **Heart Rate (HR)**
  - **Mean Arterial Pressure (MAP)**
  - **Cardiac Output (CO)**
- A **risk score** indicator (prototype heuristic)
- A JSON panel with the current inputs and derived values

### 3D integration

- The heart is loaded from a **GLB** file using Three.js (`GLTFLoader`) with orbit controls.
- The heart’s **color tint** reacts to inputs (e.g., oxygenation/blood loss effects).
- A **heartbeat animation** is applied as a safe, standalone component (scale-based pulse) that can be disabled without affecting the rest of the UI.

To swap the heart model, place your `.glb` under `ui/static/` and update the `GLB_URL` constant in `ui/static/app.js`.

---

## Dataset and preprocessing

### BioGears dataset

`final_numeric_physiological_dataset.csv` contains numeric physiological signals exported from BioGears simulation runs.

Key characteristics:

- Time-series measurements (many signals per time step)
- Multiple scenarios concatenated; **episode boundaries can be detected** because the time column restarts near zero

### Preprocessing notebook

`preprocessing.ipynb` is used to:

- Inspect signals and distributions
- Validate episode boundaries/time resets
- Prototype feature/label ideas

---

## Build the risk-labeled dataset

This step converts the raw time-series into a supervised learning table with **forward-looking labels**.

```bash
python build_risk_dataset.py
```

Output:

- `risk_labeled_dataset_30s.csv`

---

## Train the baseline model

Train a baseline multi-label classifier (one-vs-rest logistic regression pipeline):

```bash
python train_baseline_model.py
```

Outputs:

- `baseline_logreg_ovr.joblib` (model artifact)
- `baseline_metrics.json` (evaluation summary)

---

## Run inference from the command line

Interactive mode:

```bash
python predict_risk.py
```

From a JSON file:

```bash
python predict_risk.py --json input_example.json --format json
```

To print the expected feature schema/order:

```bash
python predict_risk.py --print-schema
```

---

The response contains:

- Echoed features
- Per-target risk probabilities

---

## Notes / troubleshooting

- If the UI doesn’t load the 3D model, confirm the `.glb` file exists in `ui/static/` and that `GLB_URL` matches the filename.
- If you retrain the model and want the server to use it, replace `baseline_logreg_ovr.joblib` (or edit `MODEL_PATH_DEFAULT` in `ui/server.py`).
