# FD004 Aircraft Engine RUL Project

This project predicts Remaining Useful Life (RUL) for turbofan engines using the NASA C-MAPSS FD004 dataset.

It includes:
- A full data pipeline (preprocess -> train -> evaluate -> inference)
- A PyTorch LSTM model with uncertainty estimation (Monte Carlo Dropout)
- A FastAPI backend for real-time predictions
- A cockpit-style frontend for simulation and CSV-based inference

The goal is practical and educational: you can run everything locally and understand how an end-to-end ML system is built.

## 1) What Problem This Solves

Aircraft engines degrade over time. Instead of waiting for failure, we estimate how many cycles are left before failure.

That estimate is called Remaining Useful Life (RUL).

This repository predicts RUL from engine sensor time-series data and also reports uncertainty, so predictions are not just a single number.

## 2) Tech Stack (What We Used)

- Python 3
- NumPy and pandas for data processing
- scikit-learn for scaling and metrics
- PyTorch for the LSTM model
- FastAPI + Uvicorn for serving predictions
- HTML/CSS/JavaScript + Chart.js for the frontend UI

## 3) High-Level Architecture

1. Raw FD004 text files are loaded and cleaned.
2. Features are engineered and scaled.
3. Fixed-length sequences of 30 timesteps are created.
4. LSTM is trained to predict normalized RUL.
5. Best model checkpoint is saved.
6. API loads model and exposes prediction endpoint.
7. Frontend sends sequence data to API and displays RUL + uncertainty.

## 4) Repository Structure (What Each Folder Does)

- data/raw
  - Original FD004 files: train/test/RUL tables.
- data/processed
  - Generated arrays, scaler, feature metadata, trajectories, and metrics.
- scripts
  - Core ML pipeline scripts:
    - preprocess_fd004.py
    - train_fd004.py
    - evaluate_fd004.py
    - inference_fd004.py
- model
  - LSTM model code and generated model/artifact files.
- api
  - FastAPI inference service.
- frontend
  - Cockpit UI (single-page frontend).

## 5) Data Pipeline Details (How We Did It)

### 5.1 Preprocessing

Implemented in scripts/preprocess_fd004.py.

Main steps:
- Load FD004 train/test/RUL files.
- Drop selected low-value columns.
- Build cycle_norm feature per engine.
- Compute train RUL as max_cycle - current_cycle.
- Clip RUL at 125 cycles.
- Scale sensor columns with MinMaxScaler.
- Build train sequences with length = 30.
- Build last-window test sequences for test-time scoring.
- Balance training sequences by downsampling capped-RUL examples.
- Save arrays and metadata to data/processed.

Generated files include:
- X_train_sequences.npy, y_train_sequences.npy
- X_test_last.npy, y_test_last.npy
- train_sequence_unit_ids.npy, test_unit_ids.npy
- feature_scaler.pkl
- feature_columns.json
- dataset_config.json
- test_engine_trajectories.pkl
- test_sequences_sample.json
- latest_test_metrics.json (after evaluation)

### 5.2 Training

Implemented in scripts/train_fd004.py.

Main design choices:
- Model: 2-layer LSTM + dropout + linear head.
- Target normalization: y_norm = y / 125.
- Loss: weighted MSE to prioritize lower-RUL regions.
- Early stopping with patience = 10.
- Sanity check: mandatory overfit test on 50 samples before full training.

Outputs:
- model/best_model.pth
- model/training_loss_curve.png
- model/predicted_vs_actual_rul.png

### 5.3 Evaluation

Implemented in scripts/evaluate_fd004.py.

Computes:
- RMSE
- MAE
- R2

Also saves:
- model/test_pred_vs_actual.png
- model/test_trajectory_examples.png
- data/processed/latest_test_metrics.json

### 5.4 Inference with Uncertainty

Implemented in scripts/inference_fd004.py.

Uses Monte Carlo Dropout:
- Runs multiple stochastic forward passes (n_passes = 50)
- Returns mean prediction and standard deviation
- Builds 95% confidence interval using:

CI = mean +/- 1.96 * std

Saves:
- model/inference_uncertainty_curve.png

## 6) Model Internals

Defined in model/lstm_model.py.

- LSTMRULModel:
	- Input shape: batch x 30 x feature_count
	- LSTM output at last timestep -> dropout -> linear regression head
- mc_dropout_predict:
	- Keeps dropout active during inference
	- Returns mean and std over multiple passes

## 7) API (How Predictions Are Served)

Implemented in api/main.py.

Endpoints:
- GET /health
- POST /predict_rul

POST /predict_rul accepts one of:
- sequence: nested float array with shape 30 x feature_count
- csv_data: CSV string with exactly 30 rows and feature_count columns

Response:
- rul
- uncertainty
- confidence_interval_95

Notes:
- Feature count is loaded from data/processed/feature_columns.json.
- If a CSV has one header row, it is skipped automatically.

## 8) Frontend

Implemented in frontend/index.html.

Features:
- Simulate Engine mode (sample-based request)
- Inject Engine Telemetry mode (drag-and-drop or click file upload)
- Cockpit-style gauge, diagnostics, lifecycle bar, and telemetry chart
- Loading sequence animation before inference call

## 9) Quick Start (For First-Time Users)

### Step A: Clone and enter project

```bash
git clone <your-repo-url>
cd aiproj
```

### Step B: Create and activate environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step C: Install dependencies

```bash
pip install -r requirements.txt
```

### Step D: Run full ML pipeline

```bash
python scripts/preprocess_fd004.py
python scripts/train_fd004.py
python scripts/evaluate_fd004.py
python scripts/inference_fd004.py
```

### Step E: Start API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Step F: Start frontend server (new terminal)

```bash
python3 -m http.server 5500
```

Open in browser:
- http://localhost:5500/frontend/index.html

## 10) API Usage Example (Direct Test)

Health check:

```bash
curl http://localhost:8000/health
```

Prediction with CSV payload using Python:

```python
import json
import urllib.request

csv_text = "... 30 rows of comma-separated features ..."
payload = json.dumps({"engine_id": 1, "csv_data": csv_text}).encode("utf-8")

req = urllib.request.Request(
    "http://localhost:8000/predict_rul",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(req) as response:
    print(response.read().decode("utf-8"))
```

## 11) Important Configuration Values

- Sequence length: 30
- RUL cap: 125
- Default device: CPU
- MC dropout passes: 50

## 12) Troubleshooting

### API says model checkpoint missing

Run training first:

```bash
python scripts/train_fd004.py
```

### Upload fails due wrong shape

Your input must have:
- Exactly 30 timesteps (rows)
- Exactly feature_count values per row

feature_count is set by data/processed/feature_columns.json.

### Frontend loads but prediction fails

Make sure both are running:
- API on port 8000
- Static file server on port 5500

## 13) Learning Path (If You Are New)

Read and run in this order:
1. scripts/preprocess_fd004.py
2. scripts/train_fd004.py
3. scripts/evaluate_fd004.py
4. scripts/inference_fd004.py
5. api/main.py
6. frontend/index.html

This order mirrors how a production ML system is built: data -> model -> validation -> serving -> UI.

## 14) Project Scope

This repository is intentionally FD004-only to keep the codebase focused and easier to understand.

If you want to extend it later, typical next steps are:
- Add FD001/FD002/FD003 support
- Add model versioning and experiment tracking
- Add Docker deployment and CI checks
