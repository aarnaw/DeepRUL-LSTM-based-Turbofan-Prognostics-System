import csv
import io
import json
from pathlib import Path
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from model.lstm_model import LSTMRULModel, mc_dropout_predict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "data" / "processed" / "feature_columns.json"
RUL_CAP = 125.0

FEATURE_COUNT = 15
if FEATURE_COLUMNS_PATH.exists():
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as fp:
        FEATURE_COUNT = len(json.load(fp))


class PredictRequest(BaseModel):
    engine_id: int = Field(0, ge=0)
    sequence: List[List[float]] | None = None
    csv_data: str | None = None

    @field_validator("sequence", mode="before")
    @classmethod
    def validate_sequence_shape(cls, value):
        if value is None:
            return value
        if len(value) != 30:
            raise ValueError("sequence must have exactly 30 timesteps")
        for row in value:
            if len(row) != FEATURE_COUNT:
                raise ValueError(f"each timestep must contain exactly {FEATURE_COUNT} features")
        return value


class PredictResponse(BaseModel):
    rul: float
    uncertainty: float
    confidence_interval_95: List[float]


app = FastAPI(title="FD004 RUL Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = PROJECT_ROOT / "model" / "best_model.pth"
MODEL = None
DEVICE = torch.device("cpu")


def parse_csv_sequence(csv_data: str) -> List[List[float]]:
    rows: List[List[float]] = []
    reader = csv.reader(io.StringIO(csv_data.strip()))
    for line_no, row in enumerate(reader, start=1):
        cleaned = [col.strip() for col in row if col.strip()]
        if not cleaned:
            continue

        try:
            parsed = [float(v) for v in cleaned]
        except ValueError as exc:
            # Allow a single header row like: s_1,s_2,...
            if not rows:
                continue
            raise ValueError(
                f"csv_data row {line_no} contains non-numeric values. "
                "Use only numeric values after the optional header row."
            ) from exc

        rows.append(parsed)

    if len(rows) != 30:
        raise ValueError("csv_data must contain exactly 30 rows")
    for i, row in enumerate(rows, start=1):
        if len(row) != FEATURE_COUNT:
            raise ValueError(
                f"each csv_data row must contain exactly {FEATURE_COUNT} values; "
                f"row {i} has {len(row)}"
            )
    return rows


@app.on_event("startup")
def load_model_once() -> None:
    global MODEL
    model = LSTMRULModel(input_size=FEATURE_COUNT, hidden_size=128, num_layers=2, dropout=0.2)
    if not MODEL_PATH.exists():
        MODEL = None
        return

    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    MODEL = model


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "dataset": "FD004", "feature_count": FEATURE_COUNT}


@app.post("/predict_rul", response_model=PredictResponse)
def predict_rul(payload: PredictRequest) -> PredictResponse:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model checkpoint missing. Run training first.")

    if payload.sequence is None and payload.csv_data is None:
        raise HTTPException(status_code=422, detail="Provide either 'sequence' or 'csv_data'.")

    try:
        sequence = payload.sequence if payload.sequence is not None else parse_csv_sequence(payload.csv_data or "")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    x = torch.tensor([sequence], dtype=torch.float32, device=DEVICE)
    mean, std = mc_dropout_predict(MODEL, x, n_passes=50)

    rul = float(mean.item()) * RUL_CAP
    unc = float(std.item()) * RUL_CAP
    ci_low = rul - 1.96 * unc
    ci_high = rul + 1.96 * unc

    return PredictResponse(
        rul=round(rul, 1),
        uncertainty=round(unc, 2),
        confidence_interval_95=[round(ci_low, 1), round(ci_high, 1)],
    )
