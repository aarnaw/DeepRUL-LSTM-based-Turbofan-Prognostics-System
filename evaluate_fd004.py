import json
import pickle
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.lstm_model import LSTMRULModel


PROCESSED_DIR = Path("data/processed")
MODEL_PATH = Path("model/best_model.pth")
DEVICE = torch.device("cpu")
RUL_CAP = 125.0


def main() -> None:
    X_test = np.load(PROCESSED_DIR / "X_test_last.npy")
    y_test = np.load(PROCESSED_DIR / "y_test_last.npy")
    test_unit_ids = np.load(PROCESSED_DIR / "test_unit_ids.npy")

    with open(PROCESSED_DIR / "test_engine_trajectories.pkl", "rb") as fp:
        trajectories = pickle.load(fp)

    input_size = X_test.shape[-1]
    model = LSTMRULModel(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32, device=DEVICE)).cpu().numpy() * RUL_CAP

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R2  : {r2:.4f}")
    print(f"Prediction STD: {float(np.std(preds)):.4f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.6, s=18, color="#f59e0b")
    lim = max(float(np.max(y_test)), float(np.max(preds)))
    plt.plot([0, lim], [0, lim], "r--", linewidth=1.4)
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.title("FD004 Test: Predicted vs Actual")
    plt.tight_layout()
    plt.savefig("model/test_pred_vs_actual.png", dpi=150)

    final_pairs = sorted([(uid, trajectories[int(uid)]["final_rul"]) for uid in test_unit_ids], key=lambda x: x[1])
    selected = [int(final_pairs[-1][0]), int(final_pairs[len(final_pairs) // 2][0]), int(final_pairs[0][0])]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    for ax, engine_id in zip(axes, selected):
        item = trajectories[engine_id]
        seqs = torch.tensor(item["sequences"], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            pred_traj = model(seqs).cpu().numpy() * RUL_CAP

        cycles = np.array(item["cycles"])
        actual = np.array(item["actual_rul"])

        ax.plot(cycles, actual, label="Actual", color="#2563eb", linewidth=1.5)
        ax.plot(cycles, pred_traj, label="Predicted", color="#f59e0b", linewidth=1.5)
        ax.set_ylabel("RUL")
        ax.set_title(f"Engine {engine_id} trajectory")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Cycle")
    plt.tight_layout()
    plt.savefig("model/test_trajectory_examples.png", dpi=150)

    with open(PROCESSED_DIR / "latest_test_metrics.json", "w", encoding="utf-8") as fp:
        json.dump({"rmse": rmse, "mae": mae, "r2": r2, "prediction_std": float(np.std(preds))}, fp, indent=2)


if __name__ == "__main__":
    main()
