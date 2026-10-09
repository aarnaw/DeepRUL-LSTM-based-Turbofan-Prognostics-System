import pickle
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.lstm_model import LSTMRULModel, mc_dropout_predict


PROCESSED_DIR = Path("data/processed")
MODEL_PATH = Path("model/best_model.pth")
DEVICE = torch.device("cpu")
RUL_CAP = 125.0


def main() -> None:
    with open(PROCESSED_DIR / "test_engine_trajectories.pkl", "rb") as fp:
        trajectories = pickle.load(fp)

    engine_ids = sorted(trajectories.keys())
    engine_id = engine_ids[len(engine_ids) // 2]
    payload = trajectories[engine_id]

    input_size = payload["sequences"].shape[-1]
    model = LSTMRULModel(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    sequences = torch.tensor(payload["sequences"], dtype=torch.float32, device=DEVICE)
    cycles = np.array(payload["cycles"])
    actual = np.array(payload["actual_rul"])

    mean, std = mc_dropout_predict(model, sequences, n_passes=50)
    mean = mean.numpy() * RUL_CAP
    std = std.numpy() * RUL_CAP

    lower = mean - 1.96 * std
    upper = mean + 1.96 * std

    plt.figure(figsize=(12, 5))
    plt.plot(cycles, actual, label="Actual RUL", color="#2563eb", linewidth=1.7)
    plt.plot(cycles, mean, label="Predicted Mean", color="#f59e0b", linewidth=1.7)
    plt.fill_between(cycles, lower, upper, color="#f59e0b", alpha=0.25, label="95% CI")
    plt.xlabel("Cycle")
    plt.ylabel("RUL")
    plt.title(f"FD004 Engine {engine_id}: RUL Inference with Uncertainty")
    plt.legend()
    plt.tight_layout()
    plt.savefig("model/inference_uncertainty_curve.png", dpi=150)

    final_mean = float(mean[-1])
    final_std = float(std[-1])
    print(f"Engine ID: {engine_id}")
    print(f"Final predicted RUL: {final_mean:.2f}")
    print(f"Final uncertainty (std): {final_std:.2f}")
    print(f"Final 95% CI: [{(final_mean - 1.96 * final_std):.2f}, {(final_mean + 1.96 * final_std):.2f}]")


if __name__ == "__main__":
    main()
