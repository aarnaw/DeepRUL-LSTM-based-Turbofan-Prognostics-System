from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.lstm_model import LSTMRULModel


PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
RUL_CAP = 125.0
EPOCHS = 60
PATIENCE = 10


def weighted_mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    weights = 1.0 / (y_true + 0.1)
    return torch.mean(weights * (y_pred - y_true) ** 2)


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    X_all = np.load(PROCESSED_DIR / "X_train_sequences.npy")
    y_all = np.load(PROCESSED_DIR / "y_train_sequences.npy")
    seq_unit_ids = np.load(PROCESSED_DIR / "train_sequence_unit_ids.npy")

    input_size = X_all.shape[-1]

    unique_units = np.unique(seq_unit_ids)
    train_units, val_units = train_test_split(unique_units, test_size=0.2, random_state=42)

    train_mask = np.isin(seq_unit_ids, train_units)
    val_mask = np.isin(seq_unit_ids, val_units)

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]

    print(f"Train split: {X_train.shape} | {y_train.shape}")
    print(f"Val split  : {X_val.shape} | {y_val.shape}")

    y_train_norm = y_train / RUL_CAP
    y_val_norm = y_val / RUL_CAP

    # Mandatory sanity check: overfit 50 samples.
    tiny_n = min(50, len(X_train))
    X_small = torch.tensor(X_train[:tiny_n], dtype=torch.float32, device=DEVICE)
    y_small = torch.tensor(y_train_norm[:tiny_n], dtype=torch.float32, device=DEVICE)

    sanity_model = LSTMRULModel(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.0).to(DEVICE)
    sanity_opt = torch.optim.Adam(sanity_model.parameters(), lr=0.005)
    sanity_criterion = nn.MSELoss()

    for epoch in range(1, 301):
        sanity_model.train()
        sanity_opt.zero_grad()
        pred = sanity_model(X_small)
        loss = sanity_criterion(pred, y_small)
        loss.backward()
        sanity_opt.step()

        if epoch in (1, 75, 150, 225, 300):
            print(f"Sanity epoch {epoch:03d} | MSE: {loss.item():.6f}")

    with torch.no_grad():
        sanity_preds = sanity_model(X_small).cpu().numpy() * RUL_CAP

    print(f"Sanity final MSE: {loss.item():.6f}")
    print(f"Sanity prediction STD: {np.std(sanity_preds):.4f}")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train_norm, dtype=torch.float32)),
        batch_size=128,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val_norm, dtype=torch.float32)),
        batch_size=128,
        shuffle=False,
    )

    model = LSTMRULModel(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_rmse_hist, val_rmse_hist = [], []
    best_val_rmse = float("inf")
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_preds_epoch = []
        train_true_epoch = []

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = weighted_mse(preds, yb)
            loss.backward()
            optimizer.step()

            train_preds_epoch.append(preds.detach().cpu().numpy() * RUL_CAP)
            train_true_epoch.append(yb.detach().cpu().numpy() * RUL_CAP)

        train_preds_epoch = np.concatenate(train_preds_epoch)
        train_true_epoch = np.concatenate(train_true_epoch)
        train_rmse = float(np.sqrt(mean_squared_error(train_true_epoch, train_preds_epoch)))

        model.eval()
        val_preds_epoch = []
        val_true_epoch = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb)
                val_preds_epoch.append(preds.cpu().numpy() * RUL_CAP)
                val_true_epoch.append(yb.cpu().numpy() * RUL_CAP)

        val_preds_epoch = np.concatenate(val_preds_epoch)
        val_true_epoch = np.concatenate(val_true_epoch)
        val_rmse = float(np.sqrt(mean_squared_error(val_true_epoch, val_preds_epoch)))

        train_rmse_hist.append(train_rmse)
        val_rmse_hist.append(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            no_improve = 0
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")
        else:
            no_improve += 1

        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:02d} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Best validation RMSE: {best_val_rmse:.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(train_rmse_hist, label="Train RMSE", color="#2563eb")
    plt.plot(val_rmse_hist, label="Val RMSE", color="#f59e0b")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("FD004 Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "training_loss_curve.png", dpi=150)

    best_model = LSTMRULModel(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.2).to(DEVICE)
    best_model.load_state_dict(torch.load(MODEL_DIR / "best_model.pth", map_location=DEVICE))
    best_model.eval()

    with torch.no_grad():
        val_preds = best_model(torch.tensor(X_val, dtype=torch.float32, device=DEVICE)).cpu().numpy() * RUL_CAP

    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
    print(f"Validation RMSE (real scale): {val_rmse:.4f}")
    print(f"Prediction STD: {float(np.std(val_preds)):.4f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_val, val_preds, alpha=0.45, s=12, color="#f59e0b")
    lim = max(float(np.max(y_val)), float(np.max(val_preds)))
    plt.plot([0, lim], [0, lim], "r--", linewidth=1.4)
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.title("Validation: Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "predicted_vs_actual_rul.png", dpi=150)


if __name__ == "__main__":
    main()
