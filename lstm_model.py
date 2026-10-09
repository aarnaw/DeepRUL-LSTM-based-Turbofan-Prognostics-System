import torch
import torch.nn as nn


class LSTMRULModel(nn.Module):
    """Simple LSTM regressor for RUL prediction."""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(self.dropout(last)).squeeze(-1)


def mc_dropout_predict(model: nn.Module, x: torch.Tensor, n_passes: int = 50):
    """Monte Carlo dropout inference returning mean and std predictions."""
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            preds.append(model(x).cpu())

    stacked = torch.stack(preds, dim=0)
    return stacked.mean(dim=0), stacked.std(dim=0)
