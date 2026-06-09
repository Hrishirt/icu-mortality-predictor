from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class MortalityMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MLPPredictor:
    """Sklearn-compatible wrapper around a trained PyTorch MLP."""

    def __init__(
        self,
        model: MortalityMLP,
        imputer: SimpleImputer,
        scaler: StandardScaler,
        device: torch.device,
    ):
        self.model = model
        self.imputer = imputer
        self.scaler = scaler
        self.device = device
        self.model.eval()

    def predict_proba(self, X) -> np.ndarray:
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            probs = self.model(tensor).cpu().numpy().ravel()
        return np.column_stack([1 - probs, probs])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_mlp(
    X_train,
    y_train,
    *,
    hidden_dim: int = 64,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    random_state: int = 42,
) -> MLPPredictor:
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(X_train)
    X_scaled = scaler.fit_transform(X_imputed)
    y_array = np.asarray(y_train, dtype=np.float32)

    dataset = TensorDataset(
        torch.tensor(X_scaled, dtype=torch.float32),
        torch.tensor(y_array, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MortalityMLP(input_dim=X_scaled.shape[1], hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    return MLPPredictor(model=model, imputer=imputer, scaler=scaler, device=device)
