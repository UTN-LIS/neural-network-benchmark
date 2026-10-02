import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# 1. Configuración
# ============================================================

SEED = 42

LEARNING_RATE = 0.001
EPOCHS = 100


# ============================================================
# 2. Reproducibilidad
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# 3. Carga del dataset
# ============================================================

URL = (
    "https://archive.ics.uci.edu/"
    "ml/machine-learning-databases/00267/"
    "data_banknote_authentication.txt"
)

columns = [
    "variance",
    "skewness",
    "curtosis",
    "entropy",
    "target",
]

df = pd.read_csv(
    URL,
    header=None,
    names=columns,
)

X = df.drop(columns="target").values
y = df["target"].values


print("Dataset Banknote Authentication")
print(f"Muestras: {X.shape[0]}")
print(f"Características: {X.shape[1]}")
print(f"Clases: {len(np.unique(y))}")


# ============================================================
# 4. División train / test
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y,
)


# ============================================================
# 5. Normalización
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ============================================================
# 6. Conversión a tensores
# ============================================================

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# ============================================================
# 7. Definición del MLP
# ============================================================

class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.network(x)


model = MLP()


# ============================================================
# 8. Inicialización He
# ============================================================

for layer in model.modules():

    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(
            layer.weight,
            nonlinearity="relu",
        )

        nn.init.zeros_(layer.bias)


# ============================================================
# 9. Función de pérdida y optimizador
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# 10. Entrenamiento
# ============================================================

start_time = time.perf_counter()

for epoch in range(EPOCHS):

    model.train()

    optimizer.zero_grad()

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(
            f"Época {epoch + 1:3d}/{EPOCHS} "
            f"- Loss: {loss.item():.4f}"
        )

training_time = time.perf_counter() - start_time


# ============================================================
# 11. Evaluación
# ============================================================

model.eval()

with torch.no_grad():

    outputs = model(X_test)

    predictions = torch.argmax(
        outputs,
        dim=1,
    )


# Convertir tensores a NumPy
y_true = y_test.numpy()
y_pred = predictions.numpy()


# ============================================================
# 12. Métricas
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred,
)

balanced_accuracy = balanced_accuracy_score(
    y_true,
    y_pred,
)

precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

confusion = confusion_matrix(
    y_true,
    y_pred,
)

num_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)


# ============================================================
# 13. Resultados
# ============================================================

print("\n==============================")
print("RESULTADOS")
print("==============================")

print(f"Accuracy:           {accuracy:.4f}")
print(f"Balanced Accuracy:  {balanced_accuracy:.4f}")
print(f"Precision Macro:    {precision:.4f}")
print(f"Recall Macro:       {recall:.4f}")
print(f"F1 Macro:           {f1:.4f}")

print(f"\nLoss final:         {loss.item():.4f}")
print(f"Tiempo:             {training_time:.6f} s")
print(f"Parámetros:         {num_parameters}")

print("\nMatriz de confusión:")
print(confusion)