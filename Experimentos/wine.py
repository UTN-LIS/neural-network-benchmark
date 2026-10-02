import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. Configuración
# ============================================================

SEED = 42

LEARNING_RATE = 0.001
BATCH_SIZE = 32 #Todavia no se esta usando ya que no se esta procesando por lotes, se procesa todo de una vez.
EPOCHS = 100 #Recorrido completo por el conjunto de entrenamiento


# ============================================================
# 2. Reproducibilidad
# ============================================================

np.random.seed(SEED) #Controla aleatoridad utilizada por numpy
torch.manual_seed(SEED) #Controla aleatoridad utilizada por pytorch


# ============================================================
# 3. Carga del dataset
# ============================================================

wine = load_wine() #Carga el dataset wine, con los elementos: data, target, feature_names, target_names,etc

X = wine.data
y = wine.target

print("Dataset Wine")
print(f"Muestras: {X.shape[0]}")
print(f"Características: {X.shape[1]}")
print(f"Clases: {len(np.unique(y))} \n")


# ============================================================
# 4. División train / test
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
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
            nn.Linear(13, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
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
            nonlinearity="relu"
        )

        nn.init.zeros_(layer.bias)


# ============================================================
# 9. Función de pérdida y optimizador
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
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

    predictions = torch.argmax(outputs, dim=1)

    accuracy = (
        predictions == y_test
    ).float().mean().item()


# ============================================================
# 12. Resultados
# ============================================================

num_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("\n==============================")
print("RESULTADOS")
print("==============================")

print(f"Accuracy: {accuracy:.4f}")
print(f"Tiempo de entrenamiento: {training_time:.6f} s")
print(f"Parámetros del modelo: {num_parameters}")