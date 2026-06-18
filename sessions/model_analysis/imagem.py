import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

dados = [
    {"f05": 0.9379, "num_epochs": 100, "batch_size": 128, "patience": 10, "delta": 1e-4},
    {"f05": 0.9379, "num_epochs": 100, "batch_size": 256, "patience": 10, "delta": 1e-4},
    {"f05": 0.9376, "num_epochs": 100, "batch_size": 512, "patience": 10, "delta": 1e-4},
    {"f05": 0.9379, "num_epochs": 500, "batch_size": 128, "patience": 10, "delta": 1e-4},
    {"f05": 0.9376, "num_epochs": 500, "batch_size": 128, "patience": 30, "delta": 1e-4},
    {"f05": 0.9376, "num_epochs": 500, "batch_size": 128, "patience": 30, "delta": 5e-5},
    {"f05": 0.9376, "num_epochs": 500, "batch_size": 64,  "patience": 10, "delta": 1e-4},
    {"f05": 0.9376, "num_epochs": 500, "batch_size": 64,  "patience": 30, "delta": 1e-4},
]

df = pd.DataFrame(dados)

# Transformar F0.5 em categorias para colorir as linhas
df["classe_f05"] = df["f05"].round(4).astype(str)

plt.figure(figsize=(12, 6))

parallel_coordinates(
    df[["classe_f05", "num_epochs", "batch_size", "patience", "delta"]],
    class_column="classe_f05"
)

plt.title("Parallel Coordinates Plot - Experimentos Autoencoder")
plt.ylabel("Valor")
plt.grid(True)
plt.tight_layout()
plt.savefig("parallel_coordinates.png", dpi=300)
plt.close()