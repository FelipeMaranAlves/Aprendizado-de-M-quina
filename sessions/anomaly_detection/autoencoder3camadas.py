import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from utils import documentar
from utilsDoProfessor import get_overall_metrics
from sklearn.metrics import roc_curve
import json
import os

BEST_MODEL_INFO = "models/best_ae_3.json"
BEST_MODEL_PATH = "models/checkpoint_ae_3_camadas.pt"
def salvar_se_melhor(model, score_atual, threshold, auc_val, metrica="youden"):
    melhor_score = -np.inf

    if os.path.exists(BEST_MODEL_INFO):
        with open(BEST_MODEL_INFO, "r") as f:
            dados = json.load(f)
            melhor_score = dados["score"]

    if score_atual > melhor_score:
        torch.save(model.state_dict(), BEST_MODEL_PATH)

        with open(BEST_MODEL_INFO, "w") as f:
            json.dump(
                {
                    "score": float(score_atual),
                    "threshold": float(threshold),
                    "auc_val": float(auc_val),
                    "metrica": metrica
                },
                f,
                indent=4
            )

        print(
            "-------------------------------------------"
            f"Novo melhor modelo salvo "
            f"{melhor_score:.4f} -> {score_atual:.4f}"
            "-------------------------------------------"
        )
    else:
        print(
            "-------------------------------------------"
            f"Modelo descartado. "
            f"Melhor atual = {melhor_score:.4f}"
            "-------------------------------------------"
        )

def _preparar_dados():
    df = pd.read_csv("data/PDF_All_feature_Clean.csv")
    df.drop(columns='file_path', inplace=True)

    df_Bening = df.query("label == 0")

    RAND_STATE = 2
    X_train, _ = train_test_split(df_Bening, test_size=0.2, random_state=RAND_STATE)
    _, Xy_tmp = train_test_split(df, test_size=0.2, random_state=RAND_STATE)
    Xy_val, Xy_test = train_test_split(Xy_tmp, test_size=0.5, random_state=RAND_STATE)

    X_train = X_train.drop(columns=['label'])
    y_val = Xy_val['label'].values
    X_val = Xy_val.drop(columns=['label'])
    y_test = Xy_test['label'].values
    X_test = Xy_test.drop(columns=['label'])

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    norm_X_train = scaler.transform(X_train)
    norm_X_val = scaler.transform(X_val)
    norm_X_test = scaler.transform(X_test)

    return norm_X_train, norm_X_val, y_val, norm_X_test, y_test


class _EarlyStopping:
    def __init__(self, patience=5, delta=1e-4, path='checkpoint_ae_3_camadas.pt'):
        self.patience = patience
        self.delta = delta
        self.path = BEST_MODEL_PATH
        self.counter = 0
        self.early_stop = False
        self.val_min_loss = np.inf

    def __call__(self, val_loss, model):
        if val_loss < self.val_min_loss - self.delta:
            print(f'Val loss {self.val_min_loss:.5f} -> {val_loss:.5f}. Melhorou!')
            self.val_min_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping: {self.counter}/{self.patience} (val_loss={val_loss:.5f})')
            if self.counter >= self.patience:
                self.early_stop = True


class _Autoencoder(nn.Module):
    def __init__(self, in_features, dropout_rate=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(24, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(24, in_features),
            nn.BatchNorm1d(in_features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def _treinar(model, X_train_t, X_val_benign_t, criterion, optimizer,
             num_epochs, batch_size, patience, delta):
    es = _EarlyStopping(patience=patience, delta=delta)
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        ep_losses = []
        for i in range(0, len(X_train_t), batch_size):
            batch = X_train_t[i:i + batch_size]
            loss = criterion(model(batch), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_losses.append(loss.item())
        train_avg = float(np.mean(ep_losses))
        train_losses.append(train_avg)

        model.eval()
        with torch.no_grad():
            v_losses = []
            for i in range(0, len(X_val_benign_t), batch_size):
                batch = X_val_benign_t[i:i + batch_size]
                v_losses.append(criterion(model(batch), batch).item())
        val_avg = float(np.mean(v_losses))
        val_losses.append(val_avg)

        print(f'Época {epoch + 1}/{num_epochs} | train={train_avg:.5f} | val_benign={val_avg:.5f}')
        es(val_avg, model)
        if es.early_stop:
            print(f'Early stopping na época {epoch + 1}.')
            break

    model.load_state_dict(torch.load(es.path, weights_only=True))
    model.eval()
    return train_losses, val_losses


def _scores_anomalia(model, X_t):
    with torch.no_grad():
        recon = model(X_t)
        return torch.mean((X_t - recon) ** 2, dim=1).numpy() # MSELoss


def rodar():
    os.makedirs("Images", exist_ok=True)
    RAND_STATE = 2
    torch.manual_seed(RAND_STATE)
    np.random.seed(RAND_STATE)

    X_train, X_val, y_val, X_test, y_test = _preparar_dados()

    in_features = X_train.shape[1]

    X_train_t = torch.FloatTensor(X_train)
    X_val_benign_t = torch.FloatTensor(X_val[y_val == 0])
    X_val_t = torch.FloatTensor(X_val)
    X_test_t = torch.FloatTensor(X_test)

    model = _Autoencoder(in_features)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-3)

    NUM_EPOCHS = 100
    BATCH_SIZE = 256
    PATIENCE = 5
    DELTA = 1e-4
    train_losses, val_losses = _treinar(
        model, X_train_t, X_val_benign_t, criterion, optimizer,
        num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE, delta=DELTA,
    )

    scores_val = _scores_anomalia(model, X_val_t)
    fpr, tpr, thresholds = roc_curve(y_val, scores_val)

    youden_index = tpr - fpr

    best_idx = np.argmax(youden_index)


    best_threshold = thresholds[best_idx]
    best_youden = youden_index[best_idx]

    auc_val = roc_auc_score(y_val, scores_val)


    scores_test = _scores_anomalia(model, X_test_t)
    preds_test = (scores_test > best_threshold).astype(int)
    m_test = get_overall_metrics(y_test, preds_test)
    salvar_se_melhor(
    model,
    best_youden,
    best_threshold,
    auc_val,
    "youden"
    )
    auc = roc_auc_score(y_test, scores_test)

    doc = (
        f"Selecionando modelo usando Youden Index\n"
        f"Autoencoder Anomaly Detection\n"
        f"Parametros: num_epochs: {NUM_EPOCHS} batch_size: {BATCH_SIZE} patience: {PATIENCE}, delta: {DELTA},"
        f"Arquitetura: {in_features} -> 24 -> 16 -> 8 -> 16 -> 24 -> {in_features}\n"
        f"Épocas treinadas: {len(train_losses)}\n"
        f"Melhor threshold (val, max Youden_Index): {best_threshold:.6f}\n"
        f"Melhor Youden Index: {best_youden} "
        "\nMétricas no conjunto de teste:\n"
        f"  Accuracy:  {m_test['acc']:.4f}\n"
        f"  Precision: {m_test['precision']:.4f}\n"
        f"  Recall:    {m_test['tpr']:.4f}\n"
        f"  F1-Score:  {m_test['f1-score']:.4f}\n"
        f"  AUC-ROC:   {auc:.4f}\n"
    )
    print(doc)
    documentar("AnomalyDetection_Autoencoder_Experimento3Camadas", doc,"anomaly_detection")

    fpr, tpr, _ = roc_curve(y_test, scores_test)
    plt.figure()
    plt.plot(fpr, tpr, label=f'Autoencoder (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Aleatório')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC - Autoencoder Anomaly Detection 3")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Images/anomaly_autoencoder_3_camadas_roc.png")
    plt.close()

    plt.figure()
    epochs_axis = range(1, len(train_losses) + 1)
    plt.plot(epochs_axis, train_losses, label='Train Loss')
    plt.plot(epochs_axis, val_losses, label='Val Loss (benignos)')
    plt.xlabel("Época")
    plt.ylabel("MSE Loss")
    plt.title("Autoencoder - Curva de Aprendizado 3")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Images/autoencoder_3_camadas_loss.png")
    plt.close()

    return {
        "model": model,
        "threshold": best_threshold,
        "auc": auc,
        "metrics_test": m_test,
        "scores_test": scores_test,
        "y_test": y_test,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }