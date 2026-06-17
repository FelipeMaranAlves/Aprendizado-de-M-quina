import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, roc_curve
from utils import documentar, caminho_imagem
from pipeline import carregar, normalizar
from sessions.anomaly_detection.autoencoder3camadas import _Autoencoder, _scores_anomalia

RAND_STATE = 2


def _split_benignos_treino(scaler_cls):
    df = pd.read_csv("data/PDF_All_feature_Clean.csv")
    df.drop(columns='file_path', inplace=True)

    df_Bening = df.query("label == 0")

    X_train, _ = train_test_split(df_Bening, test_size=0.2, random_state=RAND_STATE)
    _, Xy_tmp = train_test_split(df, test_size=0.2, random_state=RAND_STATE)
    Xy_val, Xy_test = train_test_split(Xy_tmp, test_size=0.5, random_state=RAND_STATE)

    X_train = X_train.drop(columns=['label'])
    y_test = Xy_test['label'].values
    X_test = Xy_test.drop(columns=['label'])

    scaler = scaler_cls()
    scaler.fit(X_train)
    norm_X_train = scaler.transform(X_train)
    norm_X_test = scaler.transform(X_test)

    return norm_X_train, norm_X_test, y_test


def _scores_kmeans():
    X_train, X_test, y_test = _split_benignos_treino(StandardScaler)
    model = KMeans(n_clusters=2, random_state=RAND_STATE, n_init=10)
    model.fit(X_train)
    scores_test = model.transform(X_test).min(axis=1)
    return scores_test, y_test


def _scores_dbscan():
    X_train, X_test, y_test = _split_benignos_treino(StandardScaler)
    EPS = 3.0
    MIN_SAMPLES = 5
    dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, algorithm='ball_tree', n_jobs=-1)
    dbscan.fit(X_train)
    X_core = X_train[dbscan.core_sample_indices_]
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    nn.fit(X_core)
    scores_test = nn.kneighbors(X_test)[0].flatten()
    return scores_test, y_test


def _scores_isolation_forest():
    df = carregar()
    Xy_train, Xy_tmp = train_test_split(df, test_size=0.2, random_state=RAND_STATE)
    _, Xy_test = train_test_split(Xy_tmp, test_size=0.5, random_state=RAND_STATE)

    X_train = Xy_train.drop(columns=['label'])
    y_test = Xy_test['label'].values
    X_test = Xy_test.drop(columns=['label'])

    norm_X_train, _, norm_X_test = normalizar(X_train, X_train, X_test)

    model = IsolationForest(
        n_estimators=15, max_samples=0.89, max_features=0.1,
        contamination=0.5, random_state=RAND_STATE,
    ).fit(norm_X_train)

    scores_test = model.decision_function(norm_X_test)
    return scores_test, y_test


def _scores_autoencoder():
    X_train, X_test, y_test = _split_benignos_treino(MinMaxScaler)
    in_features = X_train.shape[1]

    with open("models/best_ae_3.json") as f:
        info = json.load(f)

    model = _Autoencoder(in_features, 24, 20, 16, 0.2)
    model.load_state_dict(torch.load("models/checkpoint_ae_3_camadas24-20-16.pt", weights_only=True))
    model.eval()

    X_test_t = torch.FloatTensor(X_test)
    scores_test = _scores_anomalia(model, X_test_t)
    return scores_test, y_test


def rodar():
    modelos = {
        "K-Means (K=2)": _scores_kmeans,
        "DBSCAN (eps=3.0)": _scores_dbscan,
        "Isolation Forest": _scores_isolation_forest,
        "Autoencoder (3 camadas)": _scores_autoencoder,
    }

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Aleatório')

    doc_lines = ["Comparacao de curvas ROC - K-Means, DBSCAN, Isolation Forest e Autoencoder",
                 "(mesmo conjunto de teste para todos)\n"]

    for nome, fn in modelos.items():
        scores_test, y_test = fn()
        auc = roc_auc_score(y_test, scores_test)
        fpr, tpr, _ = roc_curve(y_test, scores_test)
        plt.plot(fpr, tpr, label=f"{nome} (AUC = {auc:.4f})")
        doc_lines.append(f"{nome}: AUC-ROC = {auc:.4f}")

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC - Comparação dos 4 detectores de anomalia (conjunto de teste)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(caminho_imagem("roc_comparacao_todos_modelos.png", "anomaly_detection"))
    plt.close()

    doc = "\n".join(doc_lines) + "\n"
    print(doc)
    documentar("ROC_Comparacao_Todos_Modelos", doc, categoria="anomaly_detection")


# ============================================================
#                            NOTAS
# ============================================================
#
# Por que os quatro modelos sao retreinados aqui em vez de reusar os
# scripts originais diretamente?
# Cada modelo guarda seu score de anomalia de um jeito diferente
# (distancia a centroide, distancia a core point, decision_function,
# erro de reconstrucao) e em escalas completamente diferentes — nao da'
# para comparar os VALORES dos scores entre si. A curva ROC, porem, e'
# invariante a escala (so' depende do ranking), por isso plotar as
# quatro curvas no mesmo grafico e' uma comparacao justa mesmo com
# scores em unidades diferentes. K-Means/DBSCAN/Isolation Forest sao
# baratos de retreinar (segundos); o Autoencoder usa o checkpoint ja
# salvo (models/checkpoint_ae_3_camadas24-20-16.pt) em vez de retreinar,
# pois treinar uma rede de novo so' para extrair o score seria
# desperdicio e poderia divergir levemente do modelo já validado.
#
# Por que o conjunto de teste e' o mesmo para os quatro, mesmo com
# preprocessamentos diferentes (StandardScaler vs MinMaxScaler) e
# treinos diferentes (so' benignos vs dados mistos para o Isolation
# Forest)?
# O y_test em si vem sempre da mesma linha do dataframe original (mesmo
# random_state=2 nos mesmos splits), entao os PDFs no conjunto de teste
# sao identicos entre os quatro experimentos — só a normalizacao/entrada
# do modelo muda, nao quais amostras sao avaliadas.
#
# Por que Isolation Forest usa decision_function sem inverter o sinal?
# Ver a explicacao detalhada em isolation_forest_teste.py: nesse
# protocolo de treino (dados mistos), o algoritmo aprendeu a polaridade
# invertida (malicioso = mais "normal"/denso). Sem essa correcao, a
# curva ROC do Isolation Forest apareceria abaixo da diagonal.
