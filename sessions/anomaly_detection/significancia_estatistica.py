import itertools
import json
import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from utils import documentar
from pipeline import carregar, normalizar
from sessions.anomaly_detection.autoencoder3camadas import _Autoencoder, _scores_anomalia
from sessions.anomaly_detection.roc_comparacao_final import _split_benignos_treino, RAND_STATE
from sklearn.model_selection import train_test_split


def _preds_kmeans():
    X_train, X_test, y_test = _split_benignos_treino(StandardScaler)
    model = KMeans(n_clusters=2, random_state=RAND_STATE, n_init=10)
    model.fit(X_train)
    scores_test = model.transform(X_test).min(axis=1)
    THRESHOLD = 10.5391  # melhor threshold por F0.5 (kmeans_anomaly_fbeta.py)
    return (scores_test > THRESHOLD).astype(int), y_test


def _preds_dbscan():
    X_train, X_test, y_test = _split_benignos_treino(StandardScaler)
    dbscan = DBSCAN(eps=3.0, min_samples=5, algorithm='ball_tree', n_jobs=-1)
    dbscan.fit(X_train)
    X_core = X_train[dbscan.core_sample_indices_]
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    nn.fit(X_core)
    scores_test = nn.kneighbors(X_test)[0].flatten()
    THRESHOLD = 1.7662  # melhor threshold por F0.5 (dbscan_anomaly_fbeta.py)
    return (scores_test > THRESHOLD).astype(int), y_test


def _preds_isolation_forest():
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

    preds = model.predict(norm_X_test)
    preds[preds == 1] = 1
    preds[preds == -1] = 0
    return preds, y_test


def _preds_autoencoder():
    X_train, X_test, y_test = _split_benignos_treino(MinMaxScaler)
    in_features = X_train.shape[1]

    with open("models/best_ae_3.json") as f:
        info = json.load(f)
    threshold = info["threshold"]

    model = _Autoencoder(in_features, 24, 20, 16, 0.2)
    model.load_state_dict(torch.load("models/checkpoint_ae_3_camadas24-20-16.pt", weights_only=True))
    model.eval()

    X_test_t = torch.FloatTensor(X_test)
    scores_test = _scores_anomalia(model, X_test_t)
    return (scores_test > threshold).astype(int), y_test


def _mcnemar(correct_a, correct_b):
    """McNemar exato (teste binomial), recomendado para n pequeno de discordancias."""
    so_a_certo = int(np.sum(correct_a & ~correct_b))
    so_b_certo = int(np.sum(~correct_a & correct_b))
    n = so_a_certo + so_b_certo
    if n == 0:
        return so_a_certo, so_b_certo, 1.0
    p_value = binomtest(min(so_a_certo, so_b_certo), n, 0.5).pvalue
    return so_a_certo, so_b_certo, p_value


def rodar():
    modelos = {
        "K-Means": _preds_kmeans,
        "DBSCAN": _preds_dbscan,
        "Isolation Forest": _preds_isolation_forest,
        "Autoencoder": _preds_autoencoder,
    }

    acertos = {}
    y_test_ref = None
    for nome, fn in modelos.items():
        preds, y_test = fn()
        if y_test_ref is None:
            y_test_ref = y_test
        else:
            assert np.array_equal(y_test, y_test_ref), f"y_test de {nome} difere dos demais"
        acertos[nome] = (preds == y_test)

    doc_lines = [
        "Teste de significancia estatistica (McNemar exato) entre os modelos",
        f"Tamanho do conjunto de teste: {len(y_test_ref)} amostras",
        "H0: os dois modelos tem a mesma taxa de erro (discordancias divididas 50/50 entre os dois sentidos).",
        "Comparando, par a par, apenas as amostras em que os dois modelos discordam entre si.\n",
    ]

    ALPHA = 0.05
    for nome_a, nome_b in itertools.combinations(modelos.keys(), 2):
        so_a, so_b, p = _mcnemar(acertos[nome_a], acertos[nome_b])
        significativo = "SIM" if p < ALPHA else "nao"
        doc_lines.append(
            f"{nome_a} vs {nome_b}\n"
            f"  Só {nome_a} acerta: {so_a} | Só {nome_b} acerta: {so_b}\n"
            f"  p-valor (McNemar exato): {p:.4g} | diferença significativa (alpha=0.05)? {significativo}\n"
        )

    doc = "\n".join(doc_lines) + "\n"
    print(doc)
    documentar("Significancia_Estatistica_McNemar", doc, categoria="anomaly_detection")
    return acertos


# ============================================================
#                            NOTAS
# ============================================================
#
# Por que McNemar e nao um teste-t entre as metricas?
# Accuracy/F1/F0.5 sao numeros unicos por modelo (nao ha' uma amostra de
# "repeticoes" para comparar médias). McNemar e' o teste padrao para
# comparar dois classificadores binarios AVALIADOS NO MESMO conjunto de
# teste: ele olha apenas para as amostras em que os dois modelos
# discordam entre si e testa se essa discordancia e' simetrica (nenhum
# dos dois e' sistematicamente melhor) ou nao.
#
# Por que a versao exata (binomial) em vez da aproximacao qui-quadrado?
# A aproximacao qui-quadrado com correcao de continuidade e' razoavel
# quando o numero de discordancias e' grande; a versao exata (teste
# binomial bilateral com p=0.5) e' válida em qualquer tamanho de amostra
# e evita depender de uma aproximacao assintotica. Como nao precisamos
# de velocidade aqui, preferimos a versao exata.
#
# Por que usar os thresholds escolhidos por F0.5 (e nao por F1) para
# gerar as predicoes binarias usadas no teste?
# Para manter consistencia com a comparacao final do projeto
# (Comparacao_Final_Modelos.txt), que usa F0.5 como criterio de
# referencia em todos os modelos.
#
# Limitacao: este teste compara desempenho em UM unico split de
# treino/validacao/teste (random_state=2). Ele responde "esses dois
# modelos discordam de forma nao-aleatoria nesta amostra de teste?", mas
# nao substitui uma validacao cruzada com múltiplos splits, que daria
# uma medida de variancia entre execucoes. Isso e' apontado como
# limitacao na secao de Conclusao do relatorio.
