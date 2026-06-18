import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from utils import documentar, caminho_imagem
from utilsDoProfessor import get_overall_metrics
from pipeline import tratar_outliers_treino


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

    X_train, X_val, X_test = tratar_outliers_treino(X_train, X_val, X_test)

    std_scaler = StandardScaler()
    std_scaler.fit(X_train)
    nomr_X_train = std_scaler.transform(X_train)
    norm_X_val = std_scaler.transform(X_val)
    norm_X_test = std_scaler.transform(X_test)

    return nomr_X_train, norm_X_val, y_val, norm_X_test, y_test


def _gerar_kdistance_plot(X_train, min_samples):
    nn = NearestNeighbors(n_neighbors=min_samples, algorithm='ball_tree')
    nn.fit(X_train)
    distances, _ = nn.kneighbors(X_train)
    k_distances = np.sort(distances[:, -1])[::-1]

    plt.figure()
    plt.plot(k_distances)
    plt.xlabel("Pontos (ordenados por distância decrescente)")
    plt.ylabel(f"Distância ao {min_samples}º vizinho mais próximo")
    plt.title(f"K-distance plot para estimar eps (min_samples={min_samples})")
    plt.tight_layout()
    plt.savefig(caminho_imagem("dbscan_kdistance.png", "anomaly_detection"))
    plt.close()
    print("K-distance plot salvo: Images/anomaly_detection/dbscan_kdistance.png")


def rodar():
    X_train, X_val, y_val, X_test, y_test = _preparar_dados()

    MIN_SAMPLES = 5
    EPS_VALUES = [0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]

    _gerar_kdistance_plot(X_train, MIN_SAMPLES)

    resultados = []
    best_auc_val = -1.0
    best_eps = None
    best_nn = None

    for eps in EPS_VALUES:
        dbscan = DBSCAN(eps=eps, min_samples=MIN_SAMPLES, algorithm='ball_tree', n_jobs=-1)
        dbscan.fit(X_train)

        n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
        n_noise = int((dbscan.labels_ == -1).sum())

        if len(dbscan.core_sample_indices_) == 0:
            print(f"eps={eps}: sem pontos núcleo, ignorando.")
            continue

        X_core = X_train[dbscan.core_sample_indices_]
        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        nn.fit(X_core)

        dist_val, _ = nn.kneighbors(X_val)
        scores_val = dist_val.flatten()

        try:
            auc_val = roc_auc_score(y_val, scores_val)
        except Exception:
            auc_val = 0.0

        print(f"eps={eps} | clusters={n_clusters} | ruido_treino={n_noise} | AUC_val={auc_val:.4f}")

        resultados.append({
            "eps": eps,
            "n_clusters": n_clusters,
            "n_noise_train": n_noise,
            "auc_val": auc_val,
            "nn": nn,
        })

        if auc_val > best_auc_val:
            best_auc_val = auc_val
            best_eps = eps
            best_nn = nn

    if best_nn is None:
        print("Nenhum modelo DBSCAN válido encontrado.")
        return {}

    best_result = next(r for r in resultados if r["eps"] == best_eps)
    scores_val_best = best_result["nn"].kneighbors(X_val)[0].flatten()

    thresholds = np.percentile(scores_val_best, np.arange(10, 95, 5))
    best_f1, best_threshold = 0.0, thresholds[0]

    for t in thresholds:
        preds = (scores_val_best > t).astype(int)
        if preds.sum() == 0 or preds.sum() == len(preds):
            continue
        m = get_overall_metrics(y_val, preds)
        if m['f1-score'] > best_f1:
            best_f1 = m['f1-score']
            best_threshold = t

    dist_test, _ = best_nn.kneighbors(X_test)
    scores_test = dist_test.flatten()
    preds_test = (scores_test > best_threshold).astype(int)
    m_test = get_overall_metrics(y_test, preds_test)
    auc_test = roc_auc_score(y_test, scores_test)

    doc = (
        f"DBSCAN Anomaly Detection (min_samples={MIN_SAMPLES})\n"
        "\nExperimentos por eps:\n"
    )
    for r in resultados:
        doc += (
            f"  eps={r['eps']} | clusters={r['n_clusters']} | "
            f"ruido_treino={r['n_noise_train']} | AUC_val={r['auc_val']:.4f}\n"
        )
    doc += (
        f"\nMelhor eps: {best_eps} (AUC_val={best_auc_val:.4f})\n"
        f"Threshold (val, max F1): {best_threshold:.4f}\n"
        f"\nMetricas no conjunto de teste (eps={best_eps}):\n"
        f"  Accuracy:  {m_test['acc']:.4f}\n"
        f"  Precision: {m_test['precision']:.4f}\n"
        f"  Recall:    {m_test['tpr']:.4f}\n"
        f"  F1-Score:  {m_test['f1-score']:.4f}\n"
        f"  AUC-ROC:   {auc_test:.4f}\n"
    )
    print(doc)
    documentar("AnomalyDetection_DBSCAN", doc, categoria="anomaly_detection")

    fpr, tpr, _ = roc_curve(y_test, scores_test)
    plt.figure()
    plt.plot(fpr, tpr, label=f'DBSCAN eps={best_eps} (AUC = {auc_test:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Aleatório')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC - DBSCAN Anomaly Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(caminho_imagem("anomaly_dbscan_roc.png", "anomaly_detection"))
    plt.close()

    epss = [r["eps"] for r in resultados]
    aucs = [r["auc_val"] for r in resultados]
    plt.figure()
    plt.plot(epss, aucs, marker='o')
    plt.xlabel("eps")
    plt.ylabel("AUC-ROC (validação)")
    plt.title(f"DBSCAN - AUC-ROC por eps (min_samples={MIN_SAMPLES})")
    plt.tight_layout()
    plt.savefig(caminho_imagem("dbscan_auc_por_eps.png", "anomaly_detection"))
    plt.close()

    return {
        "best_eps": best_eps,
        "auc": auc_test,
        "metrics_test": m_test,
        "scores_test": scores_test,
        "y_test": y_test,
    }


# ============================================================
#                          NOTAS 
# ============================================================
#
# O DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
# identifica regiões densas como clusters e marca automaticamente como
# ruído (-1) os pontos que não pertencem a nenhum cluster. Essa propriedade
# é naturalmente útil para detecção de anomalias: pontos no "ruído" são
# candidatos a ser maliciosos.
#
# Por que DBSCAN não tem .predict()?
# Diferente do K-Means, o DBSCAN não aprende um modelo paramétrico durante
# o treinamento. Ele calcula os clusters com base na densidade dos dados
# de treino. Para classificar novos pontos, precisamos de uma estratégia
# alternativa. A escolha foi extrair os "core points" (pontos de núcleo,
# que definem as regiões densas de PDFs benignos) e usar NearestNeighbors
# para medir a distância de cada ponto de teste ao core point mais próximo.
# Se essa distância for grande, o ponto está fora das regiões densas
# normais e portanto é suspeito.
#
# Por que o K-distance plot?
# Antes de definir eps, é preciso entender a distribuição de distâncias
# nos dados. O K-distance plot mostra, para cada ponto do treino, a distância
# ao seu k-ésimo vizinho mais próximo (k = min_samples). A "curva do cotovelo"
# nesse gráfico sugere onde eps deveria estar: abaixo do cotovelo, um eps
# pequeno demais cria ruído demais; acima, um eps grande demais funde todos
# os clusters em um só. O gráfico foi salvo para inspeção visual.
#
# Por que min_samples=5?
# É o padrão mais usado na literatura e funciona bem em datasets de tamanho
# médio (~7.400 amostras de treino). Valores maiores tornariam o DBSCAN mais
# conservador, exigindo mais vizinhos para um ponto ser considerado "core",
# reduzindo os clusters e aumentando o ruído, o que pode prejudicar a
# detecção de anomalias verdadeiras.
#
# Por que testar vários eps?
# O eps controla o raio de vizinhança e muda completamente a estrutura dos
# clusters. Testamos um range de valores e escolhemos o que maximiza o
# AUC-ROC na validação. Isso é análogo ao experimento de contamination
# feito no Isolation Forest: uma busca sistemática de hiperparâmetros.
#
# Por que algorithm='ball_tree'?
# Com 33 features, estruturas de dados como ball_tree são mais eficientes
# do que a busca por força bruta (brute). O kd_tree não funciona bem em
# dimensões altas, mas ball_tree ainda oferece bom desempenho aqui.
