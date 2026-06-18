import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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


def rodar():
    RAND_STATE = 2

    X_train, X_val, y_val, X_test, y_test = _preparar_dados()

    model = KMeans(n_clusters=2, random_state=RAND_STATE, n_init=10)
    model.fit(X_train)

    def _scores(X):
        return model.transform(X).min(axis=1)

    scores_val = _scores(X_val)
    scores_test = _scores(X_test)

    thresholds = np.percentile(scores_val, np.arange(10, 95, 5))
    best_f1, best_threshold = 0.0, thresholds[0]

    for t in thresholds:
        preds = (scores_val > t).astype(int)
        if preds.sum() == 0 or preds.sum() == len(preds):
            continue
        m = get_overall_metrics(y_val, preds)
        if m['f1-score'] > best_f1:
            best_f1 = m['f1-score']
            best_threshold = t

    preds_test = (scores_test > best_threshold).astype(int)
    m_test = get_overall_metrics(y_test, preds_test)
    auc = roc_auc_score(y_test, scores_test)

    doc = (
        "K-Means Anomaly Detection (K=2) - Distancia ao Centroide\n"
        f"Melhor threshold (val, max F1): {best_threshold:.4f}\n"
        f"F1 no val: {best_f1:.4f}\n"
        "\nMetricas no conjunto de teste:\n"
        f"  Accuracy:  {m_test['acc']:.4f}\n"
        f"  Precision: {m_test['precision']:.4f}\n"
        f"  Recall:    {m_test['tpr']:.4f}\n"
        f"  F1-Score:  {m_test['f1-score']:.4f}\n"
        f"  AUC-ROC:   {auc:.4f}\n"
    )
    print(doc)
    documentar("AnomalyDetection_KMeans", doc, categoria="anomaly_detection")

    fpr, tpr, _ = roc_curve(y_test, scores_test)
    plt.figure()
    plt.plot(fpr, tpr, label=f'K-Means K=2 (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Aleatório')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC - K-Means Anomaly Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(caminho_imagem("anomaly_kmeans_roc.png", "anomaly_detection"))
    plt.close()

    return {
        "model": model,
        "threshold": best_threshold,
        "auc": auc,
        "metrics_test": m_test,
        "scores_test": scores_test,
        "y_test": y_test,
    }


# ============================================================
#                            NOTAS 
# ============================================================
#
# Ideia central: treinar o K-Means apenas em PDFs benignos e usar a
# distância ao centroide mais próximo como score de anomalia. Quanto
# mais um PDF se afasta dos padrões normais aprendidos, maior o score
# e maior a suspeita de que ele seja malicioso.
#
# Por que K=2?
# A análise feita em clustering.py mostrou que K=2 tem silhouette=0.901,
# muito acima de qualquer outro K testado (K=3 a K=10 ficaram em torno
# de 0.24). Isso indica que os PDFs benignos se agrupam naturalmente em
# dois clusters bem definidos. Usar K=2 garante que o modelo capture
# bem a estrutura "normal" dos dados sem superajustar.
#
# Por que distância ao centroide como score de anomalia?
# PDFs maliciosos tendem a usar features incomuns (js_count alto,
# openaction_count, jbig2decode_count, etc.) que diferem bastante dos
# PDFs benignos. Eles devem estar mais distantes dos centroides benignos.
# Usar a distância como score contínuo (em vez de classificação direta)
# permite calcular AUC-ROC, que é mais robusta em cenários desbalanceados
# e não depende de um threshold fixo.
#
# Por que o threshold é encontrado no conjunto de validação?
# Usar o conjunto de teste para calibrar o threshold causaria data leakage
# e produziria uma estimativa otimista e enganosa do desempenho real.
# O conjunto de validação serve justamente para essa calibragem, sem
# contaminar a estimativa final no teste.
#
# Por que varrer percentis (10% a 95%) para encontrar o threshold?
# A distribuição das distâncias não tem uma referência absoluta de onde
# cortar. Varrer percentis nos dá uma busca simples e eficiente que
# cobre bem o espaço de possíveis thresholds, e escolhemos o que
# maximiza o F1-score na validação.
