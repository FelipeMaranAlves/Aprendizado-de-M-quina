import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, fbeta_score
from utils import documentar, caminho_imagem
from utilsDoProfessor import get_overall_metrics


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

    std_scaler = StandardScaler()
    std_scaler.fit(X_train)
    nomr_X_train = std_scaler.transform(X_train)
    norm_X_val = std_scaler.transform(X_val)
    norm_X_test = std_scaler.transform(X_test)

    return nomr_X_train, norm_X_val, y_val, norm_X_test, y_test


def _beta_label(beta):
    return f"{beta:g}"


def _melhor_threshold_fbeta(y_val, scores_val, thresholds, beta):
    best_score, best_threshold = -1.0, thresholds[0]
    for t in thresholds:
        preds = (scores_val > t).astype(int)
        if preds.sum() == 0 or preds.sum() == len(preds):
            continue
        score = fbeta_score(y_val, preds, beta=beta, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = t
    return best_threshold, best_score


def rodar():
    RAND_STATE = 2
    BETAS = [0.5, 1.0, 2.0]

    X_train, X_val, y_val, X_test, y_test = _preparar_dados()

    model = KMeans(n_clusters=2, random_state=RAND_STATE, n_init=10)
    model.fit(X_train)

    def _scores(X):
        return model.transform(X).min(axis=1)

    scores_val = _scores(X_val)
    scores_test = _scores(X_test)
    auc = roc_auc_score(y_test, scores_test)

    thresholds = np.percentile(scores_val, np.arange(10, 95, 5))

    resultados = {}
    for beta in BETAS:
        best_threshold, best_score_val = _melhor_threshold_fbeta(y_val, scores_val, thresholds, beta)

        preds_test = (scores_test > best_threshold).astype(int)
        m_test = get_overall_metrics(y_test, preds_test)
        fbeta_test = fbeta_score(y_test, preds_test, beta=beta, zero_division=0)

        resultados[beta] = {
            "threshold": best_threshold,
            "fbeta_val": best_score_val,
            "metrics_test": m_test,
            "fbeta_test": fbeta_test,
        }

    doc_lines = [
        "K-Means Anomaly Detection (K=2) - Distancia ao Centroide",
        "Selecao do threshold por F-beta (em vez de F1) na validacao\n",
        f"AUC-ROC no teste (nao depende do threshold): {auc:.4f}\n",
        "Comparacao por beta (beta<1 prioriza Precision, beta=1 equivale ao F1 original, beta>1 prioriza Recall):",
    ]
    for beta in BETAS:
        r = resultados[beta]
        m = r["metrics_test"]
        b = _beta_label(beta)
        doc_lines.append(
            f"\nbeta={b} | threshold(val)={r['threshold']:.4f} | F{b} no val={r['fbeta_val']:.4f}\n"
            f"  Accuracy:    {m['acc']:.4f}\n"
            f"  Precision:   {m['precision']:.4f}\n"
            f"  Recall:      {m['tpr']:.4f}\n"
            f"  F1-Score:    {m['f1-score']:.4f}\n"
            f"  F{b}-Score:   {r['fbeta_test']:.4f}  (metrica usada para escolher o threshold)"
        )
    doc = "\n".join(doc_lines) + "\n"
    print(doc)
    documentar("AnomalyDetection_KMeans_Fbeta", doc, categoria="anomaly_detection")

    precisions = [resultados[b]["metrics_test"]["precision"] for b in BETAS]
    recalls = [resultados[b]["metrics_test"]["tpr"] for b in BETAS]
    f1s = [resultados[b]["metrics_test"]["f1-score"] for b in BETAS]

    x = np.arange(len(BETAS))
    width = 0.25
    plt.figure()
    plt.bar(x - width, precisions, width, label='Precision')
    plt.bar(x, recalls, width, label='Recall')
    plt.bar(x + width, f1s, width, label='F1')
    plt.xticks(x, [f"beta={_beta_label(b)}" for b in BETAS])
    plt.ylabel("Score no teste")
    plt.title("K-Means - Precision/Recall/F1 por beta usado na escolha do threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(caminho_imagem("kmeans_fbeta_comparacao.png", "anomaly_detection"))
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, scores_test)
    plt.figure()
    plt.plot(fpr, tpr, label=f'K-Means K=2 (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Aleatório')
    for beta in BETAS:
        m = resultados[beta]["metrics_test"]
        plt.scatter(m['fpr'], m['tpr'], zorder=5, label=f"beta={_beta_label(beta)} (FPR={m['fpr']:.2f}, TPR={m['tpr']:.2f})")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC - K-Means Anomaly Detection (pontos de operação por beta)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(caminho_imagem("kmeans_fbeta_roc.png", "anomaly_detection"))
    plt.close()

    return {
        "model": model,
        "auc": auc,
        "resultados_por_beta": resultados,
    }


# ============================================================
#                            NOTAS
# ============================================================
#
# Por que F-beta em vez de F1 para escolher o threshold?
# O F1 (que já era usado em kmeans_anomaly.py) trata Precision e Recall
# com o mesmo peso. O F-beta generaliza essa ideia com um parametro beta
# que controla esse peso: F_beta = (1+beta^2) * P*R / (beta^2*P + R).
# Com beta=0.5, o Recall pesa 4x menos que a Precision na média
# harmônica, ou seja, o threshold escolhido passa a favorecer modelos
# com menos falsos positivos. Isso é relevante no contexto de detecção
# de anexos maliciosos: cada PDF benigno classificado como malicioso é
# um alerta falso que consome tempo de um analista de segurança (ou
# bloqueia um e-mail legítimo). Um F0.5 alto exige que o modelo seja
# mais "conservador" ao apontar uma anomalia.
#
# Por que comparar beta=0.5, 1.0 e 2.0 em vez de só usar 0.5?
# O enunciado pede para testar diferentes hiperparâmetros e avaliar a
# robustez do método. Variar beta no próprio critério de seleção do
# threshold é uma forma direta de fazer isso: beta=1.0 reproduz
# exatamente o experimento original (kmeans_anomaly.py, mesmo
# random_state e mesmos splits), servindo de checagem de consistência;
# beta=0.5 favorece Precision; beta=2.0 favorece Recall (o oposto,
# relevante quando o custo de deixar passar um ataque é maior que o
# custo de um falso alarme). Comparar os três no mesmo modelo treinado
# isola o efeito da escolha de threshold, sem misturar com mudanças no
# modelo em si.
#
# Por que o modelo (K-Means K=2) não muda entre os betas?
# O F-beta só é usado para decidir ONDE cortar o score continuo de
# anomalia (distância ao centroide) e transformá-lo em 0/1. O modelo
# em si — e portanto o AUC-ROC, que é calculado sobre o score contínuo,
# sem threshold — é o mesmo para qualquer beta. Por isso o AUC aparece
# uma única vez no relatório, fora da tabela por beta.
#
# Como comparar com o resultado já existente (AnomalyDetection_KMeans.txt)?
# A linha beta=1 deste experimento deve reproduzir os números já
# documentados em AnomalyDetection_KMeans.txt (mesmo modelo, mesmo
# random_state=2, mesmos splits de treino/validação/teste). As linhas
# beta=0.5 e beta=2 mostram o quanto Precision e Recall se deslocam ao
# mudar o critério de corte, evidenciando o trade-off que motiva a
# escolha de F0.5 em um cenário de segurança onde falsos positivos têm
# custo operacional.
