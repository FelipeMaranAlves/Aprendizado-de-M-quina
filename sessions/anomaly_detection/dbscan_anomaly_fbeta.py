import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, fbeta_score
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
    BETAS = [0.5, 1.0, 2.0]

    X_train, X_val, y_val, X_test, y_test = _preparar_dados()

    MIN_SAMPLES = 5
    EPS_VALUES = [0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]

    resultados_eps = []
    best_auc_val = -1.0
    best_eps = None
    best_nn = None

    for eps in EPS_VALUES:
        dbscan = DBSCAN(eps=eps, min_samples=MIN_SAMPLES, algorithm='ball_tree', n_jobs=-1)
        dbscan.fit(X_train)

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

        resultados_eps.append({"eps": eps, "auc_val": auc_val, "nn": nn})

        if auc_val > best_auc_val:
            best_auc_val = auc_val
            best_eps = eps
            best_nn = nn

    if best_nn is None:
        print("Nenhum modelo DBSCAN válido encontrado.")
        return {}

    scores_val_best = best_nn.kneighbors(X_val)[0].flatten()
    scores_test_best = best_nn.kneighbors(X_test)[0].flatten()
    auc_test = roc_auc_score(y_test, scores_test_best)

    thresholds = np.percentile(scores_val_best, np.arange(10, 95, 5))

    resultados = {}
    for beta in BETAS:
        best_threshold, best_score_val = _melhor_threshold_fbeta(y_val, scores_val_best, thresholds, beta)

        preds_test = (scores_test_best > best_threshold).astype(int)
        m_test = get_overall_metrics(y_test, preds_test)
        fbeta_test = fbeta_score(y_test, preds_test, beta=beta, zero_division=0)

        resultados[beta] = {
            "threshold": best_threshold,
            "fbeta_val": best_score_val,
            "metrics_test": m_test,
            "fbeta_test": fbeta_test,
        }

    doc_lines = [
        f"DBSCAN Anomaly Detection (min_samples={MIN_SAMPLES})",
        "Selecao do threshold por F-beta (em vez de F1) na validacao\n",
        f"Melhor eps escolhido por AUC-ROC na validacao (criterio nao muda com beta, "
        f"pois AUC e calculado sobre o score continuo, sem threshold): {best_eps} "
        f"(AUC_val={best_auc_val:.4f})",
        f"AUC-ROC no teste (nao depende do threshold): {auc_test:.4f}\n",
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
    documentar("AnomalyDetection_DBSCAN_Fbeta", doc, categoria="anomaly_detection")

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
    plt.title(f"DBSCAN (eps={best_eps}) - Precision/Recall/F1 por beta")
    plt.legend()
    plt.tight_layout()
    plt.savefig(caminho_imagem("dbscan_fbeta_comparacao.png", "anomaly_detection"))
    plt.close()

    return {
        "best_eps": best_eps,
        "auc": auc_test,
        "resultados_por_beta": resultados,
    }


# ============================================================
#                            NOTAS
# ============================================================
#
# Por que a escolha do eps continua sendo feita por AUC-ROC, e não por F-beta?
# AUC-ROC mede a qualidade do score de anomalia (a distância ao core
# point mais próximo) independente de qualquer threshold — ela resume
# o ranking entre PDFs benignos e maliciosos para todos os cortes
# possíveis. Já o F-beta depende de um threshold específico. Manter o
# eps escolhido por AUC garante que estamos comparando o EFEITO do
# threshold (via beta) isoladamente, sem confundir com mudanças no
# espaço de vizinhança definido pelo eps. Essa escolha replica o eps=3.0
# já obtido no experimento original (dbscan_anomaly.py).
#
# Por que F-beta em vez de F1 para escolher o threshold?
# Igual ao K-Means: o F1 original pesa Precision e Recall igualmente.
# Com beta=0.5, a Precision passa a valer 4x mais que o Recall na média
# harmônica (F_beta = (1+beta^2)*P*R / (beta^2*P+R)), produzindo um
# threshold mais conservador — menos PDFs benignos classificados
# incorretamente como maliciosos. Isso importa operacionalmente: cada
# falso positivo em um sistema de triagem de anexos é um e-mail
# legítimo bloqueado ou um alerta extra para um analista investigar.
#
# Por que testar beta=0.5, 1.0 e 2.0?
# Atende ao requisito de testar diferentes hiperparâmetros para avaliar
# a robustez do método. beta=1.0 reproduz o experimento original
# (dbscan_anomaly.py, mesmo eps, mesmos splits) e serve de checagem de
# consistência entre os dois scripts. beta=2.0 é o extremo oposto,
# priorizando Recall — relevante se o custo de não detectar um anexo
# malicioso (um ataque que passa) for considerado maior que o custo de
# um falso alarme.
#
# Como comparar com o resultado já existente (AnomalyDetection_DBSCAN.txt)?
# A linha beta=1 deste experimento deve reproduzir os números já
# documentados em AnomalyDetection_DBSCAN.txt para eps=3.0 (mesmo
# critério de seleção de eps, mesmos splits, mesmo random_state). As
# linhas beta=0.5 e beta=2 isolam o efeito de mudar o critério de corte
# do score de anomalia, mostrando o trade-off Precision/Recall que
# justifica a escolha de F0.5 em cenários de segurança sensíveis a
# falsos positivos.
