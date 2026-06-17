import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, fbeta_score
from utils import documentar, caminho_imagem
from utilsDoProfessor import get_overall_metrics
from pipeline import carregar, normalizar


def rodar():
    df = carregar()

    RAND_STATE = 2
    Xy_train, Xy_tmp = train_test_split(df, test_size=0.2, random_state=RAND_STATE)
    Xy_val, Xy_test = train_test_split(Xy_tmp, test_size=0.5, random_state=RAND_STATE)

    y_train = Xy_train['label']
    X_train = Xy_train.drop(columns=['label'])
    y_val = Xy_val['label'].values
    X_val = Xy_val.drop(columns=['label'])
    y_test = Xy_test['label'].values
    X_test = Xy_test.drop(columns=['label'])

    norm_X_train, norm_X_val, norm_X_test = normalizar(X_train, X_val, X_test)

    # Melhor configuracao encontrada no grid search de isolation_forest.py,
    # usando F0.5 na validacao como criterio (ver IsolationForest2.txt):
    # contamination=0.5 | n_estimators=15 | max_samples=0.89 | max_features=0.1
    # F0.5 na validacao = 0.9014
    CONTAMINATION = 0.5
    N_ESTIMATORS = 15
    MAX_SAMPLES = 0.89
    MAX_FEATURES = 0.1

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        max_features=MAX_FEATURES,
        contamination=CONTAMINATION,
        random_state=RAND_STATE,
    ).fit(norm_X_train)

    def _preds(X):
        p = model.predict(X)
        p[p == 1] = 1
        p[p == -1] = 0
        return p

    def _scores(X):
        # Nao invertemos o sinal aqui. Em decision_function, score alto =
        # mais "inlier" (normal) na visao do Isolation Forest. Como o
        # treino usa dados mistos (benignos + maliciosos, ver NOTAS) e o
        # mapeamento de predict() neste arquivo ja trata "inlier" como
        # label 1 (malicioso), score alto = mais provavel malicioso. Usar
        # -decision_function aqui inverteria o AUC (testado: AUC cai para
        # ~0.08, ou seja, ~1 - 0.92 do sentido correto).
        return model.decision_function(X)

    preds_val = _preds(norm_X_val)
    m_val = get_overall_metrics(y_val, preds_val)
    fbeta_val = fbeta_score(y_val, preds_val, beta=0.5, zero_division=0)

    preds_test = _preds(norm_X_test)
    m_test = get_overall_metrics(y_test, preds_test)
    fbeta_test = fbeta_score(y_test, preds_test, beta=0.5, zero_division=0)

    scores_test = _scores(norm_X_test)
    auc_test = roc_auc_score(y_test, scores_test)

    doc = (
        f"Isolation Forest - avaliacao no conjunto de teste\n"
        f"Config (melhor F0.5 na validacao, ver IsolationForest2.txt): "
        f"contamination={CONTAMINATION} | n_estimators={N_ESTIMATORS} | "
        f"max_samples={MAX_SAMPLES} | max_features={MAX_FEATURES}\n"
        "\nMetricas na validacao (checagem de consistencia com IsolationForest2.txt):\n"
        f"  Accuracy:   {m_val['acc']:.4f}\n"
        f"  Precision:  {m_val['precision']:.4f}\n"
        f"  Recall:     {m_val['tpr']:.4f}\n"
        f"  F1-Score:   {m_val['f1-score']:.4f}\n"
        f"  F0.5-Score: {fbeta_val:.4f}\n"
        "\nMetricas no conjunto de teste:\n"
        f"  Accuracy:   {m_test['acc']:.4f}\n"
        f"  Precision:  {m_test['precision']:.4f}\n"
        f"  Recall:     {m_test['tpr']:.4f}\n"
        f"  F1-Score:   {m_test['f1-score']:.4f}\n"
        f"  F0.5-Score: {fbeta_test:.4f}\n"
        f"  AUC-ROC:    {auc_test:.4f}\n"
    )
    print(doc)
    documentar("IsolationForest_Teste", doc, categoria="isolation_forest")

    fpr, tpr, _ = roc_curve(y_test, scores_test)
    plt.figure()
    plt.plot(fpr, tpr, label=f'Isolation Forest (AUC = {auc_test:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Aleatório')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC - Isolation Forest (conjunto de teste)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(caminho_imagem("isolation_forest_teste_roc.png", "isolation_forest"))
    plt.close()

    return {
        "model": model,
        "metrics_val": m_val,
        "metrics_test": m_test,
        "fbeta_test": fbeta_test,
        "auc": auc_test,
        "scores_test": scores_test,
        "y_test": y_test,
    }


# ============================================================
#                            NOTAS
# ============================================================
#
# Por que esse experimento existe?
# Os scripts originais de Isolation Forest (isolation_forest.py e
# Isolation_forest_com_feature_selection.py) fazem grid search e
# documentam apenas as metricas no conjunto de VALIDACAO — nunca chegam
# a avaliar no conjunto de teste. Para comparar o Isolation Forest com
# K-Means, DBSCAN e Autoencoder (que reportam metricas no teste), e'
# necessario rodar o melhor config encontrado e avaliar no mesmo
# conjunto de teste usado pelos outros modelos.
#
# Por que o conjunto de teste e' comparavel ao dos outros modelos?
# Aqui, assim como em isolation_forest.py, o split usa
# train_test_split(df, test_size=0.2, random_state=2) seguido de
# train_test_split(Xy_tmp, test_size=0.5, random_state=2). Os scripts de
# anomaly_detection (kmeans_anomaly.py, dbscan_anomaly.py,
# autoencoder_anomaly.py) fazem exatamente a mesma chamada sobre o mesmo
# df para obter Xy_val/Xy_test — como train_test_split e' deterministico
# para um mesmo random_state, o Xy_test aqui e' idêntico ao usado por
# esses outros experimentos, mesmo que o Xy_train seja diferente (aqui
# inclui PDFs maliciosos, conforme orientacao do monitor da disciplina;
# nos outros, o treino usa so PDFs benignos).
#
# Por que decision_function (sem inverter o sinal) e' usado como score
# de anomalia aqui, ao contrario de DBSCAN/K-Means/Autoencoder?
# Testando os dois sentidos, a versao sem inversao deu AUC ~0.92 e a
# invertida AUC ~0.08 — uma e' o complemento exato da outra (1 - AUC),
# o que confirma que o ranking esta' tecnicamente correto, so' invertido
# em relacao ao que se esperaria por convencao. Isso revela uma
# limitacao real do Isolation Forest treinado em dados MISTOS e
# aproximadamente balanceados (~52% malicioso / ~48% benigno neste
# dataset, portanto NAO um cenario de desbalanceamento extremo): o
# algoritmo nao sabe, a priori, qual classe e' a "normal" — ele apenas
# isola o que e' menos denso no espaco de features. Neste dataset, os
# PDFs maliciosos (gerados por ferramentas automatizadas, com features
# mais homogeneas) acabaram formando o cluster mais denso, e os PDFs
# benignos (muito mais heterogeneos entre si) ficaram mais isolados.
# Por isso o "outlier" do Isolation Forest e' o PDF benigno, nao o
# malicioso — o oposto do pressuposto usado por K-Means/DBSCAN/
# Autoencoder, que so' veem dados benignos no treino e por isso definem
# anomalia corretamente como "diferente do normal aprendido". E' por
# isso que o isolation_forest.py original mapeia predict()==1 (inlier)
# para label 1 (malicioso): sem essa inversao manual, accuracy/precision
# /recall tambem sairiam invertidos.
#
# Por que escolher a config n_estimators=15, max_samples=0.89,
# max_features=0.1 (sem feature selection) em vez da com feature
# selection (n_estimators=310, max_samples=0.88, max_features=0.045)?
# Comparando os F0.5 documentados na validacao: 0.9014 (sem FS,
# IsolationForest2.txt) contra um F1 de 0.8656 (com FS,
# IsolationForestComFS.txt — essa versao nem chegou a registrar F0.5).
# Mesmo convertendo o resultado da versao com FS para F0.5, ele fica
# abaixo de 0.90. Por isso a versao sem feature selection foi escolhida
# como "melhor Isolation Forest" para a comparacao final.
#
# Por que F0.5 em vez de F1 para escolher a config?
# Para manter o mesmo criterio de comparacao usado no restante deste
# trabalho (ver Comparacao_Fbeta_KMeans_DBSCAN.txt): F0.5 pesa mais a
# Precision, relevante quando falsos positivos tem custo operacional
# (alertas desnecessarios para um analista revisar).
