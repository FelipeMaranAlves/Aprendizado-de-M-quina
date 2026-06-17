import numpy as np
import matplotlib.pyplot as plt
from utils import documentar, caminho_imagem

# Numeros vindos dos experimentos ja existentes deste repositorio, todos
# avaliados no MESMO conjunto de teste (Xy_test derivado de
# train_test_split(df, test_size=0.2, random_state=2) seguido de
# train_test_split(Xy_tmp, test_size=0.5, random_state=2) — ver NOTAS em
# cada script para a prova de que o split e' identico entre eles).
# Critério de selecao de threshold/hiperparametros em todos: F0.5.
#
# K-Means (K=2):            sessions/anomaly_detection/kmeans_anomaly_fbeta.py
# DBSCAN (eps=3.0):         sessions/anomaly_detection/dbscan_anomaly_fbeta.py
# Isolation Forest:         sessions/isolation_forest/isolation_forest_teste.py
# Autoencoder (3 camadas):  sessions/anomaly_detection/autoencoder3camadas.py
#                           + models/best_ae_3.json (checkpoint_ae_3_camadas24-20-16.pt)
RESULTADOS = {
    "K-Means (K=2)": {
        "config": "threshold=10.5391",
        "acc": 0.9010, "precision": 0.9617, "recall": 0.8399,
        "f1": 0.8967, "f05": 0.9346, "auc": 0.9722,
    },
    "DBSCAN (eps=3.0)": {
        "config": "threshold=1.7662",
        "acc": 0.9580, "precision": 0.9613, "recall": 0.9564,
        "f1": 0.9589, "f05": 0.9603, "auc": 0.9853,
    },
    "Isolation Forest": {
        "config": "n_estimators=15, max_samples=0.89, max_features=0.1",
        "acc": 0.8974, "precision": 0.9219, "recall": 0.8734,
        "f1": 0.8970, "f05": 0.9118, "auc": 0.9191,
    },
    "Autoencoder (3 camadas)": {
        "config": "arquitetura 33-24-20-16-20-24-33, threshold=0.051258",
        "acc": 0.8720, "precision": 0.9973, "recall": 0.7518,
        "f1": 0.8573, "f05": 0.9362, "auc": 0.9776,
    },
}


def rodar():
    modelos = list(RESULTADOS.keys())
    ranking_f05 = sorted(modelos, key=lambda m: RESULTADOS[m]["f05"], reverse=True)

    doc_lines = [
        "Comparacao final: melhor K-Means, DBSCAN, Isolation Forest e Autoencoder",
        "Criterio de selecao do threshold/hiperparametros em todos os modelos: F0.5",
        "(mesmo conjunto de teste para todos — ver cabecalho do script para a prova)\n",
    ]
    for nome in modelos:
        r = RESULTADOS[nome]
        doc_lines.append(
            f"{nome}\n"
            f"  Config:    {r['config']}\n"
            f"  Accuracy:  {r['acc']:.4f}\n"
            f"  Precision: {r['precision']:.4f}\n"
            f"  Recall:    {r['recall']:.4f}\n"
            f"  F1-Score:  {r['f1']:.4f}\n"
            f"  F0.5:      {r['f05']:.4f}\n"
            f"  AUC-ROC:   {r['auc']:.4f}\n"
        )

    doc_lines.append("Ranking por F0.5 (metrica de referencia deste experimento):")
    for i, nome in enumerate(ranking_f05, start=1):
        doc_lines.append(f"  {i}. {nome} (F0.5={RESULTADOS[nome]['f05']:.4f})")

    doc = "\n".join(doc_lines) + "\n"
    print(doc)
    documentar("Comparacao_Final_Modelos", doc, categoria="anomaly_detection")

    metricas = ["precision", "recall", "f05", "auc"]
    labels = ["Precision", "Recall", "F0.5", "AUC-ROC"]
    x = np.arange(len(modelos))
    width = 0.2

    plt.figure(figsize=(9, 5))
    for i, (met, lab) in enumerate(zip(metricas, labels)):
        valores = [RESULTADOS[m][met] for m in modelos]
        plt.bar(x + (i - 1.5) * width, valores, width, label=lab)
    plt.xticks(x, modelos, rotation=10)
    plt.ylim(0, 1.05)
    plt.ylabel("Score no teste")
    plt.title("Comparacao final dos detectores de anomalia (criterio: F0.5)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(caminho_imagem("comparacao_final_modelos.png", "anomaly_detection"))
    plt.close()

    return RESULTADOS


# ============================================================
#                            NOTAS
# ============================================================
#
# Por que os numeros sao hardcoded em vez de recalculados aqui?
# Cada modelo ja tem seu proprio script reprodutivel (citado no
# cabecalho) que treina, faz a busca de threshold/hiperparametro por
# F0.5 e documenta o resultado no conjunto de teste. Reexecutar os
# quatro aqui duplicaria treinamento caro (principalmente o
# Autoencoder) sem agregar informacao nova: este script e' apenas a
# camada de SINTESE/comparacao, igual a Comparacao_Fbeta_KMeans_DBSCAN.txt
# ja fazia para K-Means vs DBSCAN.
#
# Por que o Isolation Forest fica em ultimo lugar em F0.5 e AUC?
# Ver as NOTAS de isolation_forest_teste.py: e' o unico modelo treinado
# em dados MISTOS (benignos + maliciosos), nao apenas benignos. Nesse
# regime, o algoritmo descobriu que os PDFs maliciosos formam o cluster
# mais denso (mais "normal" sob a logica de isolamento) e os benignos,
# por serem mais heterogeneos, ficam mais isolados — o oposto do
# pressuposto de anomalia usado pelos outros tres modelos. Isso nao
# significa que Isolation Forest seja uma tecnica pior em geral; indica
# que, NESTE dataset e com esse protocolo de treino, ele tem uma
# desvantagem estrutural em relacao aos modelos treinados apenas com
# dados normais.
#
# Por que o Autoencoder tem a maior Precision (0.9973) mas a menor
# F1 (0.8573)?
# O threshold foi escolhido maximizando F0.5 na validacao, que pesa
# Precision 4x mais que Recall. O resultado e' um modelo extremamente
# conservador: quase nunca classifica um PDF benigno como malicioso
# (poucos falsos positivos), mas em troca deixa passar ~25% dos PDFs
# maliciosos (Recall=0.7518). Isso ilustra bem o trade-off discutido em
# Comparacao_Fbeta_KMeans_DBSCAN.txt: o mesmo criterio (F0.5) pode levar
# a operating points bem diferentes dependendo de quao "afiada" e' a
# fronteira aprendida por cada modelo.
#
# Por que DBSCAN venceu em F0.5 E em AUC-ROC simultaneamente?
# AUC-ROC mede a qualidade do ranking continuo (independente de
# threshold), enquanto F0.5 mede o resultado de um corte especifico.
# Vencer nos dois ao mesmo tempo indica que a vantagem do DBSCAN nao e'
# so' ter encontrado um bom ponto de corte — o proprio score continuo
# (distancia ao core point mais proximo) separa melhor benignos de
# maliciosos do que os scores dos outros tres modelos, neste dataset.
