import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import documentar


def _preparar_dados():
    df = pd.read_csv("data/PDF_All_feature_Clean.csv")
    df.drop(columns='file_path', inplace=True)

    df_Bening = df.query("label == 0")

    RAND_STATE = 2
    X_train, _ = train_test_split(df_Bening, test_size=0.2, random_state=RAND_STATE)

    y_train = X_train['label']
    X_train = X_train.drop(columns=['label'])

    return X_train


def analisar_features(df, labels):
    if df is None:
        df = _preparar_dados()

    df_analise = df.copy()
    df_analise["cluster"] = labels

    grouped = df_analise.groupby("cluster")

    medias = grouped.mean()
    desvios = grouped.std()

    os.makedirs("Documentation/clustering", exist_ok=True)
    medias.to_csv("Documentation/clustering/cluster_feature_means.csv")
    doc_texto = "Analise de Features por Cluster\n"

    feature_cols = [c for c in df_analise.columns if c != "cluster"]

    for feature in feature_cols:
        valores = medias[feature]

        cluster_max = valores.idxmax()
        cluster_min = valores.idxmin()
        media_max = valores.max()
        media_min = valores.min()
        diferenca = media_max - media_min

        doc_texto += f"\nFeature: {feature}\n"
        for cluster_id, media_val in valores.items():
            doc_texto += f"Cluster {cluster_id} media: {media_val}\n"
        doc_texto += (
            f"Maior media: Cluster {cluster_max} ({media_max})\n"
            f"Menor media: Cluster {cluster_min} ({media_min})\n"
            f"Diferenca: {diferenca}\n"
        )

    documentar("Analise de Features", doc_texto, categoria="clustering")
    return medias, desvios