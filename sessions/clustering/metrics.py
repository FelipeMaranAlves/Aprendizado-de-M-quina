import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from utils import documentar


def _preparar_dados():
    df = pd.read_csv("data/PDF_All_feature_Clean.csv")
    df.drop(columns='file_path', inplace=True)

    df_Bening = df.query("label == 0")

    RAND_STATE = 2
    X_train, _ = train_test_split(df_Bening, test_size=0.2, random_state=RAND_STATE)
    _, Xy_tmp = train_test_split(df, test_size=0.2, random_state=RAND_STATE)
    Xy_val, Xy_test = train_test_split(Xy_tmp, test_size=0.5, random_state=RAND_STATE)

    y_train = X_train['label']
    X_train = X_train.drop(columns=['label'])
    y_val = Xy_val['label']
    X_val = Xy_val.drop(columns=['label'])
    y_test = Xy_test['label']
    X_test = Xy_test.drop(columns=['label'])

    std_scaler = StandardScaler()
    std_scaler = std_scaler.fit(X_train)
    nomr_X_train = std_scaler.transform(X_train)
    norm_X_val = std_scaler.transform(X_val)
    norm_X_test = std_scaler.transform(X_test)

    return nomr_X_train


def avaliar_clusters(X, clustering_results):
    if X is None:
        X = _preparar_dados()

    metrics_results = []

    for r in clustering_results:
        K = r["n_clusters"]
        labels = r["labels"]

        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)

        metrics_results.append({
            "n_clusters": K,
            "silhouette": sil,
            "davies_bouldin": db,
            "calinski_harabasz": ch,
        })

    for m in metrics_results:
        print(f"K={m['n_clusters']} | silhouette={m['silhouette']} | "
              f"davies_bouldin={m['davies_bouldin']} | "
              f"calinski_harabasz={m['calinski_harabasz']}")

    doc_texto = "Avaliacao de metricas adicionais dos clusters\n"
    for m in metrics_results:
        doc_texto += (
            f"K={m['n_clusters']} | "
            f"silhouette={m['silhouette']} | "
            f"davies_bouldin={m['davies_bouldin']} | "
            f"calinski_harabasz={m['calinski_harabasz']}\n"
        )

    documentar("Metricas", doc_texto)
    return metrics_results