import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from utils import documentar
from sklearn.metrics import silhouette_score
from pipeline import carregar, normalizar


def rodar():
    df = carregar()

    # print(df.head(5))
    df_Bening = df.query("label == 0")
    # df_Bening.drop(columns= 'label', inplace = True)

    #divisao diferente de treino para benigno e teste incluindo ambos.
    #No momento treino e testa ambos estao com 10% do total dos dados (não balanceado)
    RAND_STATE = 2
    X_train, _ = train_test_split(df_Bening,test_size=0.2,random_state=RAND_STATE)
    _, Xy_tmp = train_test_split(df,test_size=0.2,random_state=RAND_STATE)
    Xy_val, Xy_test = train_test_split(Xy_tmp,test_size=0.5, random_state=RAND_STATE)

    y_train = X_train['label']
    X_train = X_train.drop(columns = ['label'])
    y_val = Xy_val['label']
    X_val = Xy_val.drop(columns = ['label'])
    y_test = Xy_test['label']
    X_test =  Xy_test.drop(columns = ['label'])

    #normalizando bazeado em z
    #normalizando com a mesma escala do treino para nao ter dataleak!
    nomr_X_train, norm_X_val, norm_X_test = normalizar(X_train, X_val, X_test)

    s = []
    inertias = []
    clustering_results = []

    for i in range(2,11,1):
        K = i
        model = KMeans(n_clusters=K,random_state=RAND_STATE,n_init=10)
        model.fit(nomr_X_train)
        labels = model.predict(nomr_X_train)

        sil_score = silhouette_score(nomr_X_train, labels)
        inertia = model.inertia_

        s.append((K, sil_score))
        inertias.append((K, inertia))

        clustering_results.append({
            "n_clusters": K,
            "silhouette": sil_score,
            "inertia": inertia,
            "labels": labels,
            "model": model,
        })

    for i in range(len(s)):
        print(f"K={s[i][0]} | silhouette={s[i][1]}")

    for i in range(len(inertias)):
        print(f"K={inertias[i][0]} | inertia={inertias[i][1]}")

    doc_texto = "Experimento K-Means - silhouette e inertia por K\n"
    for r in clustering_results:
        doc_texto += f"K={r['n_clusters']} | silhouette={r['silhouette']} | inertia={r['inertia']}\n"

    documentar("Clustering", doc_texto)

    return clustering_results