import itertools
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from utils import documentar
from utilsDoProfessor import get_overall_metrics, plot_confusion_matrix
from pipeline import carregar, normalizar


def rodar(): 
    df = carregar() 
    
    #monitora falou que nao era pra deixar o treino so com dados benignos oq ue contradiz o slide do professor 
    #No momento treino e testa ambos estao com 10% do total dos dados (não balanceado)
    RAND_STATE = 2
    Xy_train, Xy_tmp = train_test_split(df, test_size=0.2, random_state=RAND_STATE)
    Xy_val, Xy_test = train_test_split(Xy_tmp, test_size=0.5, random_state=RAND_STATE)

    y_train = Xy_train['label']
    X_train = Xy_train.drop(columns=['label'])
    y_val = Xy_val['label']
    X_val = Xy_val.drop(columns=['label'])
    y_test = Xy_test['label']
    X_test = Xy_test.drop(columns=['label'])

    #normalizando baseado em z
    #normalizando com a mesma escala do treino para nao ter dataleak!
    nomr_X_train, norm_X_val, norm_X_test = normalizar(X_train, X_val, X_test)
    """
    for i in range(10):
        CONTAMINATION = (0.05 * (i+1))
        isf_model = IsolationForest(n_estimators=100,random_state=RAND_STATE,contamination=CONTAMINATION).fit(nomr_X_train)

        predicoes_val = isf_model.predict(norm_X_val)
        predicoes_val[predicoes_val == 1] = 1
        predicoes_val[predicoes_val == -1] = 0
        documentar("IsolationForest2",f"Metricas com o valor de contaminacao: {CONTAMINATION}\n"+str(get_overall_metrics(y_val,predicoes_val)), categoria="isolation_forest")
    """

    CONTAMINATION = 0.5
    n_estimators_grid = [15, 20, 25, 30, 35]
    max_samples_grid = [0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93]
    max_features_grid = [0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13]
    # Melhor métrica desse grid: n_estimators=15 | max_samples=0.89 | max_features=0.1

    grid = itertools.product(
        n_estimators_grid,
        max_samples_grid,
        max_features_grid,
    )

    for n_est, max_samp, max_feat in grid:
        isf_model = IsolationForest(
            n_estimators=n_est,
            max_samples=max_samp,
            max_features=max_feat,
            contamination=CONTAMINATION,
            random_state=RAND_STATE,
        ).fit(nomr_X_train)
        isf_model.score_samples()
        predicoes_val = isf_model.predict(norm_X_val)

        predicoes_val[predicoes_val == 1] = 1
        predicoes_val[predicoes_val == -1] = 0

        header = (
            f"contamination={CONTAMINATION} | "
            f"n_estimators={n_est} | "
            f"max_samples={max_samp} | "
            f"max_features={max_feat}\n"
        )

        documentar(
            "IsolationForest2",
            header + str(get_overall_metrics(y_val, predicoes_val)),
            categoria="isolation_forest",
        )