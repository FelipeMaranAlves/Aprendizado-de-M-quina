import itertools
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from utils import documentar
from utilsDoProfessor import get_overall_metrics, plot_confusion_matrix
from pipeline import carregar, normalizar


def rodar(): 
    df = carregar() 
    
    RAND_STATE = 2
    Xy_train, Xy_tmp = train_test_split(df, test_size=0.2, random_state=RAND_STATE)
    Xy_val, Xy_test = train_test_split(Xy_tmp, test_size=0.5, random_state=RAND_STATE)

    y_train = Xy_train['label']
    X_train = Xy_train.drop(columns=['label'])
    y_val = Xy_val['label']
    X_val = Xy_val.drop(columns=['label'])
    y_test = Xy_test['label']
    X_test = Xy_test.drop(columns=['label'])

    # Normalização
    norm_X_train, norm_X_val, norm_X_test = normalizar(X_train, X_val, X_test)

    # Feature Selection usando Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=6467)
    rf.fit(norm_X_train, y_train)
    
    # Selecionando features com importância > threshold
    selector = SelectFromModel(rf, threshold='median', prefit=True)
    norm_X_train_selected = selector.transform(norm_X_train)
    norm_X_val_selected = selector.transform(norm_X_val)
    norm_X_test_selected = selector.transform(norm_X_test)
    
    # Obtendo nomes das features selecionadas
    selected_features = X_train.columns[selector.get_support()].tolist()
    
    # Documentando as features selecionadas
    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    importances_sorted = importances.sort_values(ascending=False)
    documentar("IsolationForest_FeatureSelection_Importances", 
               f"Features selecionadas ({len(selected_features)}): {selected_features}\n\n"
               f"Importâncias:\n{str(importances_sorted)}", 
               categoria="feature_selection")

    # Grid de hiperparâmetros
    CONTAMINATION = 0.5
    n_estimators_grid = [310]
    max_samples_grid = [0.88]
    max_features_grid = [0.045]
    # Melhor resultado foi assim
    # acc: 0.865285
    # tpr: 0.856704
    # fpr: 0.125918
    # precision: 0.874608
    # f1-score: 0.865564

    grid = itertools.product(
        n_estimators_grid,
        max_samples_grid,
        max_features_grid,
    )

    # Tunagem com feature selection
    for n_est, max_samp, max_feat in grid:
        isf_model = IsolationForest(
            n_estimators=n_est,
            max_samples=max_samp,
            max_features=max_feat,
            contamination=CONTAMINATION,
            random_state=RAND_STATE,
        ).fit(norm_X_train_selected)
        
        predicoes_val = isf_model.predict(norm_X_val_selected)

        # Convertendo predições (1 = inlier, -1 = outlier)
        predicoes_val[predicoes_val == 1] = 1
        predicoes_val[predicoes_val == -1] = 0

        header = (
            f"contamination={CONTAMINATION} | "
            f"n_estimators={n_est} | "
            f"max_samples={max_samp} | "
            f"max_features={max_feat} | "
            f"features_selected={len(selected_features)}\n"
        )

        documentar(
            "IsolationForestComFS",
            header + str(get_overall_metrics(y_val, predicoes_val)),
            categoria="isolation_forest",
        )
    
    # Opcional: Testar com o melhor modelo encontrado
    # Você pode adicionar lógica para guardar o melhor modelo e testar no test set
    
    # Testar o melhor modelo encontrado (exemplo com n_estimators=200, max_samples=0.5, max_features=0.5)
    # isf_best = IsolationForest(
    #     n_estimators=200,
    #     max_samples=0.5,
    #     max_features=0.5,
    #     contamination=CONTAMINATION,
    #     random_state=RAND_STATE,
    # ).fit(norm_X_train_selected)
    # predicoes_test = isf_best.predict(norm_X_test_selected)
    # predicoes_test[predicoes_test == 1] = 1
    # predicoes_test[predicoes_test == -1] = 0
    # print("Resultados no test set:", get_overall_metrics(y_test, predicoes_test))