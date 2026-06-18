import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def carregar():
    df = pd.read_csv("data/PDF_All_feature_Clean.csv")
    df.drop(columns= 'file_path', inplace = True)
    return df


def normalizar(X_train, X_val, X_test):
    std_scaler = StandardScaler()
    std_scaler = std_scaler.fit(X_train)
    nomr_X_train = std_scaler.transform(X_train)
    norm_X_val   = std_scaler.transform(X_val)
    norm_X_test  = std_scaler.transform(X_test)
    return nomr_X_train, norm_X_val, norm_X_test

def tratar_outliers_treino(X_train, X_val, X_test):

    # Aplica clipping/capping nos outliers usando o método IQR baseado APENAS no treino.
    # Evita vazamento de dados e preserva amostras nos conjuntos de validação/teste.
    X_train_clean = X_train.copy()
    X_val_clean = X_val.copy()
    X_test_clean = X_test.copy()
    
    for col in X_train.columns:
        # Calcula os quartis baseando-se estritamente no Treino (Benignos)
        q1 = X_train[col].quantile(0.25)
        q3 = X_train[col].quantile(0.75)
        iqr = q3 - q1
        
        limite_superior = q3 + 1.5 * iqr
        
        # Aplica o capping: valores acima do limite viram o próprio limite
        # Usamos np.clip para garantir que nenhum valor ultrapasse o teto estabelecido pelo treino
        X_train_clean[col] = np.clip(X_train_clean[col], None, limite_superior)
        X_val_clean[col]   = np.clip(X_val_clean[col], None, limite_superior)
        X_test_clean[col]  = np.clip(X_test_clean[col], None, limite_superior)
        
    return X_train_clean, X_val_clean, X_test_clean