import pandas as pd
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