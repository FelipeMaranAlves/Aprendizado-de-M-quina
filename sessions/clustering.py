import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from utils import documentar
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def rodar():
    df = pd.read_csv("data/PDF_All_feature_Clean.csv")
    df.drop(columns= 'file_path', inplace = True)
    
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
    std_scaler = StandardScaler()
    std_scaler = std_scaler.fit(X_train)
    nomr_X_train = std_scaler.transform(X_train)
    norm_X_val = std_scaler.transform(X_val)
    norm_X_test = std_scaler.transform(X_test)

    s =[]
    for i in range(2,11,1):
        K = i
        model = KMeans(n_clusters=K,random_state=RAND_STATE,n_init=10)
        model.fit(X_train)
        s.append( (K,silhouette_score(X_train,model.predict(X_train))))

    for i in range(len(s)):
        print(f"Numero de clusters {s[i][0]} s_score{s[i][1]}")

