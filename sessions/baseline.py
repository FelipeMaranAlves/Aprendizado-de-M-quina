from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from utils import documentar
from utilsDoProfessor import get_overall_metrics

#vamos fazer o pipe basico pro baseline depois podemos tartar melhor os dados apra ter
#um pipe mais completo como a disciplina propõe
def rodar():
    df = pd.read_csv("data/PDF_All_feature_Clean.csv")
    df.drop(columns= 'file_path', inplace = True)
    
    # print(df.head(5))
    df_Bening = df.query("label == 0")
    # df_Bening.drop(columns= 'label', inplace = True)
    
    #divisao diferente de treino para benigno e teste incluindo ambos.
    #No momento treino e testa ambos estao com 10% do total dos dados (não balanceado)
    X_train, _ = train_test_split(df_Bening,test_size=0.2,random_state=2)
    _, Xy_tmp = train_test_split(df,test_size=0.2,random_state=2)
    Xy_val, Xy_test = train_test_split(Xy_tmp,test_size=0.5)

    #normalizando bazeado em z
    #normalizando com a mesma escala do treino para nao ter dataleak!
    std_scaler = StandardScaler()
    std_scaler = std_scaler.fit(X_train)
    nomr_X_train = std_scaler.transform(X_train)
    norm_Xy_val = std_scaler.transform(Xy_val)
    norm_Xy_test = std_scaler.transform(Xy_test)

    isf_model = IsolationForest(n_estimators=100,random_state=67).fit(nomr_X_train)
    
    predicoes = isf_model.predict(df)
    predicoes[predicoes == 1] = 1
    predicoes[predicoes == -1] = 0
    # string_T = "True\n"
    # string_T += str(df['label'].head(20).to_list())
    # documentar("baseline_1_saida",string_T)
    # string_P = "Predito\n"
    # string_P += str(predicoes[0:20].tolist())
    # documentar("baseline_1_saida",string_P)
    get_overall_metrics()

    
