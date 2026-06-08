from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from utils import documentar

#vamos fazer o pipe basico pro baseline depois podemos tartar melhor os dados apra ter
#um pipe mais completo como a disciplina propõe
def rodar():
    df = pd.read_csv("data/PDF_All_feature_Clean.csv")
    # print(df.head(5))
    df_Bening = df.query("label == 0")
    # df_Malicious = df.query("label == 1")
    df_Bening.drop(columns= 'label', inplace = True)
    std = StandardScaler()
    
    isf = IsolationForest(n_estimators=100,random_state=67)
    
