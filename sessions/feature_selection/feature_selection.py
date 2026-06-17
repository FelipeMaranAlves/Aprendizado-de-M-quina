#vou começar a experimentar mais,
#tentar selecionar features com random forest tambem

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from utils import documentar
from pipeline import carregar

def rodar():
    df = carregar()

    X = df.drop(columns=["label"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)
    rf = RandomForestClassifier(n_estimators=100,random_state=6467)
    rf.fit(X_train,y_train)
    importances = pd.Series(rf.feature_importances_,index=X_train.columns)
    importances.sort_values(ascending=False,inplace=True)
    documentar("Quinto_contato_Sp3", str(importances), categoria="feature_selection")