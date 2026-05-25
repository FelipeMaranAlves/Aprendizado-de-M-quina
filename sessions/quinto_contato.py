#vou começar a experimentar mais,
#tentar selecionar features com random forest tambem

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
def rodar():
    df = pd.read_csv('data/PDF_All_feature_Clean.csv')
    df = df.drop(columns='file_path')
    X_train, X_test, Y_train, Y_test = train_test_split(df,0.8,0.2)
    rf = RandomForestClassifier(n_estimators=100,random_state=42)
    rf.fit(X_train,Y_train)
    importances = pd.Series(rf.feature_importances_,index=X_train.columns)
    importances.sort_values(ascending=False,inplace=True)
    print(importances)