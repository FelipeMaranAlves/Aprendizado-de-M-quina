from utils import documentar, configurar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def rodar():
    df = pd.read_csv('data/PDF_All_feature_Clean.csv')
    configurar()

    # Selecionar apenas colunas numéricas
    numeric_df = df.select_dtypes(include=['number'])

    # Correlação com a label
    correlations = numeric_df.corr('spearman')['label'].drop('label')

    # Ordenar correlações
    correlations = correlations.sort_values()

    # Plot
    plt.figure(figsize=(10, 8))

    correlations.plot(kind='barh')

    plt.xlabel('Correlação com label')
    plt.ylabel('Variáveis')
    plt.title('Correlação Bivariada de pearson das Features com a Label')

    plt.grid(True)

    plt.tight_layout()
    plt.savefig('correlacao_bi_spearmen')