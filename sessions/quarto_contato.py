import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from utils import configurar, documentar
def rodar():
    configurar()
    df = pd.read_csv('data/PDF_All_feature_Clean.csv')
    # corr_matrix = df.select_dtypes(include=['number']).corr('kendall')
    # with open("matrix.txt",'w') as file:
    #     file.write(str(corr_matrix))
    # sns.heatmap(corr_matrix,xticklabels=True,yticklabels=True,cmap='jet')
    # plt.show()
    corrs = df.select_dtypes(include=['number']).corr()['label'].abs().apply(lambda x: x*x).sort_values(ascending=False)
    documentar("quarto_contato.txt",str(corrs))
