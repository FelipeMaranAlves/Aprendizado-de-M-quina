import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from scipy.stats import skew, kurtosis
from pipeline import carregar
from utils import caminho_imagem

df = carregar()

print(f"Dataset shape: {df.shape}")
print(f"Colunas: {df.columns.tolist()}")
print(f"\nTipos de dados:\n{df.dtypes.value_counts()}")
print(f"\nPrimeiras linhas:\n{df.head()}")


print(f"\nDistribuição das classes:\n{df['label'].value_counts()}")
print(f"Proporção: {df['label'].value_counts(normalize=True)}")

numeric_cols = df.select_dtypes(include=[np.number]).columns
numeric_cols = [col for col in numeric_cols if col != 'label']

desc_stats = df[numeric_cols].describe()
Path('Documentation/univariate_analysis').mkdir(parents=True, exist_ok=True)
desc_stats.to_csv('Documentation/univariate_analysis/analise_univariada_estatisticas.csv')

null_counts = df[numeric_cols].isnull().sum()
null_counts = null_counts[null_counts > 0]
if len(null_counts) > 0:
    print(f"{len(null_counts)} colunas têm valores nulos")

zero_variance = []
for col in numeric_cols:
    if df[col].nunique() <= 1:
        zero_variance.append(col)
if len(zero_variance) > 0:
    print(f"{len(zero_variance)} colunas têm variância zero: {zero_variance}")

for col in numeric_cols:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df[col], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_title(f'Distribuição de {col}')
    axes[0].set_xlabel(col)
    axes[0].set_ylabel('Frequência')
    
    axes[1].boxplot(df[col], vert=True)
    axes[1].set_title(f'Boxplot de {col}')
    axes[1].set_ylabel(col)
    
    plt.tight_layout()
    plt.savefig(caminho_imagem(f'{col}.png', 'univariate_analysis/univariate_plots'), dpi=100)
    plt.close()

outliers_info = {}
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outliers_info[col] = {
        'n_outliers': len(outliers),
        'pct_outliers': (len(outliers) / len(df)) * 100,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }
outliers_df = pd.DataFrame(outliers_info).T
outliers_df.to_csv('Documentation/univariate_analysis/analise_outliers.csv')

skewness = {}
kurt = {}
for col in numeric_cols:
    skewness[col] = skew(df[col].dropna())
    kurt[col] = kurtosis(df[col].dropna())

skew_df = pd.DataFrame([skewness, kurt], index=['Skewness', 'Kurtosis']).T
skew_df.to_csv('Documentation/univariate_analysis/analise_assimetria_curtose.csv')

plt.figure(figsize=(8, 5))
df['label'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Distribuição das Classes (Benigno vs Malicioso)')
plt.xlabel('Classe (0 = Benigno, 1 = Malicioso)')
plt.ylabel('Quantidade')
plt.xticks(rotation=0)
for i, v in enumerate(df['label'].value_counts().values):
    plt.text(i, v + 100, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(caminho_imagem('distribuicao_classes.png', 'univariate_analysis'), dpi=100)
plt.close()

with open('Documentation/univariate_analysis/analise_univariada_resumo.txt', 'w') as f:
    f.write(f"Dataset: PDF_All_feature_Clean.csv\n")
    f.write(f"Total de amostras: {len(df)}\n")
    f.write(f"Total de features: {len(numeric_cols)}\n\n")
    f.write("Distribuicao das classes:\n")
    class_dist = df['label'].value_counts()
    f.write(f"  Classe 0 (Benigno): {class_dist[0]} ({class_dist[0]/len(df)*100:.2f}%)\n")
    f.write(f"  Classe 1 (Malicioso): {class_dist[1]} ({class_dist[1]/len(df)*100:.2f}%)\n\n")
    f.write("Principais insights:\n")
    f.write(f"  - Features com variancia zero: {len(zero_variance)}\n")
    high_outliers = outliers_df[outliers_df['pct_outliers'] > 5]
    f.write(f"  - Features com >5%% outliers: {len(high_outliers)}\n")
    high_skew = skew_df[abs(skew_df['Skewness']) > 2]
    f.write(f"  - Features com alta assimetria: {len(high_skew)}\n")

print("Analise univariada concluida. Arquivos salvos em Documentation/univariate_analysis/ e Images/univariate_analysis/")