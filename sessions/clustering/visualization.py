import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import caminho_imagem


def _preparar_dados():
    df = pd.read_csv("data/PDF_All_feature_Clean.csv")
    df.drop(columns='file_path', inplace=True)

    df_Bening = df.query("label == 0")

    RAND_STATE = 2
    X_train, _ = train_test_split(df_Bening, test_size=0.2, random_state=RAND_STATE)
    _, Xy_tmp = train_test_split(df, test_size=0.2, random_state=RAND_STATE)
    Xy_val, Xy_test = train_test_split(Xy_tmp, test_size=0.5, random_state=RAND_STATE)

    X_train = X_train.drop(columns=['label'])
    y_test = Xy_test['label'].values
    X_test = Xy_test.drop(columns=['label'])

    std_scaler = StandardScaler()
    std_scaler.fit(X_train)
    nomr_X_train = std_scaler.transform(X_train)
    norm_X_test = std_scaler.transform(X_test)

    return nomr_X_train, norm_X_test, y_test


def gerar_graficos(clustering_results):
    Ks = [r["n_clusters"] for r in clustering_results]
    silhouettes = [r["silhouette"] for r in clustering_results]
    inertias = [r["inertia"] for r in clustering_results]

    plt.figure()
    plt.plot(Ks, silhouettes, marker='o')
    plt.xlabel("Numero de Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score por Numero de Clusters")
    plt.savefig(caminho_imagem("clustering_silhouette.png", "clustering"))
    plt.close()

    plt.figure()
    plt.plot(Ks, inertias, marker='o')
    plt.xlabel("Numero de Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.savefig(caminho_imagem("clustering_elbow.png", "clustering"))
    plt.close()


def gerar_graficos_pca(clustering_results):
    X_train, X_test, y_test = _preparar_dados()
    labels_train = clustering_results[0]["labels"]  # K=2

    pca = PCA(n_components=2, random_state=2)
    X_train_2d = pca.fit_transform(X_train)
    X_test_2d = pca.transform(X_test)

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    scatter = axes[0].scatter(
        X_train_2d[:, 0], X_train_2d[:, 1],
        c=labels_train, cmap='RdYlGn', alpha=0.3, s=5
    )
    axes[0].set_title("PDFs Benignos - Clusters K-Means (K=2)")
    axes[0].set_xlabel(f"PC1 ({var1:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({var2:.1f}%)")
    plt.colorbar(scatter, ax=axes[0], label='Cluster')

    colors = ['green' if l == 0 else 'red' for l in y_test]
    axes[1].scatter(
        X_test_2d[:, 0], X_test_2d[:, 1],
        c=colors, alpha=0.4, s=10
    )
    axes[1].set_title("Teste - Benigno (verde) vs Malicioso (vermelho)")
    axes[1].set_xlabel(f"PC1 ({var1:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({var2:.1f}%)")

    plt.tight_layout()
    plt.savefig(caminho_imagem("clustering_pca.png", "clustering"))
    plt.close()
    print("PCA salvo: Images/clustering/clustering_pca.png")

    n_components = min(20, X_train.shape[1])
    pca_full = PCA(n_components=n_components, random_state=2)
    pca_full.fit(X_train)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100

    plt.figure()
    plt.plot(range(1, n_components + 1), cumvar, marker='o')
    plt.axhline(y=90, color='r', linestyle='--', label='90% variância')
    plt.xlabel("Número de Componentes Principais")
    plt.ylabel("Variância Explicada Acumulada (%)")
    plt.title("PCA - Variância Explicada Acumulada")
    plt.legend()
    plt.tight_layout()
    plt.savefig(caminho_imagem("clustering_pca_variancia.png", "clustering"))
    plt.close()
    print("Variância PCA salva: Images/clustering/clustering_pca_variancia.png")


def gerar_graficos_tsne(clustering_results, sample_size=2000):
    X_train, _, _ = _preparar_dados()
    labels_train = clustering_results[0]["labels"]  # K=2

    if len(X_train) > sample_size:
        rng = np.random.RandomState(2)
        idx = rng.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train[idx]
        labels_sample = labels_train[idx]
    else:
        X_sample = X_train
        labels_sample = labels_train

    print(f"Rodando t-SNE em {len(X_sample)} amostras (pode levar alguns minutos)...")
    tsne = TSNE(n_components=2, random_state=2, perplexity=30, max_iter=1000)
    X_2d = tsne.fit_transform(X_sample)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=labels_sample, cmap='RdYlGn', alpha=0.5, s=10
    )
    plt.colorbar(scatter, label='Cluster K-Means')
    plt.title(f"t-SNE - PDFs Benignos por Cluster (K=2, {len(X_sample)} amostras)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(caminho_imagem("clustering_tsne.png", "clustering"))
    plt.close()
    print("t-SNE salvo: Images/clustering/clustering_tsne.png")


# ============================================================
#                           NOTAS 
# ============================================================
#
# Por que visualizar os clusters em 2D se os dados têm 33 features?
# Com alta dimensionalidade, é impossível enxergar a estrutura dos dados
# diretamente. A redução para 2D nos permite entender intuitivamente se
# os clusters fazem sentido e, principalmente, onde os PDFs maliciosos
# do conjunto de teste caem em relação aos clusters benignos aprendidos.
# Se os maliciosos ficarem fora das regiões densas dos benignos, isso
# valida visualmente a ideia de usar distância como score de anomalia.
#
# Por que PCA?
# PCA é linear, determinístico e muito rápido. As componentes principais
# capturam as direções de maior variância, tendendo a preservar a estrutura
# global dos clusters. O gráfico de variância acumulada complementa isso
# mostrando quantas componentes seriam necessárias para representar os dados
# fielmente (útil para decisões de feature reduction).
#
# Por que t-SNE?
# t-SNE é não-linear e preserva melhor a estrutura LOCAL dos dados, sendo
# especialmente útil para revelar clusters que PCA não consegue separar.
# A desvantagem é o custo computacional O(n²), por isso limitamos a
# amostra a 2000 pontos. Importante: distâncias globais no t-SNE não são
# significativas — só a separação entre clusters importa.
#
# Por que dois subplots separados no PCA (treino vs teste)?
# O subplot do treino mostra como K-Means dividiu os PDFs benignos no
# espaço das componentes. O subplot do teste usa a mesma projeção PCA
# (transform, não fit_transform) para mostrar onde benignos e maliciosos
# caem nesse mesmo espaço. Usar transform garante que a projeção seja
# consistente, sem data leakage.
