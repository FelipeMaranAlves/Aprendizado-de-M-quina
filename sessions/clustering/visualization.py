import os
import matplotlib.pyplot as plt


def gerar_graficos(clustering_results):
    # garante que a pasta de saida existe
    os.makedirs("Images", exist_ok=True)

    Ks = [r["n_clusters"] for r in clustering_results]
    silhouettes = [r["silhouette"] for r in clustering_results]
    inertias = [r["inertia"] for r in clustering_results]

    # Silhouette Score x K
    plt.figure()
    plt.plot(Ks, silhouettes, marker='o')
    plt.xlabel("Numero de Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score por Numero de Clusters")
    plt.savefig("Images/clustering_silhouette.png")
    plt.close()

    # Elbow Method (Inertia x K)
    plt.figure()
    plt.plot(Ks, inertias, marker='o')
    plt.xlabel("Numero de Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.savefig("Images/clustering_elbow.png")
    plt.close()