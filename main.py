from sessions.clustering import rodar
from sessions.clustering.visualization import gerar_graficos
from sessions.clustering.metrics import avaliar_clusters
from sessions.clustering.feature_analysis import analisar_features

clustering_results = rodar()

gerar_graficos(clustering_results)

avaliar_clusters(None, clustering_results)

analisar_features(
    None,
    clustering_results[0]["labels"]  # K=2
)