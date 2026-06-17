from sessions.anomaly_detection.autoencoder3camadas import rodar as rodar_autoencoder

rodar_autoencoder()


# ============================================================
#                        NOTAS
# ============================================================
#
# Este arquivo orquestra todo o pipeline de clustering e detecção
# de anomalias do projeto. A ordem de execução foi definida para
# que cada etapa aproveite os resultados da anterior:
#
# 1. rodar() - Treina K-Means com K=2 a 10 nos PDFs benignos.
#    Retorna clustering_results com os modelos e labels.
#
# 2. gerar_graficos() - Plota silhouette e elbow method para
#    justificar a escolha de K=2.
#
# 3. gerar_graficos_pca() - Projeta os dados em 2D com PCA para
#    visualizar a estrutura dos clusters e onde os PDFs maliciosos
#    do conjunto de teste se localizam nesse espaço.
#
# 4. gerar_graficos_tsne() - Visualização não-linear com t-SNE.
#    Mais lento, mas captura melhor a estrutura local dos clusters.
#
# 5. avaliar_clusters() - Calcula silhouette, Davies-Bouldin e
#    Calinski-Harabasz para todos os valores de K, complementando
#    a análise visual dos gráficos.
#
# 6. analisar_features() - Analisa as médias de cada feature por
#    cluster (K=2), identificando o que diferencia os dois grupos
#    de PDFs benignos.
#
# 7. rodar_anomaly_kmeans() - Usa o K-Means (K=2) como detector:
#    a distância ao centroide vira um score de anomalia. Avalia
#    com AUC-ROC, precision, recall e F1 no conjunto de teste.
#
# 8. rodar_dbscan() - Treina DBSCAN nos PDFs benignos, extrai os
#    core points e usa distância a eles como score de anomalia.
#    Testa múltiplos valores de eps e avalia no conjunto de teste.
#
# 9. rodar_autoencoder() - Treina um Autoencoder (rede neural encoder-
#    decoder) nos PDFs benignos com Early Stopping. Usa o erro de
#    reconstrução (MSE) como score de anomalia. Avalia com AUC-ROC,
#    precision, recall e F1 no conjunto de teste.
