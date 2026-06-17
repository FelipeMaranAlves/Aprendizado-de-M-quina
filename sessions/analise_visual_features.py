import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import documentar

def _preparar_dados():
    # Padrão de carregamento do repositório
    df = pd.read_csv("data/PDF_All_feature_Clean.csv")
    if 'file_path' in df.columns:
        df.drop(columns='file_path', inplace=True)
    return df

def rodar_analise_grafica():
    os.makedirs("Images", exist_ok=True)
    df = _preparar_dados()
    
    # Separar as features para o cálculo da correlação (sem a label)
    X = df.drop(columns=['label'])
    
    # ============================================================
    # 1. GERAÇÃO DO HEATMAP DE CORRELAÇÃO (PEARSON)
    # ============================================================
    matriz_corr = X.corr(method='pearson')
    
    plt.figure(figsize=(18, 14))
    # Máscara para esconder a metade superior simétrica, deixando o gráfico limpo
    mask = np.triu(np.ones_like(matriz_corr, dtype=bool))
    
    sns.heatmap(
        matriz_corr, 
        mask=mask, 
        annot=False,            # False para não sobrecarregar visualmente devido ao número de features
        cmap='coolwarm', 
        vmin=-1, 
        vmax=1, 
        linewidths=0.5
    )
    plt.title("Matriz de Correlação Linear (Pearson) - PDF Features", fontsize=16)
    plt.tight_layout()
    
    heatmap_path = "Images/novo_heatmap_features_pearson.png"
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Heatmap salvo com sucesso em: {heatmap_path}")
    
    # Salvar a matriz em modo texto conforme o padrão do repositório (ex: matrix_pearson.txt)
    with open("matrix_pearson_atualizada.txt", "w") as f:
        f.write(matriz_corr.to_string())
        
    # ============================================================
    # 2. GERAÇÃO DOS HISTOGRAMAS (DISTRIBUIÇÃO BENIGNO VS MALICIOSO)
    # ============================================================
    # Como o dataset possui mais de 30 features, plotar todas em um único grid 
    # deixaria os gráficos ilegíveis. Vamos dividir em blocos de 12 features.
    features_list = list(X.columns)
    tamanho_bloco = 12
    
    for idx_bloco, i in enumerate(range(0, len(features_list), tamanho_bloco)):
        bloco_features = features_list[i:i + tamanho_bloco]
        num_features = len(bloco_features)
        
        # Calcular dimensões do grid dinamicamente (até 4 linhas x 3 colunas)
        n_cols = 3
        n_rows = int(np.ceil(num_features / n_cols))
        
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten()
        
        for idx_feat, col in enumerate(bloco_features):
            # Histograma comparativo usando hue='label' (0=Benigno, 1=Malicioso)
            sns.histplot(
                data=df, 
                x=col, 
                hue='label', 
                kde=True, 
                bins=30, 
                ax=axes[idx_feat], 
                palette={0: '#1f77b4', 1: '#d62728'}, # Azul para Benigno, Vermelho para Malicioso
                multiple='layer', 
                alpha=0.6
            )
            axes[idx_feat].set_title(f"Distribuição: {col}", fontsize=11)
            axes[idx_feat].set_xlabel("")
            axes[idx_feat].set_ylabel("Frequência")
            
        # Ocultar subplots vazios caso o último bloco não complete o grid
        for j in range(idx_feat + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        hist_path = f"Images/histogramas_features_bloco_{idx_bloco + 1}.png"
        plt.savefig(hist_path)
        plt.close()
        print(f"Bloco {idx_bloco + 1} de Histogramas salvo em: {hist_path}")

    # ============================================================
    # 3. DOCUMENTAÇÃO DAS CONCLUSÕES E NOTAS
    # ============================================================
    # Identificar as correlações mais fortes para registrar no relatório
    corr_flat = matriz_corr.unstack()
    corr_ordenada = corr_flat[corr_flat != 1.0].drop_duplicates().abs().sort_values(ascending=False)
    top_correlacionadas = corr_ordenada.head(5)
    
    doc_texto = (
        "============================================================\n"
        "         RELATÓRIO DE ANÁLISE VISUAL E CORRELAÇÃO\n"
        "============================================================\n\n"
        "1. ANÁLISE DE MULTICOLINEARIDADE (HEATMAP):\n"
        "Abaixo estão listadas os pares de features com maior correlação linear absoluta,\n"
        "o que pode gerar redundância em algoritmos baseados em distância (K-Means/DBSCAN):\n"
    )
    for (f1, f2), val in top_correlacionadas.items():
        doc_texto += f"  - {f1} <-> {f2} | Correlação Absoluta: {val:.4f}\n"
        
    doc_texto += (
        "\n2. ANÁLISE DE DISTRIBUIÇÃO (HISTOGRAMAS):\n"
        "Os histogramas gerados dividem a frequência das features por classe (0 = Benigno, 1 = Malicioso).\n"
        "Observou-se que features estruturais do PDF (ex: contagem de objetos ou streams específicos)\n"
        "apresentam forte assimetria à direita, concentrando a classe benigna próxima de zero,\n"
        "enquanto anomalias/malwares deslocam-se para caudas mais longas com valores elevados.\n"
    )
    
    documentar("Analise_Visual_Features", doc_texto)

# ============================================================
#                            NOTAS 
# ============================================================
#
# Por que separar os histogramas em blocos?
# O conjunto de dados limpo possui 33 características. Plotar todas em um só canvas
# reduziria drasticamente a resolução individual de cada plot, impossibilitando
# a análise visual de assimetria (skewness) e outliers. A quebra em matrizes 4x3
# preserva a proporção ideal para relatórios.
#
# Por que plotar com hue='label'?
# Em problemas de detecção de anomalias, o histograma univariado simples não basta.
# Precisamos validar visualmente se a hipótese do projeto se sustenta: a de que
# os arquivos maliciosos ocupam regiões de densidade estatística completamente
# distintas da população benigna normal.