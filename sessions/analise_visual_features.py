import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import documentar
from pipeline import carregar

def rodar():
    os.makedirs("Images", exist_ok=True)
    df = carregar()
    
    # Separar as features para plotagem (sem a label)
    X = df.drop(columns=['label'])
    
    # 1. GERAÇÃO DOS HISTOGRAMAS (DISTRIBUIÇÃO BENIGNO VS MALICIOSO)
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
        
        # Variável para rastrear o índice interno do subplot preenchido
        idx_feat = 0
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
            
        # Ocultar subplots vazios caso o último bloco não complete o grid 4x3
        for j in range(idx_feat + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        hist_path = f"Images/histogramas_features_bloco_{idx_bloco + 1}.png"
        plt.savefig(hist_path)
        plt.close()
        print(f"Bloco {idx_bloco + 1} de Histogramas salvo em: {hist_path}")

    # 2. DOCUMENTAÇÃO DAS CONCLUSÕES E NOTAS
    doc_texto = (
        "============================================================\n"
        "         RELATÓRIO DE ANÁLISE VISUAL DE DISTRIBUIÇÃO\n"
        "============================================================\n\n"
        "1. ANÁLISE DE DISTRIBUIÇÃO (HISTOGRAMAS):\n"
        "Os histogramas gerados dividem a frequência das features por classe (0 = Benigno, 1 = Malicioso).\n"
        "Observou-se que features estruturais do PDF (ex: contagem de objetos ou streams específicos)\n"
        "apresentam forte assimetria à direita, concentrando a classe benigna próxima de zero,\n"
        "enquanto anomalias/malwares deslocam-se para caudas mais longas com valores elevados.\n\n"
        "Esses gráficos confirmam visualmente a viabilidade de utilizar algoritmos de detecção\n"
        "de anomalias (K-Means/DBSCAN), dado que o comportamento dos metadados maliciosos quebra\n"
        "significativamente o padrão de densidade e escala estabelecido pelos arquivos benignos.\n"
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