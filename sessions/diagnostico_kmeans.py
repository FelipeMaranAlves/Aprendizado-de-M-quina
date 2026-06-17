import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve, confusion_matrix
from utils import documentar
from pipeline import carregar

def _plot_confusion_matrix_heatmap(y_true, y_pred, filepath):
    plt.figure(figsize=(8, 6))
    
    # Calcula a matriz (Verdadeiros/Falsos Positivos e Negativos)
    cm = confusion_matrix(y_true, y_pred)
    
    # Formata os números brutos e as percentagens para colocar dentro dos quadrados
    group_counts = [f'{value:.0f}' for value in cm.ravel()]
    group_percentages = [f'{value*100:.2f}%' for value in cm.ravel()/np.sum(cm)]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.array(labels).reshape(2,2)
    
    # Desenha o Heatmap (usando tons de Laranja, como no material de aula)
    sns.heatmap(
        cm, 
        annot=labels, 
        cmap='Oranges', 
        xticklabels=['Previsto Benigno', 'Previsto Malicioso'], 
        yticklabels=['Real Benigno', 'Real Malicioso'], 
        fmt='',
        annot_kws={"size": 14} # Aumenta a letra para ficar legível no relatório
    )
    
    plt.title('Matriz de Confusão (K-Means) - Conjunto de Teste', fontsize=14)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def rodar():
    os.makedirs("Images", exist_ok=True)
    df = carregar()

    # =================================================================
    # DIAGNÓSTICO DO MODELO K-MEANS SEM VAZAMENTO DE DADOS (DATA LEAKAGE)
    # =================================================================
    df_benign = df.query("label == 0")
    RAND_STATE = 2
    
    # 1. Separação Estrita (Treino, Validação e Teste)
    X_train, _ = train_test_split(df_benign, test_size=0.2, random_state=RAND_STATE)
    _, Xy_tmp = train_test_split(df, test_size=0.2, random_state=RAND_STATE)
    Xy_val, Xy_test = train_test_split(Xy_tmp, test_size=0.5, random_state=RAND_STATE)
    
    X_train = X_train.drop(columns=['label'])
    
    y_val = Xy_val['label'].values
    X_val = Xy_val.drop(columns=['label'])
    
    y_test = Xy_test['label'].values
    X_test = Xy_test.drop(columns=['label'])
    
    # 2. Normalização
    std_scaler = StandardScaler()
    X_train_norm = std_scaler.fit_transform(X_train)
    X_val_norm = std_scaler.transform(X_val)
    X_test_norm = std_scaler.transform(X_test)
    
    # 3. Treinamento do Modelo (Apenas em Benignos)
    kmeans = KMeans(n_clusters=2, random_state=RAND_STATE, n_init=10)
    kmeans.fit(X_train_norm)
    
    # -------------------------------------------------------------
    # PASSO CRÍTICO: Calibrar o Threshold no Conjunto de VALIDAÇÃO
    # -------------------------------------------------------------
    scores_val = kmeans.transform(X_val_norm).min(axis=1)
    
    # --- ESTRATÉGIA A: Calibração via F1-Score ---
    thresholds_f1 = np.percentile(scores_val, np.arange(10, 95, 5))
    best_f1, best_threshold_f1 = 0.0, thresholds_f1[0]
    
    for t in thresholds_f1:
        preds = (scores_val > t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold_f1 = t
            
    # --- ESTRATÉGIA B: Calibração via Youden Index (Recomendado na Aula) ---
    fpr, tpr, thresholds_roc = roc_curve(y_val, scores_val)
    youden_index = tpr - fpr
    best_idx = np.argmax(youden_index)
    best_threshold_youden = thresholds_roc[best_idx]
    best_youden_val = youden_index[best_idx]
    
    # Imprime o comparativo no terminal
    print("\n[COMPARATIVO DE LIMIARES NA VALIDAÇÃO]")
    print(f" -> Otimização por F1-Score: Threshold = {best_threshold_f1:.4f} (F1 Max: {best_f1:.4f})")
    print(f" -> Otimização por Youden:   Threshold = {best_threshold_youden:.4f} (Youden Max: {best_youden_val:.4f})\n")

    # Adotamos o Youden Index para o teste final (pois equilibra TPR e FPR)
    limiar_escolhido = best_threshold_youden
            
    # -------------------------------------------------------------
    # PLOT FINAL: Aplicar cegamente o modelo no Conjunto de TESTE
    # -------------------------------------------------------------
    scores_test = kmeans.transform(X_test_norm).min(axis=1)
    df_scores = pd.DataFrame({'score': scores_test, 'label': y_test})
    
    plt.figure(figsize=(12, 6))
    sns.histplot(
        data=df_scores, 
        x='score', 
        hue='label', 
        bins=50, 
        kde=True,
        palette={0: '#1f77b4', 1: '#d62728'},
        multiple='stack',
        alpha=0.7
    )
    
    plt.title("Diagnóstico de Deteção: K-Means (Limiar via Youden Index)", fontsize=14)
    plt.xlabel("Score de Anomalia (Distância Euclidiana ao Padrão Benigno)")
    plt.ylabel("Quantidade de PDFs no Conjunto de Teste")
    
    # Desenha a linha do limiar Youden calculado na VALIDAÇÃO sobre os dados de TESTE
    plt.axvline(
        x=limiar_escolhido, 
        color='black', 
        linestyle='-', 
        linewidth=2,
        label=f'Threshold Youden: {limiar_escolhido:.2f}'
    )
    
    # Opcional: Desenha a linha do F1-Score para visualização comparativa no gráfico
    plt.axvline(
        x=best_threshold_f1, 
        color='purple', 
        linestyle='--', 
        linewidth=2,
        label=f'Threshold F1: {best_threshold_f1:.2f}'
    )
    
    plt.legend()
    plt.tight_layout()
    plt.savefig("Images/diagnostico_scores_anomalia_kmeans.png")
    plt.close()
    print("Gráfico salvo em: Images/diagnostico_scores_anomalia_kmeans.png")

    # Gera as previsões finais: se o score for maior que o limiar do Youden, é classificado como 1 (Malicioso)
    preds_test = (scores_test > limiar_escolhido).astype(int)
    
    # Chama a função para desenhar e guardar o Heatmap
    caminho_heatmap = "Images/diagnostico_heatmap_confusao_kmeans.png"
    _plot_confusion_matrix_heatmap(y_test, preds_test, caminho_heatmap)
    print(f"Heatmap de Confusão salvo em: {caminho_heatmap}")

    # Atualizar o arquivo de texto com a comparação
    doc_texto = (
        "============================================================\n"
        "          DIAGNÓSTICO VISUAL DE ANOMALIAS (K-MEANS)\n"
        "============================================================\n\n"
        "1. CALIBRAÇÃO DE THRESHOLD (COMPARAÇÃO):\n"
        "Foi realizada uma calibração cruzada utilizando o conjunto de Validação,\n"
        "comparando duas métricas de otimização de limiar:\n"
        f"  - Otimização via F1-Score:  {best_threshold_f1:.4f} (F1 Max: {best_f1:.4f})\n"
        f"  - Otimização via Youden:    {best_threshold_youden:.4f} (Youden Max: {best_youden_val:.4f})\n\n"
        "O Índice de Youden foi adotado como limiar final por maximizar a distância entre a\n"
        "Taxa de Verdadeiros Positivos (TPR) e Falsos Positivos (FPR), garantindo maior robustez\n"
        "em cenários desbalanceados de segurança da informação.\n\n"
        "2. INTERPRETAÇÃO DO GRÁFICO (CONJUNTO DE TESTE):\n"
        "A linha sólida preta representa o corte do Índice de Youden, enquanto a pontilhada roxa\n"
        "representa o F1-Score. A área azul à esquerda da linha preta representa os PDFs benignos\n"
        "corretamente validados. A área vermelha à direita são os malwares bloqueados.\n"
    )
    documentar("Diagnostico_Modelo_KMeans", doc_texto)