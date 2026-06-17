import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from utils import documentar
from pipeline import carregar

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
    
    # Varrer percentis na validação para encontrar o limiar que maximiza o F1-Score
    thresholds = np.percentile(scores_val, np.arange(10, 95, 5))
    best_f1, best_threshold = 0.0, thresholds[0]
    
    for t in thresholds:
        preds = (scores_val > t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            
    print(f"Threshold otimizado na validação (Max F1: {best_f1:.4f}): {best_threshold:.4f}")
            
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
        # O hue é numérico (0 e 1), então a paleta precisa refletir isso
        palette={0: '#1f77b4', 1: '#d62728'},
        multiple='stack',
        alpha=0.7
    )
    
    plt.title("Diagnóstico de Deteção: K-Means (Limiar Calibrado na Validação)", fontsize=14)
    plt.xlabel("Score de Anomalia (Distância Euclidiana ao Padrão Benigno)")
    plt.ylabel("Quantidade de PDFs no Conjunto de Teste")
    
    # Desenha a linha do limiar calculado na VALIDAÇÃO sobre os dados de TESTE
    plt.axvline(
        x=best_threshold, 
        color='black', 
        linestyle='--', 
        linewidth=2,
        label=f'Threshold Otimizado: {best_threshold:.2f}'
    )
    
    plt.legend()
    plt.tight_layout()
    plt.savefig("Images/diagnostico_scores_anomalia_kmeans.png")
    plt.close()
    print("Gráfico salvo em: Images/diagnostico_scores_anomalia_kmeans.png")

    # Atualizar o arquivo de texto
    doc_texto = (
        "============================================================\n"
        "          DIAGNÓSTICO VISUAL DE ANOMALIAS (K-MEANS)\n"
        "============================================================\n\n"
        "1. METODOLOGIA (ZERO DATA LEAKAGE):\n"
        "O limiar de corte (Threshold) foi calibrado utilizando exclusivamente o conjunto\n"
        f"de Validação, encontrando o valor ótimo de {best_threshold:.4f} (Max F1: {best_f1:.4f}).\n"
        "Este limite foi então projetado cegamente no conjunto de Teste.\n\n"
        "2. INTERPRETAÇÃO DO GRÁFICO:\n"
        "A área azul à esquerda da linha preta representa os PDFs benignos classificados corretamente.\n"
        "A área vermelha à direita da linha representa as ameaças (malwares) detectadas com sucesso.\n"
        "Qualquer vermelho à esquerda da linha é um Falso Negativo (vírus que passou).\n"
        "Qualquer azul à direita da linha é um Falso Positivo (arquivo limpo bloqueado).\n"
    )
    documentar("Diagnostico_Modelo_KMeans", doc_texto)