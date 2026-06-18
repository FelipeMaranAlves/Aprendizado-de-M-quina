import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from utils import documentar

def _preparar_dados_estritos():
    df = pd.read_csv("data/PDF_All_feature_Clean.csv")
    if 'file_path' in df.columns:
        df.drop(columns='file_path', inplace=True)
        
    df_benign = df.query("label == 0")
    RAND_STATE = 2
    
    # Separação idêntica para garantir que o Teste é 100% igual para todos os modelos
    X_train, _ = train_test_split(df_benign, test_size=0.2, random_state=RAND_STATE)
    _, Xy_tmp = train_test_split(df, test_size=0.2, random_state=RAND_STATE)
    Xy_val, Xy_test = train_test_split(Xy_tmp, test_size=0.5, random_state=RAND_STATE)
    
    X_train = X_train.drop(columns=['label'])
    
    y_val = Xy_val['label'].values
    X_val = Xy_val.drop(columns=['label'])
    
    y_test = Xy_test['label'].values
    X_test = Xy_test.drop(columns=['label'])
    
    return X_train, X_val, y_val, X_test, y_test

def rodar():
    os.makedirs("Images", exist_ok=True)
    X_train, X_val, y_val, X_test, y_test = _preparar_dados_estritos()
    RAND_STATE = 2

    # Normalização dos dados
    std_scaler = StandardScaler()
    X_train_norm = std_scaler.fit_transform(X_train)
    X_val_norm = std_scaler.transform(X_val)
    X_test_norm = std_scaler.transform(X_test)

    # =================================================================
    # 1. MODELO K-MEANS
    # =================================================================
    kmeans = KMeans(n_clusters=2, random_state=RAND_STATE, n_init=10)
    kmeans.fit(X_train_norm)
    scores_kmeans = kmeans.transform(X_test_norm).min(axis=1)
    
    fpr_km, tpr_km, _ = roc_curve(y_test, scores_kmeans)
    auc_km = roc_auc_score(y_test, scores_kmeans)

    # =================================================================
    # 2. MODELO ISOLATION FOREST (ISF)
    # =================================================================
    isf = IsolationForest(random_state=RAND_STATE, n_jobs=-1)
    isf.fit(X_train_norm)
    
    # Inverter o score do ISF para que MAIOR = MAIS ANÓMALO
    scores_isf_brutos = isf.decision_function(X_test_norm)
    scores_isf = scores_isf_brutos * -1 
    
    fpr_isf, tpr_isf, _ = roc_curve(y_test, scores_isf)
    auc_isf = roc_auc_score(y_test, scores_isf)

    # =================================================================
    # 3. MODELO DBSCAN (Treino, Validação para eps e Teste)
    # =================================================================
    MIN_SAMPLES = 5
    EPS_VALUES = [0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
    best_auc_val = -1.0
    best_nn = None
    best_eps = None

    for eps in EPS_VALUES:
        dbscan = DBSCAN(eps=eps, min_samples=MIN_SAMPLES, algorithm='ball_tree', n_jobs=-1)
        dbscan.fit(X_train_norm)

        # Se não houver core points, o modelo não formou clusters úteis
        if len(dbscan.core_sample_indices_) == 0:
            continue

        X_core = X_train_norm[dbscan.core_sample_indices_]
        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        nn.fit(X_core)

        # Avaliar na validação para escolher o melhor eps
        dist_val, _ = nn.kneighbors(X_val_norm)
        auc_val = roc_auc_score(y_val, dist_val.flatten())

        if auc_val > best_auc_val:
            best_auc_val = auc_val
            best_eps = eps
            best_nn = nn

    # Com o melhor modelo DBSCAN selecionado, extraímos o score de teste
    dist_test, _ = best_nn.kneighbors(X_test_norm)
    scores_dbscan = dist_test.flatten()
    
    fpr_db, tpr_db, _ = roc_curve(y_test, scores_dbscan)
    auc_db = roc_auc_score(y_test, scores_dbscan)

    # =================================================================
    # 4. PLOTAGEM DA CURVA ROC COMPARATIVA
    # =================================================================
    plt.figure(figsize=(10, 8))
    
    # Linha do K-Means (Azul)
    plt.plot(fpr_km, tpr_km, color='#1f77b4', lw=2, label=f'K-Means (AUC = {auc_km:.4f})')
    
    # Linha do DBSCAN (Verde)
    plt.plot(fpr_db, tpr_db, color='#2ca02c', lw=2, label=f'DBSCAN (AUC = {auc_db:.4f} | eps={best_eps})')
    
    # Linha do Isolation Forest (Laranja)
    plt.plot(fpr_isf, tpr_isf, color='#ff7f0e', lw=2, label=f'Isolation Forest (AUC = {auc_isf:.4f})')
    
    # Linha de Escolha Aleatória (Diagonal)
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', label='Aleatório (AUC = 0.5000)')
    
    # Configurações de design do gráfico
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('Taxa de Falsos Positivos (FPR) - "Alarmes Falsos"', fontsize=12)
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR) - "Malwares Detetados"', fontsize=12)
    plt.title('Curva ROC Comparativa: K-Means vs DBSCAN vs ISF', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    
    caminho_imagem = "Images/comparativo_roc_modelos.png"
    plt.tight_layout()
    plt.savefig(caminho_imagem)
    plt.close()
    print(f"Gráfico comparativo salvo em: {caminho_imagem}")

    # =================================================================
    # DOCUMENTAÇÃO
    # =================================================================
    resultados_auc = {
        "K-Means": auc_km,
        "DBSCAN": auc_db,
        "Isolation Forest": auc_isf
    }
    vencedor = max(resultados_auc, key=resultados_auc.get)
    
    doc_texto = (
        "============================================================\n"
        "       COMPARAÇÃO DE MODELOS - ANOMALY DETECTION\n"
        "============================================================\n\n"
        "Foi realizada uma avaliação frente a frente dos três modelos base utilizando o\n"
        "mesmo conjunto estrito de testes para garantir a ausência de viés metodológico.\n"
        "O parâmetro eps do DBSCAN foi afinado autonomamente via conjunto de validação.\n\n"
        "RESULTADOS (Área Sob a Curva ROC - AUC):\n"
        f"  - K-Means:          {auc_km:.4f}\n"
        f"  - DBSCAN:           {auc_db:.4f} (Melhor eps: {best_eps})\n"
        f"  - Isolation Forest: {auc_isf:.4f}\n\n"
        f"CONCLUSÃO:\n"
        f"O modelo {vencedor} apresentou a maior AUC, comprovando ser a arquitetura mais\n"
        "robusta para detetar ficheiros PDF maliciosos neste conjunto de dados.\n"
    )
    documentar("Comparativo_Modelos_ROC", doc_texto)