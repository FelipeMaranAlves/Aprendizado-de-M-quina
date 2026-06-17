import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve, confusion_matrix
from utils import documentar

def _preparar_dados():
    df = pd.read_csv("data/PDF_All_feature_Clean.csv")
    if 'file_path' in df.columns:
        df.drop(columns='file_path', inplace=True)
    return df

def _plot_confusion_matrix_heatmap(y_true, y_pred, filepath):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    
    group_counts = [f'{value:.0f}' for value in cm.ravel()]
    group_percentages = [f'{value*100:.2f}%' for value in cm.ravel()/np.sum(cm)]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.array(labels).reshape(2,2)
    
    sns.heatmap(
        cm, 
        annot=labels, 
        cmap='Blues', 
        xticklabels=['Previsto Benigno', 'Previsto Malicioso'], 
        yticklabels=['Real Benigno', 'Real Malicioso'], 
        fmt='',
        annot_kws={"size": 14}
    )
    plt.title('Matriz de Confusão (Isolation Forest) - Teste', fontsize=14)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def rodar():
    os.makedirs("Images", exist_ok=True)
    df = _preparar_dados()

    df_benign = df.query("label == 0")
    RAND_STATE = 2
    
    # 1. Divisão Estrita (Treino em Benignos, Val/Teste mistos)
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
    
    # 3. Treinamento do Isolation Forest nos dados Benignos
    isf = IsolationForest(random_state=RAND_STATE, n_jobs=-1)
    isf.fit(X_train_norm)
    
    # 4. Extração de Scores na VALIDAÇÃO (Invertendo para que MAIOR = MAIS ANÔMALO)
    scores_val = isf.decision_function(X_val_norm) * -1
    
    # --- Calibrar Threshold por F1-Score ---
    thresholds_f1 = np.percentile(scores_val, np.arange(10, 95, 5))
    best_f1, best_threshold_f1 = 0.0, thresholds_f1[0]
    for t in thresholds_f1:
        preds = (scores_val > t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold_f1 = t
            
    # --- Calibrar Threshold por Youden Index ---
    fpr, tpr, thresholds_roc = roc_curve(y_val, scores_val)
    youden_index = tpr - fpr
    best_threshold_youden = thresholds_roc[np.argmax(youden_index)]
    
    print("\n[LIMIARES DO ISF NA VALIDAÇÃO]")
    print(f" -> Threshold F1-Score: {best_threshold_f1:.4f}")
    print(f" -> Threshold Youden:   {best_threshold_youden:.4f}\n")
            
    # 5. Avaliação Final no Conjunto de TESTE
    scores_test = isf.decision_function(X_test_norm) * -1
    df_scores = pd.DataFrame({'score': scores_test, 'label': y_test})
    
    # -------------------------------------------------------------
    # GRÁFICO 1: Histograma de Distribuição de Scores do ISF
    # -------------------------------------------------------------
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
    plt.title("Diagnóstico: Distribuição de Scores de Anomalia (Isolation Forest)", fontsize=14)
    plt.xlabel("Score de Anomalia (Path Length Invertido - Quanto maior, mais suspeito)")
    plt.ylabel("Quantidade de PDFs")
    
    # Linhas dos limiares
    plt.axvline(x=best_threshold_youden, color='black', linestyle='-', linewidth=2, label=f'Threshold Youden: {best_threshold_youden:.2f}')
    plt.axvline(x=best_threshold_f1, color='purple', linestyle='--', linewidth=2, label=f'Threshold F1: {best_threshold_f1:.2f}')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig("Images/diagnostico_scores_isf.png")
    plt.close()
    print("Gráfico gerado: Images/diagnostico_scores_isf.png")

    # -------------------------------------------------------------
    # GRÁFICO 2: Matriz de Confusão Visuall (Heatmap)
    # -------------------------------------------------------------
    preds_test = (scores_test > best_threshold_youden).astype(int)
    caminho_heatmap = "Images/diagnostico_heatmap_confusao_isf.png"
    _plot_confusion_matrix_heatmap(y_test, preds_test, caminho_heatmap)
    print(f"Heatmap gerado: {caminho_heatmap}")

    # Documentação das métricas textuais
    doc_texto = (
        "============================================================\n"
        "       DIAGNÓSTICO VISUAL - ISOLATION FOREST (ISF)\n"
        "============================================================\n\n"
        "1. CALIBRAÇÃO DE LIMITES NA VALIDAÇÃO:\n"
        f"  - Otimização via F1-Score:  {best_threshold_f1:.4f}\n"
        f"  - Otimização via Youden:    {best_threshold_youden:.4f}\n\n"
        "2. ANÁLISE COMPORTAMENTAL DO MODELO:\n"
        "Como o Isolation Forest isola pontos através de divisões no espaço amostral,\n"
        "o gráfico de scores mostra se os malwares (vermelho) foram isolados com poucos\n"
        "cortes (scores altos à direita) em comparação com a estrutura compacta benigna.\n"
    )
    documentar("Diagnostico_Modelo_ISF", doc_texto)