import re
import ast
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def extrair_dados(arquivo):
    """
    Extrai todos os resultados do arquivo de documentação.
    Suporta tanto o formato com features_selected quanto sem.
    """
    with open(arquivo, 'r', encoding='utf-8') as f:
        conteudo = f.read()
    
    # Padrão COM features_selected (IsolationForestComFS)
    padrao_com_fs = re.compile(
        r'contamination=([\d.]+)\s*\|\s*'
        r'n_estimators=(\d+)\s*\|\s*'
        r'max_samples=([\d.]+)\s*\|\s*'
        r'max_features=([\d.]+)\s*\|\s*'
        r'features_selected=(\d+)\s*\n'
        r'(\{.*?\})',
        re.DOTALL
    )
    
    # Padrão SEM features_selected (IsolationForest2)
    padrao_sem_fs = re.compile(
        r'contamination=([\d.]+)\s*\|\s*'
        r'n_estimators=(\d+)\s*\|\s*'
        r'max_samples=([\d.]+)\s*\|\s*'
        r'max_features=([\d.]+)\s*\n'
        r'(\{.*?\})',
        re.DOTALL
    )
    
    dados = []
    
    # Tenta com features_selected primeiro
    for match in padrao_com_fs.finditer(conteudo):
        try:
            params = {
                'contamination': float(match.group(1)),
                'n_estimators': int(match.group(2)),
                'max_samples': float(match.group(3)),
                'max_features': float(match.group(4)),
                'features_selected': int(match.group(5))
            }
            dict_str = match.group(6).replace('np.float64(', '').replace(')', '')
            metricas = ast.literal_eval(dict_str)
            dados.append({**params, **metricas})
        except Exception as e:
            continue
    
    # Se não encontrou, tenta sem features_selected
    if not dados:
        for match in padrao_sem_fs.finditer(conteudo):
            try:
                params = {
                    'contamination': float(match.group(1)),
                    'n_estimators': int(match.group(2)),
                    'max_samples': float(match.group(3)),
                    'max_features': float(match.group(4))
                }
                dict_str = match.group(5).replace('np.float64(', '').replace(')', '')
                metricas = ast.literal_eval(dict_str)
                dados.append({**params, **metricas})
            except Exception as e:
                continue
    
    return dados

def gerar_graficos_comparativos():
    """Gera gráficos comparando IsolationForest com e sem Feature Selection"""
    
    # Cria pasta de destino
    pasta_destino = Path("Images/isolation_forest")
    pasta_destino.mkdir(parents=True, exist_ok=True)
    
    # ============================================
    # DEFINIÇÃO DOS EXPERIMENTOS
    # ============================================
    # Experimento 1: SEM Feature Selection (IsolationForest2)
    arquivo_sem_fs = Path("Documentation/isolation_forest/IsolationForest2.txt")
    
    # Experimento 2: COM Feature Selection (IsolationForestComFS)
    arquivo_com_fs = Path("Documentation/isolation_forest/IsolationForestComFS.txt")
    
    print("="*70)
    print("GRÁFICOS COMPARATIVOS - ISOLATION FOREST")
    print("="*70)
    print("\n📁 VERIFICANDO ARQUIVOS:")
    print(f"  Sem FS: {arquivo_sem_fs} ({arquivo_sem_fs.stat().st_size if arquivo_sem_fs.exists() else 'N/A'} bytes)")
    print(f"  Com FS: {arquivo_com_fs} ({arquivo_com_fs.stat().st_size if arquivo_com_fs.exists() else 'N/A'} bytes)")
    
    # Verifica se os arquivos existem
    if not arquivo_sem_fs.exists():
        print(f"\n❌ Arquivo {arquivo_sem_fs} não encontrado!")
        return
    
    if not arquivo_com_fs.exists():
        print(f"\n❌ Arquivo {arquivo_com_fs} não encontrado!")
        return
    
    # Extrai dados
    print("\n📊 EXTRAINDO DADOS:")
    dados_sem_fs = extrair_dados(arquivo_sem_fs)
    dados_com_fs = extrair_dados(arquivo_com_fs)
    
    print(f"  Sem FS: {len(dados_sem_fs)} combinações")
    print(f"  Com FS: {len(dados_com_fs)} combinações")
    
    if not dados_sem_fs or not dados_com_fs:
        print("❌ Nenhum dado encontrado em um dos arquivos!")
        return
    
    # ============================================
    # PREPARAÇÃO DOS DADOS
    # ============================================
    def criar_arrays(dados, nome):
        """Cria arrays numpy a partir dos dados"""
        return {
            'nome': nome,
            'n_estimators': np.array([d['n_estimators'] for d in dados]),
            'max_samples': np.array([d['max_samples'] for d in dados]),
            'max_features': np.array([d['max_features'] for d in dados]),
            'acc': np.array([d.get('acc', 0) for d in dados]),
            'precision': np.array([d.get('precision', 0) for d in dados]),
            'tpr': np.array([d.get('tpr', 0) for d in dados]),
            'fpr': np.array([d.get('fpr', 0) for d in dados]),
            'f1': np.array([d.get('f1-score', 0) for d in dados]),
            'f05': np.array([d.get('f0.5-score', 0) for d in dados]),
            'features_selected': np.array([d.get('features_selected', None) for d in dados])
        }
    
    sem = criar_arrays(dados_sem_fs, "Sem Feature Selection")
    com = criar_arrays(dados_com_fs, "Com Feature Selection")
    
    # Verifica quais métricas estão disponíveis
    tem_f05_sem = np.any(sem['f05'] > 0)
    tem_f05_com = np.any(com['f05'] > 0)
    
    print(f"\n📈 MÉTRICAS DISPONÍVEIS:")
    print(f"  F1-score: Sem FS ✅ | Com FS ✅")
    print(f"  F0.5-score: Sem FS {'✅' if tem_f05_sem else '❌'} | Com FS {'✅' if tem_f05_com else '❌'}")
    
    # ============================================
    # GRÁFICO 1: Boxplot - F1-score
    # ============================================
    print("\n🎨 GERANDO GRÁFICOS:")
    
    plt.figure(figsize=(8, 6))
    bp = plt.boxplot([sem['f1'], com['f1']], patch_artist=True)
    plt.gca().set_xticklabels(['Sem Feature Selection', 'Com Feature Selection'])
    
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('coral')
    
    plt.ylabel('F1-score')
    plt.title('Comparação de F1-score: Com vs Sem Feature Selection')
    plt.grid(True, alpha=0.3)
    plt.ylim(0.7, 0.95)
    plt.tight_layout()
    plt.savefig(pasta_destino / 'comparacao_f1_boxplot.png', dpi=300)
    plt.close()
    print("  ✅ comparacao_f1_boxplot.png")
    
    # ============================================
    # GRÁFICO 2: Boxplot - F0.5-score (se disponível)
    # ============================================
    if tem_f05_sem and tem_f05_com:
        plt.figure(figsize=(8, 6))
        bp = plt.boxplot([sem['f05'], com['f05']], patch_artist=True)
        plt.gca().set_xticklabels(['Sem Feature Selection', 'Com Feature Selection'])
        
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][1].set_facecolor('coral')
        
        plt.ylabel('F0.5-score')
        plt.title('Comparação de F0.5-score: Com vs Sem Feature Selection')
        plt.grid(True, alpha=0.3)
        plt.ylim(0.7, 0.95)
        plt.tight_layout()
        plt.savefig(pasta_destino / 'comparacao_f05_boxplot.png', dpi=300)
        plt.close()
        print("  ✅ comparacao_f05_boxplot.png")
    else:
        print("  ⚠️ comparacao_f05_boxplot.png - Não gerado (dados F0.5 indisponíveis)")
    
    # ============================================
    # GRÁFICO 3: Boxplots - Todas as métricas
    # ============================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metricas = [
        ('Acurácia', 'acc', axes[0, 0]),
        ('Precisão', 'precision', axes[0, 1]),
        ('Recall (TPR)', 'tpr', axes[1, 0]),
        ('FPR', 'fpr', axes[1, 1])
    ]
    
    for titulo, chave, ax in metricas:
        dados_sem = sem[chave]
        dados_com = com[chave]
        
        bp = ax.boxplot([dados_sem, dados_com], patch_artist=True)
        ax.set_xticklabels(['Sem FS', 'Com FS'])
        
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][1].set_facecolor('coral')
        
        ax.set_ylabel(titulo)
        ax.set_title(f'Comparação: {titulo}')
        ax.grid(True, alpha=0.3)
        
        if chave != 'fpr':
            ax.set_ylim(0.7, 0.95)
        else:
            ax.set_ylim(0, 0.3)
    
    plt.tight_layout()
    plt.savefig(pasta_destino / 'comparacao_metricas_boxplots.png', dpi=300)
    plt.close()
    print("  ✅ comparacao_metricas_boxplots.png")
    
    # ============================================
    # GRÁFICO 4: Barras - Melhor modelo de cada experimento
    # ============================================
    # Encontra melhores por F1
    idx_sem_f1 = np.argmax(sem['f1'])
    idx_com_f1 = np.argmax(com['f1'])
    best_sem = dados_sem_fs[idx_sem_f1]
    best_com = dados_com_fs[idx_com_f1]
    
    # Encontra melhores por F0.5 (se disponível)
    if tem_f05_sem:
        idx_sem_f05 = np.argmax(sem['f05'])
        best_sem_f05 = dados_sem_fs[idx_sem_f05]
    else:
        best_sem_f05 = best_sem
    
    if tem_f05_com:
        idx_com_f05 = np.argmax(com['f05'])
        best_com_f05 = dados_com_fs[idx_com_f05]
    else:
        best_com_f05 = best_com
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Melhor F1
    metricas_f1 = ['Acurácia', 'Precisão', 'Recall', 'F1-score']
    valores_sem_f1 = [best_sem.get('acc', 0), best_sem.get('precision', 0), 
                      best_sem.get('tpr', 0), best_sem.get('f1-score', 0)]
    valores_com_f1 = [best_com.get('acc', 0), best_com.get('precision', 0), 
                      best_com.get('tpr', 0), best_com.get('f1-score', 0)]
    
    x = np.arange(len(metricas_f1))
    width = 0.35
    
    axes[0].bar(x - width/2, valores_sem_f1, width, label='Sem FS', color='steelblue')
    axes[0].bar(x + width/2, valores_com_f1, width, label='Com FS', color='coral')
    axes[0].set_xlabel('Métrica')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Melhor F1-score por Experimento')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metricas_f1)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0.7, 1.0)
    
    # Melhor F0.5 (se disponível)
    metricas_f05 = ['Acurácia', 'Precisão', 'Recall', 'F0.5-score']
    
    valores_sem_f05 = [best_sem_f05.get('acc', 0), best_sem_f05.get('precision', 0), 
                       best_sem_f05.get('tpr', 0), 
                       best_sem_f05.get('f0.5-score', 0) if tem_f05_sem else 0]
    valores_com_f05 = [best_com_f05.get('acc', 0), best_com_f05.get('precision', 0), 
                       best_com_f05.get('tpr', 0), 
                       best_com_f05.get('f0.5-score', 0) if tem_f05_com else 0]
    
    axes[1].bar(x - width/2, valores_sem_f05, width, label='Sem FS', color='steelblue')
    axes[1].bar(x + width/2, valores_com_f05, width, label='Com FS', color='coral')
    axes[1].set_xlabel('Métrica')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Melhor F0.5-score por Experimento')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metricas_f05)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.7, 1.0)
    
    # Se não tem F0.5, adiciona nota
    if not tem_f05_sem or not tem_f05_com:
        axes[1].text(0.5, 0.72, '⚠️ F0.5-score não disponível\nno experimento Com FS', 
                    transform=axes[1].transAxes, ha='center', va='bottom',
                    fontsize=10, color='red', style='italic')
    
    plt.tight_layout()
    plt.savefig(pasta_destino / 'comparacao_melhores_barras.png', dpi=300)
    plt.close()
    print("  ✅ comparacao_melhores_barras.png")
    
    # ============================================
    # GRÁFICO 5: Scatter - Precisão vs Recall
    # ============================================
    plt.figure(figsize=(10, 6))
    
    # Sem FS (azul)
    scatter1 = plt.scatter(sem['tpr'], sem['precision'], 
                          c=sem['f1'], cmap='Blues', 
                          alpha=0.6, s=60, edgecolors='darkblue', 
                          linewidth=0.5, label='Sem FS')
    
    # Com FS (vermelho)
    scatter2 = plt.scatter(com['tpr'], com['precision'], 
                          c=com['f1'], cmap='Reds', 
                          alpha=0.6, s=60, edgecolors='darkred', 
                          linewidth=0.5, label='Com FS', marker='s')
    
    # Melhores pontos
    plt.scatter(best_sem['tpr'], best_sem['precision'], 
               color='darkblue', s=300, marker='*', 
               label=f'Melhor Sem FS (F1={best_sem["f1-score"]:.3f})')
    plt.scatter(best_com['tpr'], best_com['precision'], 
               color='darkred', s=300, marker='*', 
               label=f'Melhor Com FS (F1={best_com["f1-score"]:.3f})')
    
    plt.xlabel('Recall (TPR)')
    plt.ylabel('Precisão')
    plt.title('Precisão vs Recall: Comparação Sem vs Com Feature Selection')
    plt.grid(True, alpha=0.3)
    plt.xlim(0.7, 0.95)
    plt.ylim(0.7, 0.95)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(pasta_destino / 'comparacao_precision_recall_scatter.png', dpi=300)
    plt.close()
    print("  ✅ comparacao_precision_recall_scatter.png")
    
    # ============================================
    # GRÁFICO 6: Radar - Melhores modelos
    # ============================================
    categories = ['Acurácia', 'Precisão', 'Recall', 'F1-score', 'F0.5-score']
    
    # Valores do melhor F1
    values_sem = [best_sem.get('acc', 0), best_sem.get('precision', 0), 
                  best_sem.get('tpr', 0), best_sem.get('f1-score', 0), 
                  best_sem.get('f0.5-score', 0) if tem_f05_sem else best_sem.get('f1-score', 0)]
    values_com = [best_com.get('acc', 0), best_com.get('precision', 0),
                  best_com.get('tpr', 0), best_com.get('f1-score', 0),
                  best_com.get('f0.5-score', 0) if tem_f05_com else best_com.get('f1-score', 0)]
    
    # Fecha o círculo
    values_sem += values_sem[:1]
    values_com += values_com[:1]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    ax.plot(angles, values_sem, 'o-', linewidth=2, label='Sem FS (Melhor F1)', 
            color='steelblue')
    ax.fill(angles, values_sem, alpha=0.25, color='steelblue')
    
    ax.plot(angles, values_com, 's-', linewidth=2, label='Com FS (Melhor F1)', 
            color='coral')
    ax.fill(angles, values_com, alpha=0.25, color='coral')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0.7, 1.0)
    ax.set_title('Radar - Melhores Modelos por Experimento', size=14)
    
    # Se não tem F0.5, adiciona nota
    if not tem_f05_sem or not tem_f05_com:
        ax.text(0.5, 0.72, '⚠️ F0.5-score não disponível\nno experimento Com FS', 
               transform=ax.transAxes, ha='center', va='bottom',
               fontsize=10, color='red', style='italic')
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(pasta_destino / 'comparacao_radar.png', dpi=300)
    plt.close()
    print("  ✅ comparacao_radar.png")
    
    # ============================================
    # RESUMO FINAL
    # ============================================
    print("\n" + "="*70)
    print("📊 RESUMO DOS RESULTADOS")
    print("="*70)
    
    print(f"\n🔵 SEM FEATURE SELECTION:")
    print(f"   Melhor F1-score: {best_sem['f1-score']:.6f}")
    print(f"   Parâmetros: n_estimators={best_sem['n_estimators']}, "
          f"max_samples={best_sem['max_samples']}, "
          f"max_features={best_sem['max_features']}")
    if tem_f05_sem:
        print(f"   Melhor F0.5-score: {best_sem_f05.get('f0.5-score', 0):.6f}")
    
    print(f"\n🔴 COM FEATURE SELECTION:")
    print(f"   Melhor F1-score: {best_com['f1-score']:.6f}")
    print(f"   Parâmetros: n_estimators={best_com['n_estimators']}, "
          f"max_samples={best_com['max_samples']}, "
          f"max_features={best_com['max_features']}")
    if 'features_selected' in best_com:
        print(f"   Features selecionadas: {best_com['features_selected']}")
    if tem_f05_com:
        print(f"   Melhor F0.5-score: {best_com_f05.get('f0.5-score', 0):.6f}")
    
    print("\n" + "="*70)
    print("📁 GRÁFICOS GERADOS EM: Images/isolation_forest/")
    print("="*70)
    print("  1. comparacao_f1_boxplot.png")
    if tem_f05_sem and tem_f05_com:
        print("  2. comparacao_f05_boxplot.png")
    else:
        print("  2. comparacao_f05_boxplot.png ⚠️ (não gerado - dados F0.5 indisponíveis)")
    print("  3. comparacao_metricas_boxplots.png")
    print("  4. comparacao_melhores_barras.png")
    print("  5. comparacao_precision_recall_scatter.png")
    print("  6. comparacao_radar.png")
    print("="*70)
    
    print("\n✅ CONCLUÍDO!")


if __name__ == "__main__":
    gerar_graficos_comparativos()