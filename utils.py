##Esse codigo assume uma arquitetura do tipo, e que vc ta rodando na main
#.
#\Documentation\{categoria}
from os import path, makedirs
from math import log2
import pandas as pd
def documentar(titulo: str, conteudo: str, categoria: str = "geral"):
    '''Exemplos de uso: \n
     documentar("testeTitulo", "testeConteudo", categoria="clustering") \n
     documentar("testePandas",str(df.head(5)), categoria="feature_selection")'''
    pasta = path.join("Documentation", categoria)
    makedirs(pasta, exist_ok=True)
    arquivo = path.join(pasta, f"{titulo}.txt")

    if path.isfile(arquivo):
        with open(arquivo, "a") as f:
            f.write(f"\n{conteudo}")
    else:
        with open(arquivo, "w", encoding="utf-8") as f:
            f.write(conteudo)

def caminho_imagem(nome_arquivo: str, categoria: str = "geral") -> str:
    '''Garante a pasta Images/{categoria} e retorna o caminho completo para usar em plt.savefig(...).'''
    pasta = path.join("Images", categoria)
    makedirs(pasta, exist_ok=True)
    return path.join(pasta, nome_arquivo)

def configurar():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)

def calculate_entropy(text):
    """Compute Shannon entropy of text content."""
    if not text:
        return 0
    freq = [0] * 256
    for char in text:
        freq[ord(char) % 256] += 1
    freq = [p / len(text) for p in freq if p > 0]
    return -sum(p * log2(p) for p in freq)