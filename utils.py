##Esse codigo assume uma arquitetura do tipo, e que vc ta rodando na main
#.
#\Documentation
from os import path, makedirs
from math import log2
import pandas as pd
def documentar(titulo: str, conteudo: str):
    '''Exemplos de uso: \n
     documentar("testeTitulo", "testeConteudo") \n
     documentar("testePandas",str(df.head(5)))'''
    makedirs("./Documentation", exist_ok=True)

    if (path.isfile(f"./Documentation/{titulo}.txt")):
        with open(f"./Documentation/{titulo}.txt", "a") as f:
            f.write(f"\n{conteudo}")
    else:
        with open(f"./Documentation/{titulo}.txt", "w", encoding="utf-8") as f:
            f.write(conteudo)

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