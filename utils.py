##Esse codigo assume uma arquitetura do tipo, e que vc ta rodando na main
#.
#\Documentation
from os import path, makedirs
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