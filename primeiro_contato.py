import pandas as pd
from codigoprint import documentar
import io
from contextlib import redirect_stdout

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

df = pd.read_csv("data/PDF_All_features.csv")

# buffer = io.StringIO()

# with redirect_stdout(buffer):
#     df.info()

# saida = buffer.getvalue()

documentar('primeiro_contato',df.describe().to_string())