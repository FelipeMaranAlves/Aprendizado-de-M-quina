import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import documentar, configurar, calculate_entropy
def rodar():
    configurar()

    df = pd.read_csv("data/PDF_All_features.csv")

    colunas=['file_path', 'file_size', 'title_chars', 'encrypted', 'metadata_size',
       'page_count', 'valid_pdf_header', 'image_count', 'text_length',
       'object_count', 'font_object_count', 'embedded_file_count',
       'average_embedded_file_size', 'stream_count', 'endstream_count',
       'average_stream_size', 'entropy_of_streams', 'xref_count',
       'xref_entries', 'name_obfuscations', 'total_filters',
       'nested_filter_objects', 'objstm_count', 'js_count', 'javascript_count',
       'uri_count', 'uses_nonstandard_port', 'action_count', 'aa_count',
       'openaction_count', 'launch_count', 'submitform_count',
       'acroform_count', 'xfa_count', 'jbig2decode_count', 'colors_count',
       'richmedia_count', 'trailer_count', 'startxref_count',
       'has_multiple_behavioral_keywords_in_one_object', 'used_ocr', 'label']
    # colunas_a_remover = "colunas a remover por que as features não tem variação no seu valor: \n"
    # for col in colunas:
    #     if(df[col].unique().__len__() == 1):
    #         colunas_a_remover += ", "
    #         colunas_a_remover += col
    # documentar('segundo_contato',colunas_a_remover)
    for i in range(df.shape[0]):
        if ("/content/drive/MyDrive/UNB/Mastercard project/Dataset/PDF_Both_Done/Benign/All_Files_10000/" in str(df.iloc[i,0])):
            df.iloc[i,0] = str(df.iloc[i,0]).removeprefix("/content/drive/MyDrive/UNB/Mastercard project/Dataset/PDF_Both_Done/Benign/All_Files_10000/").removesuffix(".pdf")
        if ("/content/drive/MyDrive/UNB/Mastercard project/Dataset/PDF_Both_Done/Malicious/CIC_malicious_pdfs_All/" in str(df.iloc[i,0])):
            df.iloc[i,0] = str(df.iloc[i,0]).removeprefix("/content/drive/MyDrive/UNB/Mastercard project/Dataset/PDF_Both_Done/Malicious/CIC_malicious_pdfs_All/").removesuffix(".pdf")
    df['title_entropy'] = df['file_path'].apply(calculate_entropy)
    sobre_entropia = "exemplo de como ficou a entropia baseada no titulo e sua relacao com label"
    sobre_entropia += str(df[['file_path','title_entropy','label']].head(20))
    documentar('segundo_contato',sobre_entropia)
    # df['title_entropy_with_lenght'] = df['title_entropy'].apply()
    # documentar('segundo_contato',)
    # for col in colunas:
    #     if(df[col].unique().__len__() == 1):
    #         df.drop(col,axis=1)
    # df.to_csv("data/PDF_All_feature_Clean.csv")

    # # Separar por classe
    # df_label_0 = df[df['label'] == 0]
    # df_label_1 = df[df['label'] == 1]

    # # Criar figura com 2 gráficos lado a lado
    # fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # # ---------------------------------
    # # Gráfico da label 0 (azul)
    # # ---------------------------------
    # axes[0].scatter(
    #     df_label_0['title_chars'],
    #     df_label_0['title_entropy'],
    #     color='blue'
    # )

    # axes[0].set_xlabel('Número de caracteres')
    # axes[0].set_ylabel('Entropia do título')
    # axes[0].set_title('Label 0')
    # axes[0].grid(True)

    # # ---------------------------------
    # # Gráfico da label 1 (vermelho)
    # # ---------------------------------
    # axes[1].scatter(
    #     df_label_1['title_chars'],
    #     df_label_1['title_entropy'],
    #     color='red'
    # )

    # axes[1].set_xlabel('Número de caracteres')
    # axes[1].set_ylabel('Entropia do título')
    # axes[1].set_title('Label 1')
    # axes[1].grid(True)

    # # Ajustar espaçamento
    # plt.tight_layout()

    # # Mostrar ambos juntos
    # plt.savefig('entropia_com_numchars_por_label')
