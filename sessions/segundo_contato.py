import pandas as pd
import numpy as np
from utils import documentar, configurar
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
            print(str(df.iloc[i,0]).removeprefix("/content/drive/MyDrive/UNB/Mastercard project/Dataset/PDF_Both_Done/Benign/All_Files_10000/").removesuffix(".pdf"))
        if ("/content/drive/MyDrive/UNB/Mastercard project/Dataset/PDF_Both_Done/Malicious/CIC_malicious_pdfs_All/" in str(df.iloc[i,0])):
            print(str(df.iloc[i,0]).removeprefix("/content/drive/MyDrive/UNB/Mastercard project/Dataset/PDF_Both_Done/Malicious/CIC_malicious_pdfs_All/").removesuffix(".pdf"))
    # for col in colunas:
    #     if(df[col].unique().__len__() == 1):
    #         df.drop(col,axis=1)
    # df.to_csv("data/PDF_All_feature_Clean.csv")
        
