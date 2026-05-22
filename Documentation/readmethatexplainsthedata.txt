DESCRIPTION
-----------
Each CSV file contains a structured table of extracted features for all benign and malicious samples of that file type.

- The “All_Features” files include every feature extracted from the respective format.
- The “Top” files contain the most discriminative subset of features selected using SHAP and Random Forest feature importance analysis.

Each row in a CSV file represents one document sample.
Each column represents a feature or label.

Example columns:
filename, feature_1, feature_2, ..., feature_n, label

The “label” column indicates the class:
0 = Benign
1 = Malicious


FEATURE SUMMARY
----------------s
Word files:
- Extracted from DOC and DOCX formats.
- Features include macro presence, OLE object count, hypeBibTeX:
@dataset{nejati2025_malicious_attachment_dataset,
  author       = {Fatemeh Nejati and collaborators},
  title        = {A Comprehensive Multi-Format Malicious Attachment Dataset},
  year         = {2025},
  institution  = {Canadian Institute for Cybersecurity, University of New Brunswick},
  publisher    = {....},
  doi          = {to_be_assigned}
}rlink count, embedded object size, font family diversity, paragraph statistics, and metadata complexity.
- Word_All_Features.csv contains all 94 extracted features.
- Word_Top10_Features.csv contains the 10 most important features selected from SHAP and Random Forest analysis.

Excel files:
- Extracted from XLS and XLSX formats.
- Features include number of sheets, average cell count, formula ratio, macro count, embedded object count, numeric cell ratio, and metadata entropy.
- Excel_All_Features.csv includes all extracted features.
- Excel_Top10_Features.csv contains the top 10 selected features.

PDF files:
- Extracted from raw PDF documents.
- Features include total object count, stream count, compression filters, embedded JavaScript indicator, metadata length, and file entropy.
- PDF_All_Features.csv includes all extracted features.
- PDF_Top10_Features.csv contains the 10 most discriminative features.

HTML files:
- Extracted from benign and malicious webpages.
- Features include script count, external link ratio, keyword frequency (login, password, verify), tag ratio, URL length, and inline JavaScript density.
- HTML_All_Features.csv includes all extracted features.
- HTML_Top13_Features.csv contains the top 13 most relevant features identified during feature selection.


USAGE
-----
These CSV files can be directly loaded into any data analysis or machine learning framework (Python, R, MATLAB, Weka, etc.).

Example usage in Python:

import pandas as pd
df = pd.read_csv("Word_Top10_Features.csv")
print(df.head())

Each file can be used for:
- Model training and evaluation
- Cross-format feature comparison
- Feature importance visualization
- Dataset benchmarking


RECOMMENDED ML PIPELINE
-----------------------
1. Load CSV file into Python or your preferred environment.
2. Split the dataset into training and test sets (e.g., 70/30 or 80/20).
3. Scale or normalize numeric features if needed.
4. Train classifiers such as Random Forest, XGBoost, or Decision Tree.
5. Evaluate accuracy, precision, recall, and F1-score.
6. Generate confusion matrices or spider charts for performance visualization.


FILE STRUCTURE
---------------
Each CSV follows this structure:

filename, feature_1, feature_2, ... feature_n, label

Example:
sample1.docx, 5, 0, 0.32, 1, 1, 0
sample2.docx, 3, 1, 0.10, 0, 0, 1


NOTES
-----
- The CSV files are cleaned, formatted, and ready for machine learning analysis.
- Missing or invalid data have been handled during feature extraction.
- The same labeling convention (0 = benign, 1 = malicious) applies across all files.


LICENSE
-------
This dataset is distributed under the Creative Commons Attribution–NonCommercial 4.0 (CC BY-NC 4.0) license.
Use is permitted for non-commercial academic and research purposes with appropriate citation.


CITATION
--------
If you use this dataset or its associated scripts, please cite:

Nejati, N. et al.
"A Comprehensive Multi-Format Malicious Attachment Dataset for Email Threat Detection."
Canadian Institute for Cybersecurity (CIC), University of New Brunswick, 2025.

BibTeX:
@dataset{nejati2025_malicious_attachment_dataset,
  author       = {Fatemeh Nejati and collaborators},Q
  title        = {A Comprehensive Multi-Format Malicious Attachment Dataset},
  year         = {2025},
  institution  = {Canadian Institute for Cybersecurity, University of New Brunswick},
  publisher    = {....},
  doi          = {to_be_assigned}
}




SUMMARY
--------
This folder provides the complete set of structured features extracted from Word, Excel, PDF, and HTML documents. 
The provided CSV files are standardized, labeled, and designed for reproducible malicious attachment analysis using both classical and modern machine learning methods.
