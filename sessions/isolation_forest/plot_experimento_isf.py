import matplotlib.pyplot as plt

contaminacao2 = [0.05, 0.10, 0.15, 0.20, 0.25,
                0.30, 0.35, 0.40, 0.45, 0.50]

acc2 = [0.5233, 0.5544, 0.5632, 0.6026, 0.6342,
       0.6503, 0.6451, 0.6622, 0.6933, 0.7207]

tpr2 = [0.9724, 0.9539, 0.9171, 0.9038, 0.8915,
       0.8628, 0.8014, 0.7625, 0.7441, 0.7155]

precision2 = [0.5155, 0.5335, 0.5404, 0.5675, 0.5921,
             0.6091, 0.6146, 0.6395, 0.6801, 0.7281]

f12 = [0.6738, 0.6843, 0.6801, 0.6972, 0.7116,
      0.7141, 0.6957, 0.6956, 0.7107, 0.7217]

contaminacao1 = [
    0.05,
    0.10, 0.15, 0.20, 0.25, 0.30,
    0.35, 0.40, 0.45, 0.50
]

acc1 = [
    0.4834,
    0.4813, 0.5005, 0.1477, 0.1746, 0.1969,
    0.2202, 0.2440, 0.2699, 0.2959
]

tpr1 = [
    0.9079,
    0.8557, 0.8536, 0.1013, 0.0972, 0.0972,
    0.0972, 0.0972, 0.0972, 0.0962
]

precision1 = [
    0.4944,
    0.4929, 0.5039, 0.1143, 0.1179, 0.1245,
    0.1323, 0.1414, 0.1527, 0.1649
]

f11 = [
    0.6402,
    0.6255, 0.6337, 0.1074, 0.1066, 0.1092,
    0.1121, 0.1152, 0.1188, 0.1215
]


plt.figure(figsize=(10, 6))

plt.plot(contaminacao1, acc1, marker='o', label='Accuracy')
plt.plot(contaminacao1, tpr1, marker='s', label='TPR (Recall)')
plt.plot(contaminacao1, precision1, marker='^', label='Precision')
plt.plot(contaminacao1, f11, marker='d', label='F1-score')

plt.xlabel('Contaminação')
plt.ylabel('Valor da métrica')
plt.title('Isolation Forest vs Taxa de Contaminação')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()