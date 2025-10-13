import numpy as np

# Para texto
X_train_text = np.load("D:/PIA-IA/dataset/datasetDividido/X_train_text.npy", allow_pickle=True)

# Para features numéricas
X_train_numeric = np.load("D:/PIA-IA/dataset/datasetDividido/X_train_numeric.npy")

# Para labels
y_train = np.load("D:/PIA-IA/dataset/datasetDividido/y_train.npy")

# Texto
'''for i, text in enumerate(X_train_text):
    print(f"{i}: {text}")'''

# Numéricos
for i, row in enumerate(X_train_numeric):
    print(f"{i}: {row}")

# Labels
'''for i, label in enumerate(y_train):
    print(f"{i}: {label}")'''