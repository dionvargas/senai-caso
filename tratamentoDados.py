import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Trata os dados
def tratar_dados(matriz):
    matriz = substituir_NaN(matriz)
    matriz = normalizar_min_max(matriz)
    return matriz

# Substituir os dados NaN por 0
def substituir_NaN(matriz, valor = 0):
    # Substituir os dados NaN por 0
    matriz[np.isnan(matriz)] = 0
    return matriz

# Normaliza os dados utilizando MinMax
def normalizar_min_max(matriz):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    matriz = scaler.fit_transform(matriz)
    return matriz

# Normaliza s dados utilizando normalização Z-score
def normalizar_z_score(matriz):
    # Calcula a média e o desvio padrão dos dados
    media = np.mean(matriz)
    desvio_padrao = np.std(matriz)

    # Aplica a normalização Z-score
    matriz = (matriz - media) / desvio_padrao
    return matriz