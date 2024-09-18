import numpy as np
import os
from tratamentoDados import tratar_dados

# ETAPA 1: Tratamento dos dados
# Verificar se os dados já foram tratados
if os.path.exists("./dados_tratados/Dados_1.npy"):
    print("Os dados já foram tratados")
    classes = np.load("./dados/Classes.npy", allow_pickle=True)
    sensor1 = np.load("./dados_tratados/Dados_1.npy", allow_pickle=True)
    sensor2 = np.load("./dados_tratados/Dados_2.npy", allow_pickle=True)
    sensor3 = np.load("./dados_tratados/Dados_3.npy", allow_pickle=True)
    sensor4 = np.load("./dados_tratados/Dados_4.npy", allow_pickle=True)
    sensor5 = np.load("./dados_tratados/Dados_5.npy", allow_pickle=True)

else:
    print("Tratando os dados")
    # Carregar o arquivo .npy
    classes = np.load("./dados/Classes.npy", allow_pickle=True)
    sensor1 = np.load("./dados/Dados_1.npy", allow_pickle=True)
    sensor2 = np.load("./dados/Dados_2.npy", allow_pickle=True)
    sensor3 = np.load("./dados/Dados_3.npy", allow_pickle=True)
    sensor4 = np.load("./dados/Dados_4.npy", allow_pickle=True)
    sensor5 = np.load("./dados/Dados_5.npy", allow_pickle=True)

    print(sensor5[100])

    # Trata os dados
    sensor1 = tratar_dados(sensor1)
    sensor2 = tratar_dados(sensor2)
    sensor3 = tratar_dados(sensor3)
    sensor4 = tratar_dados(sensor4)
    sensor5 = tratar_dados(sensor5)

    # Salva os dados tratados para que não seja preciso tratar novamnte
    np.save("./dados_tratados/Dados_1.npy", sensor1)
    np.save("./dados_tratados/Dados_2.npy", sensor2)
    np.save("./dados_tratados/Dados_3.npy", sensor3)
    np.save("./dados_tratados/Dados_4.npy", sensor4)
    np.save("./dados_tratados/Dados_5.npy", sensor5)

print(sensor5[100])

'''
# Exibir o conteúdo do arquivo
print("Classes", classes.shape)
valores_unicos, contagens = np.unique(classes, return_counts=True)
print("Quantidade de classes:", contagens)
print("Classes:", valores_unicos)

coluna = 5000
print("Sensor 1:", sensor1.shape)
print(f"Sensor 1-{coluna}:{sensor1[coluna]}")
print("Sensor 2:", sensor2.shape)
print(f"Sensor 2-{coluna}:{sensor2[coluna]}")
print("Sensor 3:", sensor3.shape)
print(f"Sensor 3-{coluna}:{sensor3[coluna]}")
print("Sensor 4:", sensor4.shape)
print(f"Sensor 4-{coluna}:{sensor4[coluna]}")
print("Sensor 5:", sensor5.shape)
print(f"Sensor 5-{coluna}:{sensor5[coluna]}")

valores_unicos, contagens = np.unique(sensor4, return_counts=True)
print("Sensor 4:",valores_unicos, contagens)


for linha in range(sensor5.shape[0]):
    for coluna in range(sensor5.shape[1]):
        if(coluna != 200):
            if(np.isnan(sensor5[linha,coluna])):
                print(f"Sensor: {linha}, {coluna}")

print("Concluido")
'''