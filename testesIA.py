import pandas as pd   
import numpy as np 
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

targets = pd.DataFrame(np.load('./dados/Classes.npy', allow_pickle=True))
df_sensor1 = pd.DataFrame(np.load('./dados_tratados/Dados_1.npy'))
df_sensor2 = pd.DataFrame(np.load('./dados_tratados/Dados_2.npy'))
df_sensor3 = pd.DataFrame(np.load('./dados_tratados/Dados_3.npy'))
df_sensor4 = pd.DataFrame(np.load('./dados_tratados/Dados_4.npy'))
df_sensor5 = pd.DataFrame(np.load('./dados_tratados/Dados_5.npy'))

'''
O sensor 4 não foi utilizado devido ele não alternar durante o tempo.
Usar os dados deste sensor não iriam melhorar a performance dos algoritmos de ML.
'''
inputs = pd.concat([df_sensor1, df_sensor2, df_sensor3, df_sensor5], axis=1)

classes = targets[0].unique()
num_classes = len(classes)

# Aplicação do factorize para mapear as classes para números
targets[0], mapeamento = pd.factorize(targets[0])

inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.3, random_state=123)

print('inputs_train.shape', inputs_train.shape)
print('targets_train.shape', targets_train.shape)
print('inputs_test.shape', inputs_test.shape)
print('targets_test.shape', targets_test.shape)

# ******************************************************************************************************
# Criando a rede

# Criando modelo
model = tf.keras.models.Sequential()

# Camada de entrada
model.add(tf.keras.Input(shape=(inputs.shape[1],), name='Entrada'))

# Camada oculta
model.add(tf.keras.layers.Dense(1, name='camadaOculta', kernel_initializer='zeros'))

# Camada de Saída
model.add(tf.keras.layers.Dense(units=num_classes, name='camadaSaida', activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['sparse_categorical_accuracy'])
model.summary()

# Treinando rede
training = model.fit(inputs_train, targets_train, epochs=20, batch_size=5, verbose=1)

# Plotando a acurácia do modelo
plt.plot(training.history['sparse_categorical_accuracy'])
plt.title('Acurácia do modelo')
plt.ylabel('Acurácia')
plt.xlabel('Época')
plt.show()