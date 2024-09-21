import pandas as pd   
import numpy as np 
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# ******************************************************************************************************
# Preparação dos dados

# Lê o arquivo dos estados de funcionamento dessa máquina
targets = pd.DataFrame(np.load('./dados/Classes.npy', allow_pickle=True))
classes = targets[0].unique()
num_classes = len(classes)
# Aplicação do factorize para mapear as classes para números
targets[0], mapeamento = pd.factorize(targets[0])

# Lê os dados dos sensores
df_sensor1 = pd.DataFrame(np.load('./dados/Dados_1.npy'))
df_sensor2 = pd.DataFrame(np.load('./dados/Dados_2.npy'))
df_sensor3 = pd.DataFrame(np.load('./dados/Dados_3.npy'))
df_sensor4 = pd.DataFrame(np.load('./dados/Dados_4.npy'))
df_sensor5 = pd.DataFrame(np.load('./dados/Dados_5.npy'))

# ******************************************************************************************************************
# Etapa 1: Pré-processamento

# Substitui os valores acima de 1 do sensor 1 para 0
df_sensor1 = df_sensor1.apply(np.vectorize(lambda x: 0 if x > 1 else x))

# Junta todos os dados de sensores em um único df
inputs = pd.concat([df_sensor1, df_sensor2, df_sensor3, df_sensor5], axis=1)

# Substitui as entradas com NaN por zeros
inputs = inputs.fillna(0)

# Data Normalization
scaler = StandardScaler()
scaler.fit(inputs)
inputs = scaler.transform(inputs)

# ******************************************************************************************************************
# Divisão dos dados em conjuntos de treino e teste.
inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.2)

# ******************************************************************************************************
# Criando a rede neural
model = tf.keras.models.Sequential()

# Camada de entrada
model.add(tf.keras.Input(shape=(inputs.shape[1],), name='Entrada'))

# Camada oculta1
model.add(tf.keras.layers.Dense(256,
                                name='camadaOculta',
                                kernel_initializer="glorot_uniform",
                                activation="relu"))
model.add(tf.keras.layers.Dropout(0.3, name='DOcamadaOculta'))

# Camada de Saída
model.add(tf.keras.layers.Dense(units=num_classes,
                                name='camadaSaida',
                                activation='softmax'))

# Compila a rede e apresenta a estrutura
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
model.summary()

# Treinar a rede com early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)

# Treinando rede
training = model.fit(inputs_train, targets_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)

# Avaliar o modelo
loss, accuracy = model.evaluate(inputs_test, targets_test)
print(f'Acurácia no conjunto de testes: {(accuracy*100):.2f}%')
print(f'Erro no conjunto de testes: {loss}')

# Prever os valores no conjunto de teste
predictions = model.predict(inputs_test)
predictions = np.argmax(predictions, axis=1)

# Criando subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Gráfico de acurácia
ax1.plot(training.history['sparse_categorical_accuracy'], label='Train Accuracy', marker='o')
ax1.plot(training.history['val_sparse_categorical_accuracy'], label='Validation Accuracy', marker='s')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend(loc='upper left')
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

# Gráfico de erro
ax2.plot(training.history['loss'], label='Train Loss', marker='o')
ax2.plot(training.history['val_loss'], label='Validation Loss', marker='s')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend(loc='upper right')
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

# Mostrando os gráficos
plt.tight_layout()
plt.savefig('./static/images/training.png')
plt.show()


# Calcular a matriz de confusão
cm = confusion_matrix(targets_test, predictions)

# Plotar a matriz de confusão
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('./static/images/confusion-matrix.png')
plt.show()