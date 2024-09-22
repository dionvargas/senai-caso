from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    # Lê os dados
    classes = np.load('./dados/Classes.npy', allow_pickle=True)
    df_data1 = pd.DataFrame(np.load('./dados/Dados_1.npy'))
    df_data2 = pd.DataFrame(np.load('./dados/Dados_2.npy'))
    df_data3 = pd.DataFrame(np.load('./dados/Dados_3.npy'))
    df_data4 = pd.DataFrame(np.load('./dados/Dados_4.npy'))
    df_data5 = pd.DataFrame(np.load('./dados/Dados_5.npy'))

    data_classes = {}
    valores_unicos, contagens = np.unique(classes, return_counts=True)
    for i in range(len(valores_unicos)):
        data_classes[valores_unicos[i]] = contagens[i]

    def resume_sensor(df_sensor, nome):
        resumo = {
            "nome": nome,
            "canais": df_sensor.columns.tolist(),
            "formato": df_sensor.shape,
            "minimo": df_sensor.min(axis=1).min(),
            "maximo": df_sensor.max(axis=1).max(),
            "inf": np.isinf(df_sensor.values).any(),
            "NaN": df_sensor.isnull().values.any()
        }

        # Gerando as imagens de histograma dos dataframes
        # Ajustando o tamanho da figura
        plt.figure(figsize=(16, 4))
        # Gerando o histograma
        df_sensor[i].hist(bins=500, grid=False)
        # Montando o histograma
        plt.xlabel('Valores')
        plt.ylabel('Frequência')
        plt.title(f'Histograma do {nome}')
        # Salvando o histograma em um arquivo
        plt.savefig(f'./static/images/histogram/{nome}.png')

        return resumo

    # Resumo dos dados
    data_sensores = {
        "sensor1": resume_sensor(df_data1, "Sensor 1"),
        "sensor2": resume_sensor(df_data2, "Sensor 2"),
        "sensor3": resume_sensor(df_data3, "Sensor 3"),
        "sensor4": resume_sensor(df_data4, "Sensor 4"),
        "sensor5": resume_sensor(df_data5, "Sensor 5")
    }

    # Substitui os NaN para zero para poderem ser plotados
    df_data1 = df_data1.fillna(0)
    df_data2 = df_data2.fillna(0)
    df_data3 = df_data3.fillna(0)
    df_data4 = df_data4.fillna(0)
    df_data5 = df_data5.fillna(0)

    if request.method == 'POST':
        selected_sensor1 = request.form.get('canal_sensor1')
        selected_sensor2 = request.form.get('canal_sensor2')
        selected_sensor3 = request.form.get('canal_sensor3')
        selected_sensor4 = request.form.get('canal_sensor4')
        selected_sensor5 = request.form.get('canal_sensor5')
    else: 
        selected_sensor1 = '0'
        selected_sensor2 = '0'
        selected_sensor3 = '0'
        selected_sensor4 = '0'
        selected_sensor5 = '0'

    df_graph = pd.DataFrame()
    df_graph['S1-C' + selected_sensor1] = df_data1[int(selected_sensor1)]
    df_graph['S2-C' + selected_sensor2] = df_data2[int(selected_sensor2)]
    df_graph['S3-C' + selected_sensor3] = df_data3[int(selected_sensor3)]
    df_graph['S4-C' + selected_sensor4] = df_data4[int(selected_sensor4)]
    df_graph['S5-C' + selected_sensor5] = df_data5[int(selected_sensor5)]

    fig = px.line(df_graph, x=df_graph.index/df_graph.shape[0]*5, y=df_graph.columns)
    fig.update_layout(
        xaxis_title="t(s)",
        yaxis_title="Valor",
        legend_title="Canais"
    )
    graph_html = pio.to_html(fig, full_html=False, config={'displaylogo': False})

    return render_template('index.html', graph_html=graph_html, data = data_sensores, data_classes = data_classes)

if __name__ == '__main__':
    app.run(debug=True)