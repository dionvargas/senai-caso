from flask import Flask, render_template, request
import pandas as pd
import os
import numpy as np
import plotly.express as px
import plotly.io as pio
from tratar_dados import tratar_dados, substituir_NaN

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analise', methods=['GET', 'POST'])
def analise():
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
    print(valores_unicos, contagens)
    print (data_classes)

    def resume_sensor(df_sensor):
        resumo = {
            "canais": df_sensor.columns.tolist(),
            "formato": df_sensor.shape,
            "minimo": df_sensor.min(axis=1).min(),
            "maximo": df_sensor.max(axis=1).max(),
            "inf": np.isinf(df_sensor.values).any(),
            "NaN": df_sensor.isnull().values.any()
        }
        return resumo

    # Resumo dos dados
    data_sensores = {
        "sensor1": resume_sensor(df_data1),
        "sensor2": resume_sensor(df_data2),
        "sensor3": resume_sensor(df_data3),
        "sensor4": resume_sensor(df_data4),
        "sensor5": resume_sensor(df_data5)
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

    fig = px.line(df_graph, x=range(0,df_graph.shape[0]), y=df_graph.columns)
    fig.update_layout(
        xaxis_title="t(s)",
        yaxis_title="Valor",
        legend_title="Canais"
    )
    graph_html = pio.to_html(fig, full_html=False, config={'displaylogo': False})

    return render_template('analise.html', graph_html=graph_html, data = data_sensores, data_classes = data_classes)

@app.route('/tratamento', methods=['GET', 'POST'])
def tratamento():
    print("Entrou aqui")
    if os.path.exists("./dados_tratados/Dados_5.npy"):
        dados_tratados = True
    else:
        dados_tratados = False

    if request.method == 'POST':
        print("É POST")
        if dados_tratados:
            for file_name in os.listdir("./dados_tratados"):
                file_path = os.path.join("./dados_tratados", file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            dados_tratados = False
        else:
            # Carregar o arquivo .npy
            sensor1 = np.load("./dados/Dados_1.npy", allow_pickle=True)
            sensor2 = np.load("./dados/Dados_2.npy", allow_pickle=True)
            sensor3 = np.load("./dados/Dados_3.npy", allow_pickle=True)
            sensor4 = np.load("./dados/Dados_4.npy", allow_pickle=True)
            sensor5 = np.load("./dados/Dados_5.npy", allow_pickle=True)

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

            dados_tratados = True

    return render_template('tratamento_dos_dados.html', dados_tratados=dados_tratados)

if __name__ == '__main__':
    app.run(debug=True)