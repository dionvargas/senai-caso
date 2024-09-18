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
    df_data1 = pd.DataFrame(substituir_NaN(np.load('./dados/Dados_1.npy')))
    df_data2 = pd.DataFrame(substituir_NaN(np.load('./dados/Dados_2.npy')))
    df_data3 = pd.DataFrame(substituir_NaN(np.load('./dados/Dados_3.npy')))
    df_data4 = pd.DataFrame(substituir_NaN(np.load('./dados/Dados_4.npy')))
    df_data5 = pd.DataFrame(substituir_NaN(np.load('./dados/Dados_5.npy')))
    
    if request.method == 'POST':
        selected_sensor1 = request.form.get('sensor1')
        selected_sensor2 = request.form.get('sensor2')
        selected_sensor3 = request.form.get('sensor3')
        selected_sensor4 = request.form.get('sensor4')
        selected_sensor5 = request.form.get('sensor5')
    else: 
        selected_sensor1 = '0'
        selected_sensor2 = '0'
        selected_sensor3 = '0'
        selected_sensor4 = '0'
        selected_sensor5 = '0'

    df = pd.DataFrame()

    if(selected_sensor1 != "Não mostrar"):
        df['S1-C' + selected_sensor1] = df_data1[int(selected_sensor1)]
    if(selected_sensor2 != "Não mostrar"):
        df['S2-C' + selected_sensor2] = df_data2[int(selected_sensor2)]
    if(selected_sensor2 != "Não mostrar"):
        df['S3-C' + selected_sensor3] = df_data3[int(selected_sensor3)]
    if(selected_sensor2 != "Não mostrar"):
        df['S4-C' + selected_sensor4] = df_data4[int(selected_sensor4)]
    if(selected_sensor2 != "Não mostrar"):
        df['S5-C' + selected_sensor5] = df_data5[int(selected_sensor5)]

    fig = px.line(df, x=range(0,df.shape[0]), y=df.columns)
    graph_html = pio.to_html(fig, full_html=False)

    return render_template('analise.html', graph_html=graph_html,
                           data_sensor_1=df_data1.columns,
                           data_sensor_2=df_data2.columns,
                           data_sensor_3=df_data3.columns,
                           data_sensor_4=df_data4.columns,
                           data_sensor_5=df_data5.columns)

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