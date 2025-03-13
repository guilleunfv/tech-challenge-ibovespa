import streamlit as st
import pandas as pd
from google.cloud import storage
import io
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Configurações
BUCKET_NAME = "ibovespa-data-tech-challenge"  # Substitua pelo nome do seu bucket
FILE_PATH = "ibovespa/ibovespa_data.csv"  # Caminho para o arquivo no bucket

@st.cache_data
def obter_dados_ibovespa():
    """Carrega os dados do IBOVESPA do Google Cloud Storage."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(FILE_PATH)
        csv_data = blob.download_as_string()
        df = pd.read_csv(io.BytesIO(csv_data))

        # Converte la columna 'Último' a numérico, maneja los errores
        if not pd.api.types.is_numeric_dtype(df['Último']):
            df['Último'] = pd.to_numeric(df['Último'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')
        df = df.dropna(subset=['Último'])

        # Convierte la columna 'Data' a datetime y establece como índice
        df['Data'] = pd.to_datetime(df['Data'])
        df = df.set_index('Data')
        return df
    except Exception as e:
        st.error(f"Erro ao carregar os dados do Google Cloud Storage: {e}")
        return None

def criar_modelo_lstm(df):
    if df is None:
        st.error("Dados do IBOVESPA não carregados corretamente. Verifique a fonte de dados.")
        return

    # Extrai o valor de fechamento
    data = df['Último'].values.reshape(-1, 1)

    # Normalizando os dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # Divide los datos en conjuntos de entrenamiento y prueba
    train_size = int(len(data_scaled) * 0.7)
    train_data = data_scaled[:train_size]
    test_data = data_scaled[train_size:]

    # Define el look_back
    look_back = min(30, len(test_data) - 1)

    # Verifica si el look_back es válido
    if look_back <= 0:
        st.error("Erro: Dados de teste insuficientes para o look_back definido. Ajuste o look_back ou carregue mais dados.")
        return

    # Funciona para crear la set de datos
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            X.append(dataset[i:(i + look_back), 0])
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    X_train, Y_train = create_dataset(train_data, look_back)
    X_test, Y_test = create_dataset(test_data, look_back)

    # Redimensiona os dados para el formato 3D
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Cria modelo LSTM aprimorado
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Entrenar
    model.fit(X_train, Y_train, epochs=300, batch_size=16, verbose=0)

    # Prdecir
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse
    train_predict = scaler.inverse_transform(train_predict)
    Y_train = scaler.inverse_transform([Y_train])
    test_predict = scaler.inverse_transform(test_predict)
    Y_test = scaler.inverse_transform([Y_test])

    # Avaliar
    train_rmse = np.sqrt(mean_squared_error(Y_train[0], train_predict[:, 0]))
    test_rmse = np.sqrt(mean_squared_error(Y_test[0], test_predict[:, 0]))
    r2 = r2_score(Y_test[0], test_predict[:, 0])
    st.write(f"**RMSE no conjunto de treino:** {train_rmse:.2f}")
    st.write(f"**RMSE no conjunto de teste:** {test_rmse:.2f}")
    st.write(f"**R² Score: {r2:.2f}**")

    # Preparando para análise por data
    test_dates = df.index[train_size + look_back + 1: len(df)]
    df_results = pd.DataFrame({'Real': Y_test[0], 'Previsão': test_predict[:, 0]}, index=test_dates)

    # Cálculo do erro absoluto percentual médio (MAPE)
    mape = np.mean(np.abs((df_results['Real'] - df_results['Previsão']) / df_results['Real'])) * 100

    # Cálculo da acuracidade
    accuracy = 100 - mape
    st.write(f"**Erro Absoluto Percentual Médio (MAPE): {mape:.2f}%**")
    st.write(f"**Acuracidade do Modelo: {accuracy:.2f}%**")

    # Plot dos resultados
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_results['Real'], label='Real')
    ax.plot(df_results['Previsão'], label='Previsão')
    plt.title("Resultados da Previsão LSTM")
    plt.xlabel("Data")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)

    return accuracy, fig

# Llama a la función para obter los dados del bucket
data = obter_dados_ibovespa()

# Llamar a la función
if data is not None:
    criar_modelo_lstm(data)
else:
    st.write("Ocorreu um erro ao carregar os dados.")

st.sidebar.header("Sobre o Projeto")
st.sidebar.info(
    "Este projeto tem como objetivo realizar a previsão do índice IBOVESPA utilizando dados históricos e um modelo de séries temporais. "
    "Desenvolvido como parte do Tech Challenge para demonstrar a aplicação de conhecimentos em análise de dados e modelagem preditiva."
)

st.sidebar.header("Integrantes")
st.sidebar.info(
    """
    - Rosicleia Cavalcante Mota
    - Nathalia Dias Araujo
    - Guillermo Jesus Camahuali Privat
    - Kelly Priscilla Matos Campos
    - José Victor Barros de Lima
    """
)

st.write("## Previsão do Fechamento Diário do IBOVESPA")
st.write("Bem-vindos ao nosso projeto de previsão do IBOVESPA! Aqui, exploramos dados históricos da bolsa de valores brasileira para construir um modelo preditivo capaz de antecipar o valor de fechamento diário do índice.")
