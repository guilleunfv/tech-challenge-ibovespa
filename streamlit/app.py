import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf  # Importa yfinance
import time

# Configurações
CSV_URL = "https://storage.googleapis.com/ibovespa-data-tech-challenge/ibovespa/ibovespa_data.csv"
LOOK_BACK = 60
TEST_SIZE = 0.3  # Tamanho do conjunto de teste em relação ao total

@st.cache_data
def obter_dados_ibovespa():
    """Carrega os dados do IBOVESPA diretamente do URL público ou usa yfinance."""
    st.write("Obtendo dados do IBOVESPA...")
    start_time = time.time()
    try:
        df = pd.read_csv(CSV_URL, skiprows=[1])  # Carrega do CSV
    except Exception as e:
        st.error(f"Erro ao carregar os dados do CSV: {e}. Tentando yfinance...")

        ticker = "^BVSP"  # Ticker do IBOVESPA no yfinance
        df = yf.download(ticker, start="2015-01-01")  # Define um período inicial

        if df.empty:
            st.error("Erro ao carregar dados do yfinance. Verifique o ticker ou a conexão.")
            return None

    df.rename(columns={'Close': 'Último', 'High': 'Máxima', 'Low': 'Mínima', 'Open': 'Abertura', 'Volume': 'Volume', 'Date': 'Data'}, inplace=True)

    df['Data'] = pd.to_datetime(df.index)  # Converte o índice para datetime
    df = df.set_index('Data')  # Define a coluna 'Data' como índice
    end_time = time.time()
    st.write(f"Tempo para carregar e processar os dados: {end_time - start_time:.2f} segundos")
    return df

def criar_modelo_lstm(df):
    """Cria e treina o modelo LSTM para prever o IBOVESPA."""
    st.write("Criando modelo LSTM...")
    if df is None:
        st.error("Dados do IBOVESPA não carregados corretamente. Verifique a fonte de dados.")
        return

    # Extrai o valor de fechamento
    data = df['Último'].values.reshape(-1, 1)

    # Normalizando os dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # Divide os dados em conjuntos de treinamento e teste
    train_size = int(len(data_scaled) * (1 - TEST_SIZE))
    train_data = data_scaled[:train_size]
    test_data = data_scaled[train_size:]

    # Função para criar o dataset
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            X.append(dataset[i:(i + look_back), 0])
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    X_train, Y_train = create_dataset(train_data, LOOK_BACK)
    X_test, Y_test = create_dataset(test_data, LOOK_BACK)

    st.write(f"Forma de X_train: {X_train.shape}")
    st.write(f"Forma de Y_train: {Y_train.shape}")
    st.write(f"Forma de X_test: {X_test.shape}")
    st.write(f"Forma de Y_test: {Y_test.shape}")

    if len(X_test) == 0 or len(Y_test) == 0:
        st.error("Erro: Não há dados suficientes no conjunto de teste. Aumente o tamanho dos dados ou ajuste o look_back.")
        return

    # Redimensiona os dados para o formato 3D
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

    # Treina o modelo
    st.write("Iniciando treinamento do modelo...")
    model.fit(X_train, Y_train, epochs=300, batch_size=16, verbose=0)
    st.write("Treinamento do modelo concluído.")

    # Faz as previsões
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverte a normalização
    train_predict = scaler.inverse_transform(train_predict)
    Y_train = scaler.inverse_transform([Y_train])
    test_predict = scaler.inverse_transform(test_predict)
    Y_test = scaler.inverse_transform([Y_test])

    # Avalia o modelo
    train_rmse = np.sqrt(mean_squared_error(Y_train[0], train_predict[:, 0]))
    test_rmse = np.sqrt(mean_squared_error(Y_test[0], test_predict[:, 0]))
    r2 = r2_score(Y_test[0], test_predict[:, 0])
    st.write(f"**RMSE no conjunto de treino:** {train_rmse:.2f}")
    st.write(f"**RMSE no conjunto de teste:** {test_rmse:.2f}")
    st.write(f"**R² Score: {r2:.2f}**")

    # Preparando para análise por data
    test_dates = df.index[train_size + LOOK_BACK + 1: len(df)]
    df_results = pd.DataFrame({'Real': Y_test[0], 'Previsão': test_predict[:, 0]}, index=test_dates)

    # Cálculo do erro absoluto percentual médio (MAPE)
    mape = np.mean(np.abs((df_results['Real'] - df_results['Previsão']) / df_results['Real'])) * 100

    # Cálculo da acuracidade
    accuracy = 100 - mape
    st.write(f"**Erro Absoluto Percentual Médio (MAPE): {mape:.2f}%**")
    st.write(f"**Acuracidade do Modelo: {accuracy:.2f}%**")

    # Exibe estatísticas descritivas
    st.subheader("Estatísticas Descritivas")
    st.dataframe(df_results.describe())

    # Exibe os dados reais e previstos
    st.subheader("Dados Reais vs. Previstos")
    st.dataframe(df_results)

    # Plot dos resultados
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_results['Real'], label='Real', color='blue')
    ax.plot(df_results['Previsão'], label='Previsão', color='orange')
    plt.title("Resultados da Previsão LSTM")
    plt.xlabel("Data")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)

    # Plot dos erros
    df_results['Erro'] = df_results['Real'] - df_results['Previsão']
    fig_erro, ax_erro = plt.subplots(figsize=(12, 6))
    ax_erro.plot(df_results['Erro'], label='Erro', color='red')
    plt.title("Erros de Previsão")
    plt.xlabel("Data")
    plt.ylabel("Erro")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig_erro)

    # Histograma dos erros
    fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
    ax_hist.hist(df_results['Erro'], bins=30, color='purple', alpha=0.7)
    plt.title("Distribuição dos Erros")
    plt.xlabel("Erro")
    plt.ylabel("Frequência")
    st.pyplot(fig_hist)

    return accuracy, fig, fig_erro, fig_hist

# Título principal
st.title("Previsão do IBOVESPA com LSTM")

# Descrição do projeto
st.write("Este projeto utiliza um modelo LSTM para prever o fechamento diário do IBOVESPA.")

# Chama a função para obter os dados do IBOVESPA
data = obter_dados_ibovespa()

# Executa o modelo LSTM e exibe os resultados
if data is not None:
    accuracy, fig, fig_erro, fig_hist = criar_modelo_lstm(data)
else:
    st.write("Ocorreu um erro ao carregar os dados.")

# Barra lateral
st.sidebar.header("Sobre o Projeto")
st.sidebar.info(
    "Este projeto tem como objetivo realizar a previsão do índice IBOVESPA utilizando dados históricos e um modelo de séries temporais. "
    "Desenvolvido como parte do Tech Challenge para demonstrar a aplicação de conhecimentos em análise de dados e modelagem preditiva."
)

st.sidebar.header("Integrantes")
st.sidebar.info(
    """
    - Rosicleia Cavalcante Mota
    - Guillermo Jesus Camahuali Privat
    - Kelly Priscilla Matos Campos
    """
)