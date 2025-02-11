# Importações
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from prophet import Prophet  # Importando a biblioteca Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# Configuração da página
st.set_page_config(page_title="Tech Challenge IBOVESPA", layout="wide")

# Título da Aplicação
st.title("Tech Challenge - Previsão do IBOVESPA")

# Informações do Projeto e Integrantes
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
    - Guillermo Jesus camahuali privat
    - Kelly Priscilla Matos Campos
    - José Victor Barros de Lima
    """
)

st.write("## Previsão do Fechamento Diário do IBOVESPA")
st.write("Bem-vindos ao nosso projeto de previsão do IBOVESPA! Aqui, exploramos dados históricos da bolsa de valores brasileira para construir um modelo preditivo capaz de antecipar o valor de fechamento diário do índice.")

# Função para obter dados do IBOVESPA
@st.cache_data
def obter_dados_ibovespa():
    ticker = "^BVSP"  # Símbolo do IBOVESPA no Yahoo Finance
    data = yf.download(ticker, period="10y", interval="1d")  # Aumentando o período para 10 anos
    data.reset_index(inplace=True)

    # *** BLOCO DE DEBUG - ADICIONE ISTO DENTRO DA FUNÇÃO obter_dados_ibovespa ***
    st.write("### Debug - data['Close'] do yfinance:")
    st.write("Tipo de data['Close']:")
    st.write(type(data['Close']))
    st.write("Primeiros 10 valores de data['Close']:")
    st.write(data['Close'].head(10))
    st.write("Existem valores não numéricos em data['Close']?:")
    st.write(pd.to_numeric(data['Close'], errors='coerce').isna().sum() > 0) # Verificar por NaNs após coerção
    # *** FIM DO NOVO BLOCO DE DEBUG ***

    return data

# Carregar os dados
with st.spinner('Carregando dados do IBOVESPA...'):
    try:
        data = obter_dados_ibovespa()
        st.success("Dados do IBOVESPA carregados com sucesso!")
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        st.stop()

# Análise Exploratória Inicial
st.header("1. Análise Exploratória dos Dados")
st.write("Primeiramente, realizamos uma análise exploratória dos dados históricos do IBOVESPA para entender o comportamento do índice ao longo do tempo.")

st.subheader("Visualização dos Dados Históricos")
st.write("Abaixo, podemos observar o gráfico da série temporal do preço de fechamento do IBOVESPA nos últimos 10 anos.")
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Close'], label='Preço de Fechamento', color='blue')
plt.title("Histórico do IBOVESPA (Últimos 10 Anos)")
plt.xlabel("Data")
plt.ylabel("Preço de Fechamento")
plt.legend()
plt.grid(True)
st.pyplot(plt)

st.subheader("Estatísticas Descritivas")
st.write("Um resumo estatístico dos dados nos ajuda a entender a distribuição e as principais características da série.")
st.dataframe(data[['Open', 'High', 'Low', 'Close', 'Volume']].describe())

# Preparação dos Dados
st.header("2. Preparação dos Dados")
st.write("Nesta etapa, preparamos os dados para o treinamento do modelo. Isso inclui a seleção das colunas relevantes e a divisão dos dados em conjuntos de treinamento e teste.")

df_fechamento = data[['Date', 'Close']].set_index('Date')

# Dividir os dados em treinamento e teste
tamanho_treino = int(len(df_fechamento) * 0.8)
treino_data, teste_data = df_fechamento[:tamanho_treino], df_fechamento[tamanho_treino:]

st.write(f"Tamanho do conjunto de treinamento: {len(treino_data)}")
st.write(f"Tamanho do conjunto de teste: {len(teste_data)}")

# Modelagem - Prophet
st.header("3. Modelagem Preditiva: Prophet")
st.write("Para realizar a previsão da série temporal, agora utilizaremos o modelo Prophet.")
st.write("Prophet é um modelo de previsão desenvolvido pelo Facebook, projetado para séries temporais com sazonalidade e tendências.")
st.write("Justificativa da técnica: Prophet é robusto, fácil de usar e frequentemente proporciona bons resultados com configurações padrão, o que o torna uma excelente alternativa ao ARIMA, especialmente dadas as dificuldades que encontramos.")

# Preparar dados para Prophet (Prophet requer uma coluna 'ds' para a data e 'y' para o valor)
df_prophet = df_fechamento.reset_index() # Resetar o índice para ter a data como coluna
df_prophet = df_prophet.rename(columns={'Date': 'ds', 'Close': 'y'})

# Converter a coluna 'ds' para datetime e 'y' para numérico explicitamente
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
df_prophet['y'] = pd.to_numeric(df_prophet['y'])


treino_prophet = df_prophet[:tamanho_treino].copy()
teste_prophet = df_prophet[tamanho_treino:].copy()

# Treinamento do Modelo Prophet
st.subheader("Treinamento do Modelo Prophet")
st.write("Treinamos o modelo Prophet com os dados de treinamento. Prophet detecta automaticamente a sazonalidade e as tendências nos dados.")

# *** BLOCO DE DEBUG DOS DADOS DE TREINO - MANTER PARA VERIFICAÇÃO ***
st.write("### Debug dos dados de treino para Prophet:")
st.dataframe(treino_prophet.head()) # Mostrar as primeiras linhas do DataFrame de treino
st.write("Tipos de dados das colunas:")
st.write(treino_prophet.dtypes) # Mostrar os tipos de dados das colunas
st.write("Tipo de df_prophet:") # ADICIONADO
st.write(type(df_prophet))      # ADICIONADO
st.write("Tipo de df_prophet['y']:") # ADICIONADO
st.write(type(df_prophet['y']))   # ADICIONADO
# *** FIM DO BLOCO DE DEBUG ***


try:
    modelo_prophet = Prophet()
    modelo_prophet.fit(treino_prophet)
    st.success("Modelo Prophet treinado com sucesso!")
except Exception as e:
    st.error(f"Erro ao treinar o modelo Prophet: {e}")
    st.stop()

# Previsões com Prophet
st.subheader("Previsões com Prophet")
st.write("Realizamos as previsões utilizando o modelo Prophet treinado.")

futuro_prophet = modelo_prophet.make_future_dataframe(periods=len(teste_data), freq='D') # Criar dataframe para previsões futuras
previsoes_prophet = modelo_prophet.predict(futuro_prophet)
previsoes_prophet_teste = previsoes_prophet.iloc[-len(teste_data):]['yhat'] # Obter previsões para o período de teste

# Avaliação do Modelo Prophet
st.header("4. Avaliação do Modelo Prophet")
st.write("Avaliamos o desempenho do modelo Prophet utilizando o conjunto de teste, da mesma forma que fizemos com ARIMA.")

mae_prophet = mean_absolute_error(teste_data['Close'], previsoes_prophet_teste)
rmse_prophet = sqrt(mean_squared_error(teste_data['Close'], previsoes_prophet_teste))

st.write(f"**Erro Médio Absoluto (MAE):** {mae_prophet:.2f}")
st.write(f"**Raiz do Erro Quadrático Médio (RMSE):** {rmse_prophet:.2f}")

# Calcular acurácia (usando um critério simples baseado em porcentagem de erro)
erro_percentual_medio_prophet = np.mean(np.abs((teste_data['Close'].values - previsoes_prophet_teste.values) / teste_data['Close'].values)) * 100
acuracidade_modelo_prophet = 100 - erro_percentual_medio_prophet

st.write(f"**Erro Percentual Médio (MAPE):** {erro_percentual_medio_prophet:.2f}%")
st.write(f"**Acuracidade do Modelo (aproximada):** {acuracidade_modelo_prophet:.2f}%")

if acuracidade_modelo_prophet >= 70:
    st.success("Acuracidade do modelo Prophet atingiu o mínimo de 70%!")
else:
    st.warning("A acuracidade do modelo Prophet está abaixo de 70%. Podemos tentar ajustar o modelo ou explorar outras opções se necessário.")

# Visualização das Previsões Prophet
st.subheader("Visualização das Previsões Prophet vs. Dados Reais")
st.write("O gráfico abaixo compara as previsões do modelo Prophet com os valores reais do IBOVESPA no conjunto de teste.")
plt.figure(figsize=(12, 6))
plt.plot(treino_data.index, treino_data['Close'], label='Dados de Treinamento', color='blue')
plt.plot(teste_data.index, teste_data['Close'], label='Dados Reais de Teste', color='green')
plt.plot(teste_data.index, previsoes_prophet_teste.index, previsoes_prophet_teste.values, label='Previsões Prophet', color='red', linestyle='--') # Correção aqui para graficar corretamente
plt.title("Previsões do Modelo Prophet vs. Dados Reais")
plt.xlabel("Data")
plt.ylabel("Preço de Fechamento")
plt.legend()
plt.grid(True)
plt.xlim(teste_data.index.min(), teste_data.index.max()) # Ajustar o eixo X para focar no teste
st.pyplot(plt)


# Conclusão (Modificada para Prophet)
st.header("5. Conclusão")
st.write(f"Neste projeto, desenvolvemos um modelo de previsão para o IBOVESPA utilizando a técnica **Prophet**. O modelo foi treinado com dados históricos de 10 anos e avaliado em um conjunto de teste, alcançando uma acuracidade de aproximadamente **{acuracidade_modelo_prophet:.2f}%**. ")

if acuracidade_modelo_prophet >= 70:
    st.write("O modelo Prophet demonstrou ser efetivo, atingindo a acuracidade desejada de 70% ou mais, e superando os problemas de instalação que tivemos com ARIMA.")
else:
    st.write("Embora o modelo Prophet não tenha atingido 70% de acuracidade nesta tentativa, continua sendo um modelo robusto e efetivo. Poderíamos tentar ajustar parâmetros do Prophet ou explorar outras variáveis externas se fosse necessário para melhorar a precisão.")


st.write("Para melhorias futuras, pode-se explorar:")
st.write("- Ajuste dos parâmetros do Prophet (embora Prophet seja menos sensível a isso do que ARIMA).")
st.write("- Incorporação de variáveis exógenas (fatores externos) que podem influenciar o IBOVESPA, se encontrarmos dados relevantes.")
st.write("- Avaliar o modelo Prophet com um horizonte de previsão mais longo.")

st.write("Este projeto fornece uma base sólida para a previsão do IBOVESPA utilizando Prophet, demonstrando uma alternativa viável e eficaz ao ARIMA, especialmente em situações onde as dependências de bibliotecas são problemáticas.")

st.write("---")
st.write("Desenvolvido como parte do Tech Challenge.")