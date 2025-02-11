# Importações
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import pmdarima as pm  # Importando a biblioteca pmdarima para AutoARIMA

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
    - Rosicleia Cavalcante Mota\n
    - Nathalia Dias Araujo\n
    - Guillermo Jesus camahuali privat\n
    - Kelly Priscilla Matos Campos\n
    - José Victor Barros de Lima\n
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

# Modelagem - ARIMA
st.header("3. Modelagem Preditiva: ARIMA")
st.write("Para realizar a previsão da série temporal, escolhemos o modelo ARIMA (Autoregressive Integrated Moving Average).")
st.write("O ARIMA é uma classe de modelos estatísticos para analisar e prever séries temporais. Ele utiliza a autocorrelação nos dados para fazer previsões futuras.")
st.write("Justificativa da técnica: O modelo ARIMA é adequado para séries temporais que apresentam autocorrelação e estacionariedade (ou podem ser transformadas em estacionárias). O IBOVESPA, como muitas séries financeiras, exibe essas características, tornando o ARIMA uma escolha razoável para modelagem.")

# Treinamento do Modelo ARIMA com AutoARIMA
st.subheader("Treinamento do Modelo ARIMA com AutoARIMA")
st.write("Utilizaremos a função `auto_arima` para encontrar automaticamente os melhores parâmetros (p, d, q) para o modelo ARIMA. Isso otimiza o modelo para os nossos dados.")

try:
    modelo_auto_arima = pm.auto_arima(treino_data['Close'], seasonal=False, stepwise=True,
                                 suppress_warnings=True, error_action="ignore", trace=True) # Rodar o auto_arima
    st.write("Melhores parâmetros ARIMA encontrados pelo AutoARIMA:")
    st.write(f"ARIMA Ordem: **{modelo_auto_arima.order}**") # Exibe a ordem do modelo

    modelo_arima = ARIMA(treino_data['Close'], order=modelo_auto_arima.order) # Usa a ordem otimizada
    modelo_arima_fit = modelo_arima.fit()
    st.success("Modelo ARIMA treinado com sucesso usando AutoARIMA!")

except Exception as e:
    st.error(f"Erro ao treinar o modelo ARIMA com AutoARIMA: {e}")
    st.stop()

# Avaliação do Modelo
st.header("4. Avaliação do Modelo")
st.write("Após treinar o modelo, avaliamos seu desempenho utilizando o conjunto de teste. As métricas utilizadas são o Erro Médio Absoluto (MAE) e a Raiz do Erro Quadrático Médio (RMSE), que medem a magnitude dos erros de previsão.")

previsoes_arima = modelo_arima_fit.predict(start=len(treino_data), end=len(df_fechamento)-1)
mae_arima = mean_absolute_error(teste_data['Close'], previsoes_arima)
rmse_arima = sqrt(mean_squared_error(teste_data['Close'], previsoes_arima))

st.write(f"**Erro Médio Absoluto (MAE):** {mae_arima:.2f}")
st.write(f"**Raiz do Erro Quadrático Médio (RMSE):** {rmse_arima:.2f}")

# Calcular acurácia (usando um critério simples baseado em porcentagem de erro)
erro_percentual_medio = np.mean(np.abs((teste_data['Close'] - previsoes_arima) / teste_data['Close'])) * 100
acuracidade_modelo = 100 - erro_percentual_medio

st.write(f"**Erro Percentual Médio (MAPE):** {erro_percentual_medio:.2f}%")
st.write(f"**Acuracidade do Modelo (aproximada):** {acuracidade_modelo:.2f}%")

if acuracidade_modelo >= 70:
    st.success("Acuracidade do modelo atingiu o mínimo de 70%!")
else:
    st.warning("A acuracidade do modelo ainda está abaixo de 70%. Considere explorar modelos mais avançados ou ajustar os dados.")

# Visualização das Previsões
st.subheader("Visualização das Previsões ARIMA vs. Dados Reais")
st.write("O gráfico abaixo compara as previsões do modelo ARIMA com os valores reais do IBOVESPA no conjunto de teste.")
plt.figure(figsize=(12, 6))
plt.plot(treino_data.index, treino_data['Close'], label='Dados de Treinamento', color='blue')
plt.plot(teste_data.index, teste_data['Close'], label='Dados Reais de Teste', color='green')
plt.plot(teste_data.index, previsoes_arima, label='Previsões ARIMA', color='red', linestyle='--')
plt.title("Previsões do Modelo ARIMA vs. Dados Reais")
plt.xlabel("Data")
plt.ylabel("Preço de Fechamento")
plt.legend()
plt.grid(True)
plt.xlim(teste_data.index.min(), teste_data.index.max()) # Ajustar o eixo X para focar no teste
st.pyplot(plt)


# Conclusão
st.header("5. Conclusão")
st.write("Neste projeto, desenvolvemos um modelo de previsão para o IBOVESPA utilizando a técnica ARIMA. O modelo foi treinado com dados históricos de 10 anos e avaliado em um conjunto de teste, alcançando uma acuracidade de aproximadamente {:.2f}%. ".format(acuracidade_modelo))

if acuracidade_modelo >= 70:
    st.write("Com a otimização dos parâmetros usando AutoARIMA, o modelo atingiu a acuracidade desejada de 70% ou mais, demonstrando uma melhoria significativa em relação à versão anterior.")
else:
    st.write("Apesar da otimização com AutoARIMA, a acuracidade do modelo ainda não atingiu 70%. Isso pode indicar a necessidade de explorar modelos mais avançados ou considerar a inclusão de variáveis externas (fatores exógenos) que possam influenciar o comportamento do IBOVESPA.")

st.write("Para melhorias futuras, pode-se explorar:")
st.write("- Análise dos resíduos do modelo ARIMA para verificar se há padrões não capturados.")
st.write("- Incorporação de variáveis exógenas (fatores externos) que podem influenciar o IBOVESPA, como taxas de juros, inflação, eventos políticos, etc. (Modelo SARIMAX).")
st.write("- Testar outros modelos de séries temporais mais avançados, como modelos Prophet ou redes neurais recorrentes (LSTMs), para capturar padrões mais complexos nos dados.")
st.write("- Ajustar o período de dados históricos para treinamento e teste, e realizar validação cruzada para uma avaliação mais robusta do modelo.")


st.write("Este projeto fornece uma base sólida para a previsão do IBOVESPA e demonstra a aplicação de técnicas de análise de séries temporais no contexto do mercado financeiro.")

st.write("---")
st.write("Desenvolvido como parte do Tech Challenge.")