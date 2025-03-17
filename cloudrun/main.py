import pandas as pd
import yfinance as yf
from google.cloud import bigquery
import logging
from datetime import datetime

from flask import Flask
app = Flask(__name__)

# Configura logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route("/", methods=['GET'])
def atualizar_dados_ibovespa():
    """
    Baixa os dados do IBOVESPA do yfinance e insere diretamente no BigQuery.
    Ativado pelo Cloud Scheduler.
    """
    try:
        logging.info("Iniciando a função atualizar_dados_ibovespa...")

        # Configurações do BigQuery
        PROJECT_ID = "tech-challenge-ibovespa-new"  # Substitua pelo seu ID do projeto
        DATASET_ID = "ibovespa_dataset"
        TABLE_ID = "ibovespa_table"
        TABLE_URI = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

        # Inicializa o cliente BigQuery
        client = bigquery.Client()
        logging.info("Conectado ao BigQuery.")

        # Baixar dados do IBOVESPA usando yfinance
        ticker = "^BVSP"
        logging.info(f"Baixando dados do ticker: {ticker}")
        data = yf.download(ticker, period="10y", interval="1d")

        if data.empty:
            logging.error("Não foi possível baixar dados do IBOVESPA.")
            return "Erro: Não foi possível baixar dados do IBOVESPA.", 500

        data.reset_index(inplace=True)

        df = data[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

        logging.info("Dados baixados e formatados corretamente.")

        # Inserir os dados no BigQuery
        data_records = df.to_dict(orient='records')

        for row in data_records:
            row['Open'] = float(row['Open'])
            row['High'] = float(row['High'])
            row['Low'] = float(row['Low'])
            row['Close'] = float(row['Close'])
            row['Volume'] = int(row['Volume'])

        table = client.get_table(TABLE_URI)
        errors = client.insert_rows(table, data_records)
        if errors == []:
            logging.info("Dados inseridos com sucesso no BigQuery.")
            return "Dados inseridos com sucesso no BigQuery."
        else:
            logging.error(f"Ocorreram erros ao inserir dados no BigQuery: {errors}")
            return f"Ocorreram erros ao inserir dados no BigQuery: {errors}", 500

    except Exception as e:
        logging.error(f"Erro durante la ejecución: {e}")
        return f"Erro al actualizar los datos: {e}", 500
