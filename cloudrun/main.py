import pandas as pd
import io
import os
from google.cloud import storage
from google.cloud import bigquery
import yfinance as yf
from flask import Flask
import logging

app = Flask(__name__)

# Configura logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route("/", methods=['GET'])
def atualizar_dados_ibovespa():
    try:
        logging.info("Iniciando a função atualizar_dados_ibovespa...")

        # Validar la variable de entorno BUCKET_NAME
        bucket_name = os.environ.get("BUCKET_NAME")
        if not bucket_name:
            logging.error("O nome do bucket não está configurado na variável de ambiente BUCKET_NAME.")
            return "Erro: Nome do bucket não definido na variável de ambiente BUCKET_NAME.", 500

        logging.info(f"Nome do bucket: {bucket_name}")

        # Inicializar el cliente de Google Cloud Storage
        storage_client = storage.Client()
        logging.info("Conectado ao serviço do Google Cloud Storage.")

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob("ibovespa/ibovespa_data.csv")
        logging.info("Blob configurado: ibovespa/ibovespa_data.csv")

        # Configuraciones de BigQuery
        PROJECT_ID = "tech-challenge-ibovespa-new"
        DATASET_ID = "ibovespa_dataset"
        TABLE_ID = "ibovespa_table"
        TABLE_URI = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

        # Inicializar el cliente de BigQuery
        client = bigquery.Client()
        logging.info("Conectado ao BigQuery.")

        # Descargar datos del IBOVESPA usando yfinance
        ticker = "^BVSP"
        logging.info(f"Baixando dados do ticker: {ticker}")
        data = yf.download(ticker, period="10y", interval="1d")

        if data.empty:
            logging.error("Não foi possível baixar dados do IBOVESPA.")
            return "Erro: Não foi possível baixar dados do IBOVESPA.", 500

        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')
        df = data[["Date", "Open", "High", "Low", "Close", "Volume"]]

        logging.info("Dados baixados e formatados corretamente.")

        # Sanitizar los datos para evitar tuplas
        def sanitize_data(df):
            for column in df.columns:
                df[column] = df[column].apply(lambda x: str(x) if isinstance(x, tuple) else x)
            return df

        df = sanitize_data(df)

        # Insertar los datos en BigQuery
        data_records = df.to_dict(orient='records')
        table = client.get_table(TABLE_URI)
        errors = client.insert_rows(table, data_records)
        if errors:
            logging.error(f"Errores al insertar datos en BigQuery: {errors}")
            return f"Error al insertar datos en BigQuery: {errors}", 500
        logging.info("Dados inseridos com sucesso no BigQuery.")

        # Guardar los datos en Google Cloud Storage
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")
        logging.info(f"Arquivo ibovespa_data.csv subido com sucesso ao bucket: gs://{bucket_name}.")

        return f"Arquivo ibovespa_data.csv atualizado com sucesso em gs://{bucket_name} e dados inseridos no BigQuery."

    except Exception as e:
        logging.error(f"Erro durante a execução: {e}")
        return f"Erro ao atualizar os dados: {e}", 500
