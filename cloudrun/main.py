import pandas as pd
import yfinance as yf
from google.cloud import bigquery
import logging
import os

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
        PROJECT_ID = "tech-challenge-ibovespa-new"
        DATASET_ID = "ibovespa_dataset"
        TABLE_ID = "ibovespa_table"
        TABLE_URI = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

        # Inicializa o cliente BigQuery
        client = bigquery.Client()
        logging.info("Conectado ao BigQuery.")

        # --- Obtener datos de YFinance PRIMERO ---
        ticker = "^BVSP"
        logging.info(f"Baixando dados do ticker: {ticker}")
        data = yf.download(ticker, period="1d", interval="1d") #Solo un dia para definir Schema

        if data.empty:
           logging.error("Não foi possível baixar dados do IBOVESPA.")
           return "Erro: Não foi possível baixar dados do IBOVESPA.", 500

        data.reset_index(inplace=True)
        # --- CREACIÓN DE TABLA (si no existe) ---
        try:
            # Intenta obtener la tabla.
            client.get_table(TABLE_URI)
            logging.info(f"La tabla {TABLE_URI} ya existe.")
        except:
            # Si la tabla NO existe, la creamos.
            logging.info(f"Creando la tabla {TABLE_URI}...")

            # --- Definir el esquema EXPLÍCITAMENTE ---
            schema = [
                bigquery.SchemaField("Date", "DATE"),
                bigquery.SchemaField("Open_Price", "FLOAT"),
                bigquery.SchemaField("High_Price", "FLOAT"),
                bigquery.SchemaField("Low_Price", "FLOAT"),
                bigquery.SchemaField("Close_Price", "FLOAT"),
                bigquery.SchemaField("Volume", "INTEGER"),
            ]

            table = bigquery.Table(TABLE_URI, schema=schema)
            client.create_table(table)
            logging.info(f"Tabla {TABLE_URI} creada correctamente.")
        # --- FIN CREACIÓN DE TABLA ---

        # --- AHORA descargamos los datos completos (10 años) ---
        data = yf.download(ticker, period="10y", interval="1d")
        if data.empty:
            logging.error("Não foi possível baixar dados do IBOVESPA.")
            return "Erro: Não foi possível baixar dados do IBOVESPA.", 500
        data.reset_index(inplace=True)


        # Filtrar y renomear colunas relevantes
        df = data[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df = df.rename(columns={
            "Date": "Date",
            "Open": "Open_Price",
            "High": "High_Price",
            "Low": "Low_Price",
            "Close": "Close_Price",
            "Volume": "Volume"
        })
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

        logging.info("Dados baixados e formatados corretamente.")

        # --- MANEJO DE VALORES FALTANTES ---
        df = df.replace({(): None})  # Reemplaza tuplas vacías por None
        df = df.fillna({
            'Open_Price': 0.0,  # O el valor que consideres apropiado
            'High_Price': 0.0,
            'Low_Price': 0.0,
            'Close_Price': 0.0,
            'Volume': 0,
        })

        # Converter valores para tipos compatíveis com BigQuery
        df = df.astype({
            "Open_Price": float,
            "High_Price": float,
            "Low_Price": float,
            "Close_Price": float,
            "Volume": int
        })

        # --- Depuración (opcional, puedes quitarla una vez que funcione) ---
        #for col in df.columns:
        #    print(f"Tipos de datos en la columna {col}: {df[col].apply(type).unique()}")

        # Converter DataFrame para lista de dicionários
        data_records = df.to_dict(orient='records')


        # --- Inserción de datos ---
        table_ref = client.get_table(TABLE_URI)
        errors = client.insert_rows(table_ref, data_records)
        if not errors:
            logging.info("Dados inseridos com sucesso no BigQuery.")
            return "Dados inseridos com sucesso no BigQuery."
        else:
            logging.error(f"Ocorreram erros ao inserir dados no BigQuery: {errors}")
            return f"Ocorreram erros ao inserir dados no BigQuery: {errors}", 500

    except Exception as e:
        logging.error(f"Erro durante a execução: {e}")
        return f"Erro ao atualizar os dados: {e}", 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
