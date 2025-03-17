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



        # --- Obtener datos de YFinance  ---

        ticker = "^BVSP"

        logging.info(f"Baixando dados do ticker: {ticker}")

        data = yf.download(ticker, period="10y", interval="1d")  # <--- SIN auto_adjust



        if data.empty:

           logging.error("Não foi possível baixar dados do IBOVESPA.")

           return "Erro: Não foi possível baixar dados do IBOVESPA.", 500



        # --- Imprimir el schema inicial ---

        logging.info(f"Schema inicial del DataFrame:\n{data.dtypes}")

        logging.info(f"Columnas inicial del DataFrame:\n{data.columns}")

        logging.info(f"Index inicial del DataFrame:\n{data.index}")



        #Verificamos si el index es Datetime, y si no lo es, lo reseteamos.

        if isinstance(data.index, pd.DatetimeIndex):

            data = data.reset_index()

        else:

            data = data.reset_index(drop=True)



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

                bigquery.SchemaField("Open", "FLOAT"),

                bigquery.SchemaField("High", "FLOAT"),

                bigquery.SchemaField("Low", "FLOAT"),

                bigquery.SchemaField("Close", "FLOAT"),

                bigquery.SchemaField("Adj Close", "FLOAT"), #  <-  Añadimos Adj Close

                bigquery.SchemaField("Volume", "INTEGER"),

            ]



            table = bigquery.Table(TABLE_URI, schema=schema)

            client.create_table(table)

            logging.info(f"Tabla {TABLE_URI} creada correctamente.")

        # --- FIN CREACIÓN DE TABLA ---



        # --- YA NO ES NECESARIO RENOMBRAR ---

        logging.info("Dados baixados e formatados corretamente.")



        # --- MANEJO DE VALORES FALTANTES ---

        data = data.replace({(): None})

        data = data.fillna({

            'Open': 0.0,

            'High': 0.0,

            'Low': 0.0,

            'Close': 0.0,

            'Adj Close': 0.0, #  <-  Añadimos Adj Close

            'Volume': 0,

        })



        # ---Asegurar que los nombres de columna sean string---

        data.columns = data.columns.astype(str)



        # Converter valores para tipos compatíveis con BigQuery

        data = data.astype({

            "Open": float,

            "High": float,

            "Low": float,

            "Close": float,

            "Adj Close": float,  #  <-  Añadimos Adj Close

            "Volume": int

        })



        # Converter DataFrame para lista de dicionários

        data_records = data.to_dict(orient='records')



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
