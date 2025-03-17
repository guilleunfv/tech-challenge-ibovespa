import pandas as pd
import yfinance as yf
from google.cloud import bigquery
import logging
import os  # Importante para obtener la variable de entorno PORT

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

        # --- CREACIÓN DE TABLA (si no existe) ---
        try:
            table = client.get_table(TABLE_URI)  # Intenta obtener la tabla
            logging.info(f"La tabla {TABLE_URI} ya existe.")
        except:  # Si la tabla no existe, la creamos
            logging.info(f"Creando la tabla {TABLE_URI}...")
            schema = [
                bigquery.SchemaField("Date", "DATE"),
                bigquery.SchemaField("Open_Price", "FLOAT"),
                bigquery.SchemaField("High_Price", "FLOAT"),
                bigquery.SchemaField("Low_Price", "FLOAT"),
                bigquery.SchemaField("Close_Price", "FLOAT"),
                bigquery.SchemaField("Volume", "INTEGER"),
            ]
            table = bigquery.Table(TABLE_URI, schema=schema)
            table = client.create_table(table)
            logging.info(f"Tabla {TABLE_URI} creada correctamente.")
        # --- FIN CREACIÓN DE TABLA ---

        # Baixar dados do IBOVESPA usando yfinance
        ticker = "^BVSP"
        logging.info(f"Baixando dados do ticker: {ticker}")
        data = yf.download(ticker, period="10y", interval="1d")

        if data.empty:
            logging.error("Não foi possível baixar dados do IBOVESPA.")
            return "Erro: Não foi possível baixar dados do IBOVESPA.", 500

        # Resetar índice para transformar o índice de datas em uma coluna
        data.reset_index(inplace=True)

        # Filtrar e renomear colunas relevantes
        df = data[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df = df.rename(columns={
            "Date": "Date",
            "Open": "Open_Price",
            "High": "High_Price",
            "Low": "Low_Price",
            "Close": "Close_Price",
            "Volume": "Volume"
        })
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')  # Formatar datas como string

        logging.info("Dados baixados e formatados corretamente.")

        # Converter valores para tipos compatíveis com BigQuery
        df = df.astype({
            "Open_Price": float,
            "High_Price": float,
            "Low_Price": float,
            "Close_Price": float,
            "Volume": int
        })

        # Converter DataFrame para lista de dicionários
        data_records = df.to_dict(orient='records')

        # Obter referência da tabela no BigQuery
        # table = client.get_table(TABLE_URI) # YA NO ES NECESARIO, se crea o obtiene arriba

        # Inserir dados no BigQuery
        errors = client.insert_rows(table, data_records)  # Usa la variable 'table'
        if not errors:
            logging.info("Dados inseridos com sucesso no BigQuery.")
            return "Dados inseridos com sucesso no BigQuery."
        else:
            logging.error(f"Ocorreram erros ao inserir dados no BigQuery: {errors}")
            return f"Ocorreram erros ao inserir dados no BigQuery: {errors}", 500

    except Exception as e:
        logging.error(f"Erro durante a execução: {e}")
        return f"Erro ao atualizar os dados: {e}", 500
# Este if __name__ solo se ejecuta si corres el script directamente (python main.py).
# No se ejecuta cuando Gunicorn (en Cloud Run) importa el módulo.
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
