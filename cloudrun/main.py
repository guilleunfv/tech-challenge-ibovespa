import pandas as pd
import io
import os
from google.cloud import storage
from google.cloud import bigquery
import yfinance as yf
from flask import Flask
import logging

app = Flask(__name__)

# Configura logging (opcional, mas recomendado)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

@app.route("/", methods=['GET'])
def atualizar_dados_ibovespa():
    """
    Baixa os dados do IBOVESPA do yfinance, salva no Google Cloud Storage e insere no BigQuery.
    """
    try:
        logging.info("Iniciando a função atualizar_dados_ibovespa...")

        # Validar a variável de ambiente BUCKET_NAME
        bucket_name = os.environ.get("BUCKET_NAME")
        if not bucket_name:
            logging.error("O nome do bucket não está configurado na variável de ambiente BUCKET_NAME.")
            return "Erro: Nome do bucket não definido na variável de ambiente BUCKET_NAME.", 500

        logging.info(f"Nome do bucket: {bucket_name}")

        # Inicializar o cliente do Google Cloud Storage
        storage_client = storage.Client()
        logging.info("Conectado ao serviço do Google Cloud Storage.")

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob("ibovespa/ibovespa_data.csv")  # Nome do arquivo no bucket
        logging.info("Blob configurado: ibovespa/ibovespa_data.csv")

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
        data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')  # Formatar datas
        df = data[["Date", "Open", "High", "Low", "Close", "Volume"]]

        logging.info("Dados baixados e formatados corretamente.")

        # Inserir os dados no BigQuery
        data_records = df.to_dict(orient='records')
        table = client.get_table(TABLE_URI)  # Use TABLE_URI aqui
        errors = client.insert_rows(table, data_records)
        if errors == []:
            logging.info("Dados inseridos com sucesso no BigQuery.")
        else:
            logging.error(f"Ocorreram erros ao inserir dados no BigQuery: {errors}")
            return f"Ocorreram erros ao inserir dados no BigQuery: {errors}", 500

        # Salvar os dados em Google Cloud Storage (Mantendo o CSV!)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")
        logging.info(f"Arquivo ibovespa_data.csv subido com sucesso ao bucket: gs://{bucket_name}.")

        return f"Arquivo ibovespa_data.csv atualizado com sucesso em gs://{bucket_name} e dados inseridos no BigQuery."

    except Exception as e:
        logging.error(f"Erro durante a execução: {e}")
        return f"Erro ao atualizar os dados: {e}", 500

# A linha abaixo deve ser removida para a produção.
# O Gunicorn irá servir a aplicação.
# port = int(os.environ.get("PORT", 8080))
# if __name__ == "__main__":
#    app.run(debug=True, host="0.0.0.0", port=port)
