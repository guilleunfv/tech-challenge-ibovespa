import pandas as pd
import io
import os
from google.cloud import storage
import yfinance as yf  # Importa yfinance
from flask import Flask, request

app = Flask(__name__)

@app.route("/", methods=['GET'])
def atualizar_dados_ibovespa():
    """Baixa os dados do IBOVESPA do yfinance e salva no Google Cloud Storage."""
    try:
        # Extrai el nombre del bucket de las variables de entorno
        bucket_name = os.environ.get("BUCKET_NAME")
        if not bucket_name:
            return "Error: Nome do bucket não definido na variável de ambiente BUCKET_NAME.", 500

        # Inicializar o cliente de Google Cloud Storage
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob("ibovespa/ibovespa_data.csv")  # Nombre del archivo en el bucket

        # Baixar dados do IBOVESPA usando yfinance
        ticker = "^BVSP"
        data = yf.download(ticker, period="10y", interval="1d")
        data.reset_index(inplace=True)

        # Salvar os dados no Google Cloud Storage
        csv_buffer = io.StringIO()
        data.to_csv(csv_buffer, index=False)
        blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")

        print(f"Arquivo ibovespa_data.csv atualizado com sucesso em gs://{bucket_name}.")
        return f"Arquivo ibovespa_data.csv atualizado com sucesso em gs://{bucket_name}."

    except Exception as e:
        print(f"Error al actualizar los datos: {e}")
        return f"Error al actualizar los datos: {e}", 500

# Esta parte es importante para que el contenedor funcione correctamente
port = int(os.environ.get("PORT", 8080))
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=port)
