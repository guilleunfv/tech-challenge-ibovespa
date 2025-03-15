import pandas as pd
import io
import os
from google.cloud import storage
import yfinance as yf  # Importa yfinance
from flask import Flask, request
import logging

app = Flask(__name__)

print("¡Este es un c555ambio de prueba!")
# Configura logging (opcional, pero recomendado)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

@app.route("/", methods=['GET'])
def atualizar_dados_ibovespa():
    """Baixa os dados do IBOVESPA do yfinance e salva no Google Cloud Storage."""
    try:
        logging.info("Iniciando la función atualizar_dados_ibovespa...")

        # Validar la variable de entorno BUCKET_NAME
        bucket_name = os.environ.get("BUCKET_NAME")
        if not bucket_name:
            logging.error("El nombre del bucket no está configurado en la variable de entorno BUCKET_NAME.")
            return "Error: Nome do bucket não definido na variável de ambiente BUCKET_NAME.", 500

        logging.info(f"Nombre del bucket: {bucket_name}")

        # Inicializar el cliente de Google Cloud Storage
        storage_client = storage.Client()
        logging.info("Conectado al servicio de Google Cloud Storage.")

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob("ibovespa/ibovespa_data.csv")  # Nombre del archivo en el bucket
        logging.info("Blob configurado: ibovespa/ibovespa_data.csv")

        # Descargar datos del IBOVESPA usando yfinance
        ticker = "^BVSP"
        logging.info(f"Descargando datos del ticker: {ticker}")
        data = yf.download(ticker, period="10y", interval="1d")
        if data.empty:
            logging.error("No se pudieron descargar datos del IBOVESPA.")
            return "Error: No se pudieron descargar datos del IBOVESPA.", 500

        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')  # Formatear fechas
        logging.info("Datos descargados y formateados correctamente.")

        # Guardar los datos en Google Cloud Storage
        csv_buffer = io.StringIO()
        data.to_csv(csv_buffer, index=False)
        blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")
        logging.info(f"Archivo ibovespa_data.csv subido con éxito al bucket: gs://{bucket_name}.")

        return f"Archivo ibovespa_data.csv actualizado con éxito en gs://{bucket_name}."

    except Exception as e:
        logging.error(f"Error durante la ejecución: {e}")
        return f"Error al actualizar los datos: {e}", 500

# Configuración del puerto
port = int(os.environ.get("PORT", 8080))
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=port)
