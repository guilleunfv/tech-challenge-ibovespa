import pandas as pd
import yfinance as yf
import logging
import os
import uuid  # Importar uuid

from flask import Flask

app = Flask(__name__)

# Configura logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@app.route("/", methods=['GET'])
def atualizar_dados_ibovespa():
    """
    Baixa os dados do IBOVESPA do yfinance.
    """
    try:
        # --- AÑADIR UN IDENTIFICADOR ÚNICO ---
        unique_id = str(uuid.uuid4())
        logging.info(f"Iniciando a função atualizar_dados_ibovespa... (ID: {unique_id})")

        # --- Obtener datos de YFinance  ---
        ticker = "^BVSP"
        logging.info(f"Baixando dados do ticker: {ticker}")
        data = yf.download(ticker, period="10y", interval="1d",
                           cache=False)  # <--- SIN auto_adjust y CON cache=False

        if data.empty:
            logging.error("Não foi possível baixar dados do IBOVESPA.")
            return "Erro: Não foi possível baixar dados do IBOVESPA.", 500

        logging.info("Datos descargados de yfinance.")  # Punto de control

        # --- Imprimir el schema inicial ---
        logging.info(f"Schema inicial del DataFrame:\n{data.dtypes}")
        logging.info(f"Columnas inicial del DataFrame:\n{data.columns}")
        logging.info(f"Index inicial del DataFrame:\n{data.index}")

        # Verificamos si el index es Datetime, y si no lo es, lo reseteamos.
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.reset_index()
            logging.info("Index reseteado (era DatetimeIndex).")
        else:
            data = data.reset_index(drop=True)
            logging.info("Index reseteado (no era DatetimeIndex).")

        logging.info("Después de reset_index")  # Punto de control
        logging.info(f"Schema después de reset_index:\n{data.dtypes}")
        logging.info(f"Columnas después de reset_index:\n{data.columns}")
        logging.info(f"Index después de reset_index:\n{data.index}")
        # --- MANEJO DE VALORES FALTANTES ---
        data = data.replace({(): None})
        logging.info("Tuplas vacías reemplazadas por None.")
        data = data.fillna({
            'Open': 0.0,
            'High': 0.0,
            'Low': 0.0,
            'Close': 0.0,
            'Adj Close': 0.0,
            'Volume': 0,
        })

        logging.info("Valores NaN rellenados.")  # Punto de control
        logging.info(f"Schema después de fillna:\n{data.dtypes}")

        # ---Asegurar que los nombres de columna sean string---
        data.columns = data.columns.astype(str)
        logging.info("Nombres de columna convertidos a string.")

        # Converter valores para tipos compatíveis con BigQuery

        try:  # Un try-except especifico para el astype
            data = data.astype({
                "Open": float,
                "High": float,
                "Low": float,
                "Close": float,
                "Adj Close": float,
                "Volume": int
            })
            logging.info("astype aplicado correctamente.")
        except Exception as e:
            logging.error(f"Error en astype: {e}")
            return f"Error en astype: {e}", 500

        # --- VERIFICAR TIPOS DESPUÉS DE ASTYPE ---
        for col in data.columns:
            logging.info(f"Tipo de dato de la columna '{col}': {data[col].dtype}")

        # Converter DataFrame para lista de dicionários
        try:
            data_records = data.to_dict(orient='records')
            logging.info("to_dict aplicado correctamente.")
        except Exception as e:
            logging.error(f"Error en to_dict: {e}")
            return f"Error en to_dict: {e}", 500
        return "Datos descargados y procesados (sin BigQuery)", 200

    except Exception as e:
        logging.error(f"Erro durante a execução: {e}")
        return f"Erro ao atualizar os dados: {e}", 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
