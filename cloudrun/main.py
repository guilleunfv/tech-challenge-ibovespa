import pandas as pd
import yfinance as yf
import logging
import os

from flask import Flask

app = Flask(__name__)

# Configura logging GUARDE
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route("/", methods=['GET'])
def atualizar_dados_ibovespa():
    """
    Baixa os dados do IBOVESPA do yfinance.
    """
    try:
        logging.info("Iniciando a função atualizar_dados_ibovespa...")

        # --- Obtener datos de YFinance  ---
        ticker = "^BVSP"
        logging.info(f"Baixando dados do ticker: {ticker}")
        data = yf.download(ticker, period="10y", interval="1d")  # <--- SIN auto_adjust

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


        logging.info("Dados baixados e formatados corretamente.")

        # --- MANEJO DE VALORES FALTANTES ---
        data = data.replace({(): None})
        data = data.fillna({
            'Open': 0.0,
            'High': 0.0,
            'Low': 0.0,
            'Close': 0.0,
            'Adj Close': 0.0,
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
            "Adj Close": float,
            "Volume": int
        })

        # --- VERIFICAR TIPOS DESPUÉS DE ASTYPE ---
        for col in data.columns:
            logging.info(f"Tipo de dato de la columna '{col}': {data[col].dtype}")


        return "Datos descargados y procesados (sin BigQuery)", 200

    except Exception as e:
        logging.error(f"Erro durante a execução: {e}")
        return f"Erro ao atualizar os dados: {e}", 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
