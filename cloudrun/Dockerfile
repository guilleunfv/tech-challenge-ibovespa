FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn  # Añade Gunicorn

COPY cloudrun/main.py .  # Corrige la ruta y el nombre del archivo

ENV BUCKET_NAME=""

EXPOSE 8080 # Importante para que Cloud Run saiba a porta

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "main:app"]
