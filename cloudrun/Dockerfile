FROM python:3.9-slim-buster

# Este Dockerfile está configurado para desarrollo
# Usa el servidor de desarrollo de Flask

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

ENV BUCKET_NAME=""

CMD ["python", "main.py"]

# Fin del Dockerfile
