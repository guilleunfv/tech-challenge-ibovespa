# Usa la imagen oficial de Python como base
FROM python:3.9-slim

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia los archivos de requerimientos al contenedor
COPY ./cloudrun/requirements.txt /app/requirements.txt

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código fuente al contenedor
COPY ./cloudrun /app

# Define el comando para ejecutar la aplicación
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
