FROM python:3.9-slim-buster

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo de requerimientos
COPY cloudrun/requirements.txt .

# Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Instalar Gunicorn
RUN pip install gunicorn

# Copiar el código de la aplicación
COPY cloudrun/main.py .

# Exponer el puerto 8080 para acceder a la aplicación
EXPOSE 8080

# Ejecutar la aplicación Flask usando Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "main:app"]
