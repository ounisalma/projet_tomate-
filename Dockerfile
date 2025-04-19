# Utiliser une image Python officielle
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copier l'application Flask
COPY . /app

# Définir la variable d'environnement pour Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Exposer le port
EXPOSE 5000

# Lancer l'application Flask
CMD ["flask", "run"]
