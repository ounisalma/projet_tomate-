import os
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from flask import Flask, jsonify, request
from io import BytesIO

app = Flask(__name__)

# Charger le modèle
model = load_model("modele_tomate (1).h5")
print("LOADED MODEL ✅")

# Liste des classes selon l'ordre des dossiers dans l'entraînement
class_names = ['Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_healthy']

@app.route('/predict', methods=['POST'])
def predict():
    # Vérifier si un fichier a été envoyé dans la requête
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # Vérifier si le nom de fichier est vide
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Charger l'image et la prétraiter
    img = image.load_img(BytesIO(file.read()), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normaliser l'image
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter la dimension du batch

    # Prédire la classe de l'image
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Retourner la réponse sous forme de JSON
    return jsonify({
        'class': int(predicted_class),  # Convertir en entier
        'confidence': float(np.max(prediction)),  # Convertir en float
        'class_name': class_names[predicted_class]
    })

if __name__ == '__main__':
    # Démarrer l'application Flask sur 0.0.0.0 pour accepter les connexions externes
    app.run(host='0.0.0.0', port=10000)
