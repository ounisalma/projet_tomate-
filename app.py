import os
import numpy as np
from io import BytesIO
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import gc
import requests

app = Flask(__name__)

# اسم الملف المحلي للنموذج
MODEL_FILENAME = "modele_tomate.h5"

# Google Drive File ID
GDRIVE_FILE_ID = "17POPUvx7l12kwXsPVjhI-YWR-IxNEAh8"

# 📥 دالة لتحميل النموذج من Google Drive بطريقة آمنة
def download_model_from_gdrive(gdrive_id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': gdrive_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': gdrive_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)
    print("✅ Modèle téléchargé et enregistré localement.")
    print(f"📦 Taille du fichier modèle: {os.path.getsize(destination)} octets")

# تحميل النموذج إذا غير موجود محليًا
if not os.path.exists(MODEL_FILENAME):
    print("🔄 Téléchargement du modèle depuis Google Drive...")
    download_model_from_gdrive(GDRIVE_FILE_ID, MODEL_FILENAME)

# تحميل النموذج
model = load_model(MODEL_FILENAME)
print("✅ Modèle chargé")

# أسماء الأصناف
class_names = ['Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_healthy']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = image.load_img(BytesIO(file.read()), target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        result = {
            'class': int(predicted_class),
            'confidence': float(np.max(prediction)),
            'class_name': class_names[predicted_class]
        }

        # تنظيف الموارد
        del img_array, img, prediction
        gc.collect()

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
