import os
import numpy as np
from io import BytesIO
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import gc

app = Flask(__name__)

# تحميل النموذج مرة واحدة فقط عند تشغيل التطبيق
model = None  # النموذج يتم تحميله عند الحاجة فقط
class_names = ['Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_healthy']

# تحميل النموذج المحسن أو المحول
def load_model_once():
    global model
    if model is None:
        model = load_model("modele_tomate (1).h5")
        print("LOADED MODEL ✅")
    return model

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # تحميل وتحضير الصورة مع تصغير الحجم
        img = image.load_img(BytesIO(file.read()), target_size=(128, 128))  # تصغير الحجم إلى 128x128
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        model = load_model_once()  # تحميل النموذج فقط عند الحاجة

        # التنبؤ بالصنف
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        result = {
            'class': int(predicted_class),
            'confidence': float(np.max(prediction)),
            'class_name': class_names[predicted_class]
        }

        # تنظيف الذاكرة
        del img_array, img, prediction
        gc.collect()

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
