import os
import numpy as np
from io import BytesIO
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import gc

app = Flask(__name__)

# تحميل النموذج مرة واحدة فقط عند تشغيل التطبيق
model = load_model("modele_tomate (1).h5")
print("LOADED MODEL ✅")

# أسماء الأصناف حسب ترتيب الدوسيات في التدريب
class_names = ['Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_healthy']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # تحميل وتحضير الصورة
        img = image.load_img(BytesIO(file.read()), target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

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
