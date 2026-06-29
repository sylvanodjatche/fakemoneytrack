from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import joblib
import pywt
import cv2
import base64
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import io

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'model_svm_banknote.pkl'

def get_model():
    if not os.path.exists(MODEL_PATH):
        train_and_save()
    return joblib.load(MODEL_PATH)

def train_and_save():
    columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
    local_file = 'data_banknote_authentication.txt'
    if os.path.exists(local_file):
        df = pd.read_csv(local_file, header=None, names=columns)
    else:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
        df = pd.read_csv(url, header=None, names=columns)
        df.to_csv(local_file, index=False, header=False)
    X = df.drop('class', axis=1)
    y = df['class']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    clf = SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced', probability=True)
    clf.fit(X_train, y_train)
    joblib.dump(clf, MODEL_PATH)

def extract_features(image_np):
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_np
    gray = cv2.resize(gray, (256, 256))
    coeffs = pywt.dwt2(gray.astype(float), 'haar')
    cA, (cH, cV, cD) = coeffs
    variance = float(np.var(cA))
    mean = np.mean(cA)
    std = np.std(cA)
    skewness = float(np.mean(((cA - mean) / (std + 1e-10)) ** 3))
    kurtosis = float(np.mean(((cA - mean) / (std + 1e-10)) ** 4) - 3)
    flat = cA.flatten()
    hist, _ = np.histogram(flat, bins=256, range=(flat.min(), flat.max()))
    hist = hist / (np.sum(hist) + 1e-10)
    entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))
    return variance, skewness, kurtosis, entropy

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/manual', methods=['POST'])
def predict_manual():
    data = request.json
    try:
        features = np.array([[
            float(data['variance']),
            float(data['skewness']),
            float(data['curtosis']),
            float(data['entropy'])
        ]])
        model = get_model()
        pred = int(model.predict(features)[0])
        proba = model.predict_proba(features)[0].tolist()
        return jsonify({
            'prediction': pred,
            'probability_authentic': round(proba[0] * 100, 2),
            'probability_fake': round(proba[1] * 100, 2),
            'features': {
                'variance': round(float(data['variance']), 4),
                'skewness': round(float(data['skewness']), 4),
                'curtosis': round(float(data['curtosis']), 4),
                'entropy': round(float(data['entropy']), 4)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/image', methods=['POST'])
def predict_image():
    try:
        data = request.json
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'Image invalide'}), 400
        var, skew, curt, ent = extract_features(image)
        features = np.array([[var, skew, curt, ent]])
        model = get_model()
        pred = int(model.predict(features)[0])
        proba = model.predict_proba(features)[0].tolist()
        return jsonify({
            'prediction': pred,
            'probability_authentic': round(proba[0] * 100, 2),
            'probability_fake': round(proba[1] * 100, 2),
            'features': {
                'variance': round(var, 4),
                'skewness': round(skew, 4),
                'curtosis': round(curt, 4),
                'entropy': round(ent, 4)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    get_model()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
