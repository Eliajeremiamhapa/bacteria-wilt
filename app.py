import os
import numpy as np
import onnxruntime as ort
import threading
import time
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)  # Muhimu sana kwa ajili ya Mobile App na Web integration yako

# 1. MFUMO WA DYNAMIC PATH KWA AJILI YA RENDER (HAKUNA D:\ TENA)
# Mfumo huu unasoma faili popote lilipo mradi wako (Local PC au Render Cloud)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'bacteria wilt.onnx')

if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Faili la model halipatikani kwenye njia hii: {MODEL_PATH}")
    print("[HINT] Hakikisha faili la 'bacteria wilt.onnx' lipo kwenye folda moja na hii app.py")

# Anzisha ONNX Inference Session
# Weka CPU execution provider ili isilete errors kwenye seva za bure za Render
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# 2. LABELS ZA MODEL YA BACTERIAL WILT
class_names = ['Bacterial Wilt', 'Healthy']

# --- LOGIC YA KUAMSHA SERVER (KEEP-ALIVE) ---
def keep_alive():
    """Inapiga picha server kila baada ya dakika 10 kuzuia isilale kwenye Render (Free Tier)"""
    while True:
        try:
            host = os.environ.get('RENDER_EXTERNAL_HOSTNAME')
            if host:
                url = f"https://{host}/health"
                requests.get(url, timeout=5)
                print("Keep-alive: Ping sent successfully!")
        except Exception as e:
            print(f"Keep-alive error: {e}")
        time.sleep(600)  # Dakika 10

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "message": "I am awake!"}), 200
# --------------------------------------------

# MWONGOZO WA NJIA KUU (HOME API ENDPOINT)
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "Welcome to Bacterial Wilt Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Upload image file with key 'file'",
            "/health": "GET - Check API status"
        }
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded under key "file"'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # 3. PREPROCESSING (PYTORCH COOPERATION SYSTEM)
        img = Image.open(file).convert('RGB')
        img = img.resize((224, 224))
        
        # Geuza kwenda numpy array na gawa kwa 255.0 
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # ImageNet Normalization ya PyTorch
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        
        # Badilisha muundo kutoka [H, W, C] kwenda [C, H, W]
        img_array = np.transpose(img_array, (2, 0, 1))
        
        # Ongeza Batch dimension -> [1, 3, 224, 224]
        img_array = np.expand_dims(img_array, axis=0)

        # 4. RUN INFERENCE KWA ONNX
        outputs = session.run(None, {input_name: img_array})
        predictions = np.squeeze(outputs[0])  

        # 5. KOKOTOA MAJIBU
        if predictions.ndim == 0 or (predictions.ndim == 1 and len(predictions) == 1):
            score = float(predictions) if predictions.ndim == 0 else float(predictions[0])
            sigmoid_score = 1 / (1 + np.exp(-score))
            
            if sigmoid_score > 0.5:
                result = class_names[1] if len(class_names) > 1 else "Healthy"
                confidence = sigmoid_score * 100
            else:
                result = class_names[0]
                confidence = (1.0 - sigmoid_score) * 100
        else:
            exp_preds = np.exp(predictions - np.max(predictions))  
            probabilities = exp_preds / np.sum(exp_preds)
            
            highest_idx = np.argmax(probabilities)
            if highest_idx >= len(class_names):
                highest_idx = len(class_names) - 1
                
            result = class_names[highest_idx]
            confidence = probabilities[highest_idx] * 100

        # RETURNING STANDARDIZED API JSON RESPONSE
        return jsonify({
            'success': True,
            'prediction': result,
            'confidence': f"{confidence:.2f}%"
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    if os.environ.get('RENDER'):
        threading.Thread(target=keep_alive, daemon=True).start()
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)