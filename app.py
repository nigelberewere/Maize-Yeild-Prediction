"""Flask web app for maize yield prediction."""
import os
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load model at startup
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'maize_yield_model.pkl')
model = joblib.load(MODEL_PATH)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        rainfall = float(data.get('rainfall', 700))
        temperature = float(data.get('temperature', 22))
        fertilizer = float(data.get('fertilizer', 80))
        area = float(data.get('area', 380000))

        # Predict
        X = np.array([[rainfall, temperature, fertilizer, area]])
        yield_kg_per_ha = float(model.predict(X)[0])
        production_tonnes = (yield_kg_per_ha * area) / 1000

        return jsonify({
            'success': True,
            'yield_kg_per_ha': round(yield_kg_per_ha, 2),
            'production_tonnes': round(production_tonnes, 0),
            'area_hectares': area
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
