"""Flask web app for maize yield prediction."""
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load model at startup
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'maize_yield_model.pkl')
FEATURES = ['Rainfall_mm', 'Average_Temperature_C', 'Fertilizer_kg_per_ha', 'Area_Harvested_Ha']
model = joblib.load(MODEL_PATH)


@app.route('/')
def index():
    # Try to load the detailed dataset notes from DATASETS.md
    datasets_md = os.path.join(PROJECT_ROOT, 'DATASETS.md')
    data_info_text = ''
    if os.path.exists(datasets_md):
        try:
            with open(datasets_md, 'r', encoding='utf-8') as f:
                data_info_text = f.read()
        except Exception:
            data_info_text = ''

    data_sources = [
        'Rainfall: Open-Meteo (country average) / NASA POWER (legacy)',
        'Temperature: Open-Meteo ERA5-Land (multiple locations average)',
        'Agriculture: World Bank indicators (fertilizer, cereal yield, area)'
    ]
    return render_template('index.html', data_sources=data_sources, data_info_text=data_info_text)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        rainfall = float(data.get('rainfall', 700))
        temperature = float(data.get('temperature', 22))
        fertilizer = float(data.get('fertilizer', 80))
        area = float(data.get('area', 380000))

        X = pd.DataFrame([{
            'Rainfall_mm': rainfall,
            'Average_Temperature_C': temperature,
            'Fertilizer_kg_per_ha': fertilizer,
            'Area_Harvested_Ha': area,
        }], columns=FEATURES)
        yield_kg_per_ha = max(0.0, float(model.predict(X)[0]))
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
