"""Flask web app for maize yield prediction."""
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load model at startup
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'maize_yield_model.pkl')
FEATURES = [
    'Rainfall_mm',
    'Average_Temperature_C',
    'Fertilizer_kg_per_ha',
    'ENSO_Index',
    'Soil_Clay_Percent',
    'Soil_pH',
    'Soil_Organic_Carbon_Percent',
]
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
        'Rainfall: Open-Meteo annual aggregates',
        'Temperature: NASA POWER annual temperature series',
        'Fertilizer: World Bank agricultural indicators',
        'ENSO: NOAA Oceanic Niño Index (El Niño / La Niña)',
        'Soil: SoilGrids / Zimbabwe soil estimates',
        'Yield: FAOSTAT maize yield records'
    ]
    return render_template('index.html', data_sources=data_sources, data_info_text=data_info_text)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        rainfall = float(data.get('rainfall', 700))
        temperature = float(data.get('temperature', 22))
        fertilizer = float(data.get('fertilizer', 80))
        enso_index = float(data.get('enso_index', 0.0))
        soil_clay = float(data.get('soil_clay', 28.0))
        soil_ph = float(data.get('soil_ph', 6.0))
        soil_carbon = float(data.get('soil_carbon', 2.0))
        area = float(data.get('area', 380000))

        X = pd.DataFrame([{
            'Rainfall_mm': rainfall,
            'Average_Temperature_C': temperature,
            'Fertilizer_kg_per_ha': fertilizer,
            'ENSO_Index': enso_index,
            'Soil_Clay_Percent': soil_clay,
            'Soil_pH': soil_ph,
            'Soil_Organic_Carbon_Percent': soil_carbon,
        }], columns=FEATURES)
        yield_kg_per_ha = max(0.0, float(model.predict(X)[0]))
        production_tonnes = (yield_kg_per_ha * area) / 1000

        return jsonify({
            'success': True,
            'yield_kg_per_ha': round(yield_kg_per_ha, 2),
            'production_tonnes': round(production_tonnes, 0),
            'area_hectares': area,
            'enso_index': enso_index,
            'soil_clay_percent': soil_clay,
            'soil_ph': soil_ph,
            'soil_organic_carbon_percent': soil_carbon
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
