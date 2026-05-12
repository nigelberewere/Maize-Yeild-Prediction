# AI-Based Maize Yield Prediction Model for Zimbabwe Farmers

This repository contains a computational modelling mini project to predict maize yield in Zimbabwe using agricultural and climate-related variables.

**Problem statement:** Build a data-driven model that predicts maize yield (kg/ha) using rainfall, temperature, fertilizer use, and area harvested.

**Aim:** Provide an interpretable and reproducible modelling workflow suitable for a university mini project in computational modelling.

## Project Structure

- `app.py` - Flask web application (run with `python app.py`)
- `templates/`
  - `index.html` - Modern responsive web interface
- `data/`
  - `raw/` - Raw input datasets
  - `processed/` - Cleaned and processed datasets
  - `zimbabwe_maize_yield.csv` - Main training dataset (merged Zimbabwe agricultural data)
- `notebooks/`
  - `maize_yield_analysis.ipynb` - Notebook for exploration and visualisation
- `src/`
  - `data_preprocessing.py` - Load and clean data
  - `train_model.py` - Train LR and RF models and save best model
  - `predict.py` - CLI to predict yield from input features
  - `evaluate_model.py` - Evaluate saved model on test data
- `models/`
  - `maize_yield_model.pkl` - Saved model (created after training)
- `reports/`
  - `graphs/` - Generated figures
  - `project_report.md` - Project report template
- `tests/` - Unit and integration tests
- `requirements.txt` - Python dependencies
- `README.md` - This file
- `WEB_UI.md` - Web interface documentation

## Datasets and sources
Current data sources used in the project:
- NASA POWER rainfall: used to generate the baseline rainfall series for Zimbabwe
- World Bank agriculture indicators: fertilizer consumption, cereal yield, cereal production, and land under cereal production
- Open-Meteo Historical Weather API (ERA5-Land): actual Zimbabwe temperature series aggregated from multiple reference points

The current training file is the merged dataset in `data/zimbabwe_maize_yield_realvars_temp.csv`, which combines real rainfall, real temperature, and available Zimbabwe agriculture indicators.

## Model workflow
1. Data loading and cleaning (`src/data_preprocessing.py`).
2. Exploratory data analysis (notebook).
3. Train Linear Regression and Random Forest models (`src/train_model.py` or `src/train_with_file.py`).
4. Compare hybrid versus real-data runs (`src/compare_models.py`).
5. Save the best model and use `src/predict.py` for predictions.

## Algorithms used
- Linear Regression (interpretable baseline)
- Random Forest Regression (non-linear ensemble)

## Evaluation metrics
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of determination)

## How to run (local)

### Setup (One-time)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Option 1: Web Interface (Recommended) 🌐

Run the modern web application with an easy-to-use interface:

```powershell
python app.py
```

Then open your browser to: **http://127.0.0.1:5000**

✨ Features:
- Beautiful responsive design (works on mobile)
- Real-time prediction results
- Input validation with helpful hints
- No command-line knowledge required

### Option 2: Interactive CLI 💬

Answer prompts step by step with helpful defaults:

```powershell
python src/predict.py --interactive
```

You'll be prompted for:
- Annual rainfall (mm) — _Press Enter for default: 700_
- Average temperature (°C) — _Press Enter for default: 22_
- Fertilizer use (kg/ha) — _Press Enter for default: 80_
- Area harvested (hectares) — _Press Enter for default: 380000_

### Option 3: Command-line Flags 🚀

Quick prediction with explicit values:

```powershell
python src/predict.py --rainfall 650 --temp 24 --fert 120 --area 1000000
```

### Data Preprocessing & Model Training

To rebuild the model from scratch:

```powershell
# Preprocess data
python src/data_preprocessing.py

# Train Linear Regression and Random Forest models
python src/train_model.py

# Evaluate model performance
python src/evaluate_model.py
```

### Jupyter Notebook Analysis

For exploratory data analysis and visualization:

```powershell
jupyter notebook notebooks/maize_yield_analysis.ipynb
```

## Expected output
- Trained model saved to `models/maize_yield_model.pkl`.
- Performance metrics printed to console and saved plots in `reports/graphs/`.

## Current model notes
- The current model is trained on the merged real-data pipeline, not the early synthetic bootstrap dataset.
- Temperature now comes from the Open-Meteo historical weather export in `Datasets/download_zimbabwe_temperature_actual_1981.py`.
- The latest model comparison report is saved in `reports/model_comparison.csv`.

## License
MIT License
