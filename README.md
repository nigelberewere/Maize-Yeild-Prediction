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
  - `zimbabwe_maize_yield.csv` - Sample synthetic dataset (46 records)
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
Suggested real data sources for a full study:
- FAOSTAT: maize yield, production, area harvested
- World Bank: fertilizer consumption, agricultural land, cereal yield
- NASA POWER: rainfall and temperature data
- ZIMSTAT or Zimbabwe Ministry of Agriculture reports for local statistics

Note: The included `zimbabwe_maize_yield.csv` is synthetic sample data to bootstrap development. Replace with real data when available.

## Model workflow
1. Data loading and cleaning (`src/data_preprocessing.py`).
2. Exploratory data analysis (notebook).
3. Train Linear Regression and Random Forest models (`src/train_model.py`).
4. Evaluate models using RMSE, MAE, and R² (`src/evaluate_model.py`).
5. Save best model and use `src/predict.py` for predictions.

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

## License
MIT License
