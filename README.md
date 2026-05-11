# AI-Based Maize Yield Prediction Model for Zimbabwe Farmers

This repository contains a computational modelling mini project to predict maize yield in Zimbabwe using agricultural and climate-related variables.

**Problem statement:** Build a data-driven model that predicts maize yield (kg/ha) using rainfall, temperature, fertilizer use, and area harvested.

**Aim:** Provide an interpretable and reproducible modelling workflow suitable for a university mini project in computational modelling.

## Project Structure

- `data/`
  - `raw/` - Raw input datasets
  - `processed/` - Cleaned and processed datasets
  - `zimbabwe_maize_yield.csv` - Sample synthetic dataset (30 records)
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
- `requirements.txt` - Python dependencies
- `README.md` - This file

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
1. Create a Python environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Preprocess data:

```powershell
python src/data_preprocessing.py
```

3. Train models:

```powershell
python src/train_model.py
```

4. Predict with the saved model (example):

```powershell
python src/predict.py --rainfall 650 --temp 24 --fert 120 --area 1000000
```

5. Evaluate or rerun notebook for analysis: open `notebooks/maize_yield_analysis.ipynb`.

## Expected output
- Trained model saved to `models/maize_yield_model.pkl`.
- Performance metrics printed to console and saved plots in `reports/graphs/`.

## License
MIT License
