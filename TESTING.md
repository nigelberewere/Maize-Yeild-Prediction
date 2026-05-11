# Testing Guide

## Unit Tests

The project includes unit tests to validate preprocessing and model training:

### Run all tests

```powershell
cd .\tests
python -m pytest -v
```

Or with unittest directly:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

### Individual test files

```powershell
python -m unittest tests.test_data_preprocessing -v
python -m unittest tests.test_train_model -v
```

## Integration Tests

### 1. End-to-end pipeline (preprocessing → training → prediction)

```powershell
# Preprocess the raw data
python src/data_preprocessing.py

# Train models and save the best one
python src/train_model.py

# Predict with the saved model
python src/predict.py --rainfall 650 --temp 24 --fert 120 --area 1000000
```

**Expected output:** Predicted maize yield around 1600–1800 kg/ha.

### 2. Notebook exploration

Open and run the notebook for visual EDA:

```powershell
jupyter notebook notebooks/maize_yield_analysis.ipynb
```

## Validation Checks

- **Data integrity**: Check for NaN/missing values
- **Model performance**: Ensure RMSE < 200 kg/ha and R² > 0.8 on test set
- **Prediction consistency**: Same inputs → same outputs
- **File existence**: Verify processed data, models, and graphs are saved

## Troubleshooting

- **Import errors**: Ensure `pip install -r requirements.txt`
- **Path errors**: Run scripts from the project root directory
- **Model not found**: Run training script first with `python src/train_model.py`
