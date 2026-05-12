"""Train and compare models on hybrid vs real-variables datasets.
Saves models to `models/model_hybrid.pkl` and `models/model_real.pkl` and
writes a CSV report to `reports/model_comparison.csv`.
"""
import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# prefer an explicit hybrid file (regenerated) to avoid having replaced the main CSV
hybrid_path = os.path.join(PROJECT_ROOT, 'data', 'zimbabwe_maize_yield_hybrid.csv')
if not os.path.exists(hybrid_path):
    hybrid_path = os.path.join(PROJECT_ROOT, 'data', 'zimbabwe_maize_yield.csv')
real_path = os.path.join(PROJECT_ROOT, 'data', 'zimbabwe_maize_yield_realvars_temp.csv')
if not os.path.exists(real_path):
    real_path = os.path.join(PROJECT_ROOT, 'data', 'zimbabwe_maize_yield_realvars.csv')
models_dir = os.path.join(PROJECT_ROOT, 'models')
reports_dir = os.path.join(PROJECT_ROOT, 'reports')

os.makedirs(models_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

FEATURES = ['Rainfall_mm', 'Average_Temperature_C', 'Fertilizer_kg_per_ha', 'Area_Harvested_Ha']
TARGET = 'Yield_kg_per_ha'

def train_and_eval(path):
    df = pd.read_csv(path)
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LinearRegression().fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    results = {}
    for name, m in [('LinearRegression', lr), ('RandomForest', rf)]:
        preds = m.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results[name] = {'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2)}
    return results, lr, rf

# Hybrid (explicit regenerated file when available)
print(f'Training on hybrid dataset ({hybrid_path})...')
hybrid_results, hybrid_lr, hybrid_rf = train_and_eval(hybrid_path)
joblib.dump(hybrid_rf, os.path.join(models_dir, 'model_hybrid.pkl'))

# Real variables
print(f'Training on real-variables dataset ({real_path})...')
real_results, real_lr, real_rf = train_and_eval(real_path)
joblib.dump(real_rf, os.path.join(models_dir, 'model_real.pkl'))

# Save comparison
rows = []
for dataset_name, res in [('hybrid', hybrid_results), ('real', real_results)]:
    for model_name, metrics in res.items():
        rows.append({
            'dataset': dataset_name,
            'model': model_name,
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r2': metrics['r2']
        })
report_df = pd.DataFrame(rows)
report_path = os.path.join(reports_dir, 'model_comparison.csv')
report_df.to_csv(report_path, index=False)
print(f'Comparison report saved to {report_path}')
print(report_df)
