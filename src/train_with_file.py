"""Train models using a specific CSV file path.
Usage: python src/train_with_file.py data/zimbabwe_maize_yield_realvars.csv
"""
import sys
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train_models(df, target='Yield_kg_per_ha'):
    features = ['Rainfall_mm', 'Average_Temperature_C', 'Fertilizer_kg_per_ha', 'Area_Harvested_Ha']
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    models = {'LinearRegression': lr, 'RandomForest': rf}
    results = {}
    for name, m in models.items():
        preds = m.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results[name] = {'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2), 'preds': preds}

    return models, results, X_test, y_test


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python src/train_with_file.py <csv_path>')
        sys.exit(1)
    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f'File not found: {csv_path}')
        sys.exit(1)
    df = pd.read_csv(csv_path)
    models, results, X_test, y_test = train_models(df)
    # choose best by RMSE
    best = min(results.items(), key=lambda kv: kv[1]['rmse'])[0]
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'maize_yield_model.pkl')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(models[best], out_path)
    print('Training results:')
    for k, v in results.items():
        print(f"{k}: RMSE={v['rmse']:.2f}, MAE={v['mae']:.2f}, R2={v['r2']:.3f}")
    print(f"Best model: {best} saved to {out_path}")
