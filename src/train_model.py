"""Train and compare Linear Regression and Random Forest models."""
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data(path):
    return pd.read_csv(path)


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


def save_best_model(models, results, out_path=os.path.join('..', 'models', 'maize_yield_model.pkl')):
    # choose best by RMSE
    best = min(results.items(), key=lambda kv: kv[1]['rmse'])[0]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(models[best], out_path)
    return best, out_path


def plot_actual_vs_pred(y_test, preds, out_file):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_test, y=preds)
    plt.xlabel('Actual Yield (kg/ha)')
    plt.ylabel('Predicted Yield (kg/ha)')
    plt.title('Actual vs Predicted Yield')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_path = os.path.join(project_root, 'data', 'processed', 'zimbabwe_maize_yield_processed.csv')
    if not os.path.exists(processed_path):
        # fall back to raw if processed not present
        processed_path = os.path.join(project_root, 'data', 'zimbabwe_maize_yield.csv')
    df = load_data(processed_path)
    models, results, X_test, y_test = train_models(df)
    model_out = os.path.join(project_root, 'models', 'maize_yield_model.pkl')
    best_name, model_path = save_best_model(models, results, out_path=model_out)
    print('Training results:')
    for k, v in results.items():
        print(f"{k}: RMSE={v['rmse']:.2f}, MAE={v['mae']:.2f}, R2={v['r2']:.3f}")
    print(f"Best model: {best_name} saved to {model_path}")
    # save plot
    graph_dir = os.path.join(project_root, 'reports', 'graphs')
    os.makedirs(graph_dir, exist_ok=True)
    plot_actual_vs_pred(y_test, results[best_name]['preds'], os.path.join(graph_dir, 'actual_vs_predicted.png'))
