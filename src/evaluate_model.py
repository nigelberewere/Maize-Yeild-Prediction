"""Evaluate a saved model on the processed test set and produce comparison graphs."""
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data(path):
    return pd.read_csv(path)


def evaluate(model_path, processed_csv_path):
    model = joblib.load(model_path)
    df = load_data(processed_csv_path)
    features = ['Rainfall_mm', 'Average_Temperature_C', 'Fertilizer_kg_per_ha', 'Area_Harvested_Ha']
    X = df[features]
    y = df['Yield_kg_per_ha']
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    return {'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2)}, y, preds


def plot_actual_vs_pred(y, preds, out_file):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y, y=preds)
    plt.xlabel('Actual Yield (kg/ha)')
    plt.ylabel('Predicted Yield (kg/ha)')
    plt.title('Actual vs Predicted Yield')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, 'models', 'maize_yield_model.pkl')
    processed = os.path.join(project_root, 'data', 'processed', 'zimbabwe_maize_yield_processed.csv')
    graph_dir = os.path.join(project_root, 'reports', 'graphs')
    os.makedirs(graph_dir, exist_ok=True)
    metrics, y, preds = evaluate(model_path, processed)
    print('Evaluation metrics:')
    print(metrics)
    plot_actual_vs_pred(y, preds, os.path.join(graph_dir, 'actual_vs_predicted_full.png'))
