"""Train models using a specific CSV file path.
Usage: python src/train_with_file.py data/zimbabwe_maize_yield_realvars.csv
"""
import sys
import os
import pandas as pd

from train_model import save_best_model, save_metrics_report, train_models


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
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(project_root, 'models', 'maize_yield_model.pkl')
    best, out_path = save_best_model(models, results, out_path)
    metrics_path = save_metrics_report(results, os.path.join(project_root, 'reports', 'model_training_metrics.csv'))
    print('Training results:')
    for k, v in results.items():
        print(
            f"{k}: "
            f"CV_RMSE={v['cv_rmse']:.2f}, CV_MAE={v['cv_mae']:.2f}, CV_R2={v['cv_r2']:.3f}; "
            f"Holdout_RMSE={v['rmse']:.2f}, Holdout_MAE={v['mae']:.2f}, Holdout_R2={v['r2']:.3f}"
        )
    print(f"Best model: {best} saved to {out_path}")
    print(f"Metrics report saved to {metrics_path}")
