"""Train and compare models on hybrid vs real-variables datasets.
Saves models to `models/model_hybrid.pkl` and `models/model_real.pkl` and
writes a CSV report to `reports/model_comparison.csv`.
"""
import os
import pandas as pd

from train_model import TARGET, save_best_model, train_models

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

def train_and_eval(path):
    df = pd.read_csv(path)
    models, results, _, _ = train_models(df, target=TARGET)
    return results, models

# Hybrid (explicit regenerated file when available)
print(f'Training on hybrid dataset ({hybrid_path})...')
hybrid_results, hybrid_models = train_and_eval(hybrid_path)
hybrid_best, _ = save_best_model(hybrid_models, hybrid_results, os.path.join(models_dir, 'model_hybrid.pkl'))

# Real variables
print(f'Training on real-variables dataset ({real_path})...')
real_results, real_models = train_and_eval(real_path)
real_best, _ = save_best_model(real_models, real_results, os.path.join(models_dir, 'model_real.pkl'))

# Save comparison
rows = []
for dataset_name, res in [('hybrid', hybrid_results), ('real', real_results)]:
    for model_name, metrics in res.items():
        rows.append({
            'dataset': dataset_name,
            'model': model_name,
            'selected_for_dataset': (
                model_name == hybrid_best if dataset_name == 'hybrid' else model_name == real_best
            ),
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r2': metrics['r2'],
            'cv_rmse': metrics.get('cv_rmse'),
            'cv_mae': metrics.get('cv_mae'),
            'cv_r2': metrics.get('cv_r2'),
        })
report_df = pd.DataFrame(rows)
report_path = os.path.join(reports_dir, 'model_comparison.csv')
report_df.to_csv(report_path, index=False)
print(f'Comparison report saved to {report_path}')
print(report_df)
