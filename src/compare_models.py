"""Train and evaluate models on empirical Zimbabwe maize yield data only.

This script trains multiple models on the real, integrated historical data
that combines actual rainfall, temperature, fertilizer, ENSO, and soil data.
Synthetic or hybrid datasets are no longer used for final model evaluation
to prevent false confidence in model performance.

The final best model is saved for deployment.
"""
import os
import pandas as pd

from train_model import TARGET, save_best_model, train_models, save_metrics_report, plot_actual_vs_pred

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(PROJECT_ROOT, 'models')
reports_dir = os.path.join(PROJECT_ROOT, 'reports')

os.makedirs(models_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

def train_and_eval(path):
    """Train all model candidates on a dataset and return results."""
    df = pd.read_csv(path)
    models, results, X_test, y_test = train_models(df, target=TARGET)
    return results, models, X_test, y_test


# REAL DATA ONLY - empirical integrated dataset
# This dataset combines:
# - Actual rainfall (via OpenMeteo API)
# - Actual temperature (via NASA POWER API)
# - Fertilizer use (via FAOSTAT/World Bank)
# - ENSO index (El Niño/La Niña, via NOAA CPC)
# - Soil properties (via SoilGrids API)
# - Maize yield (via FAOSTAT)

print("=" * 80)
print("MODEL TRAINING ON EMPIRICAL DATA ONLY")
print("=" * 80)

# Find the real data file
real_paths = [
    os.path.join(PROJECT_ROOT, 'data', 'zimbabwe_maize_yield_realvars.csv'),
    os.path.join(PROJECT_ROOT, 'data', 'zimbabwe_maize_yield_realvars_temp.csv'),
    os.path.join(PROJECT_ROOT, 'data', 'zimbabwe_maize_yield_real.csv'),
]

real_path = next((p for p in real_paths if os.path.exists(p)), None)
if real_path is None:
    raise FileNotFoundError(
        f"Real data file not found. Tried: {real_paths}\n"
        f"Please run src/fetch_enso_index.py, src/fetch_soil_data.py, and src/integrate_datasets.py first."
    )

print(f"\nTraining on empirical dataset:\n  {real_path}")
print(f"\nFeatures: Rainfall, Temperature, Fertilizer, ENSO Index, Soil Properties")
print(f"Target: Maize Yield (kg/ha)")
print(f"Training period: Pre-2016 | Test period: 2016+")
print()

real_results, real_models, X_test, y_test = train_and_eval(real_path)
real_best, real_model_path = save_best_model(
    real_models, real_results, 
    os.path.join(models_dir, 'maize_yield_model.pkl')
)

# Save detailed metrics report
metrics_path = save_metrics_report(
    real_results,
    os.path.join(reports_dir, 'model_training_metrics.csv')
)

# Create visualization
graph_dir = os.path.join(reports_dir, 'graphs')
os.makedirs(graph_dir, exist_ok=True)
plot_actual_vs_pred(
    y_test, 
    real_results[real_best]['preds'], 
    os.path.join(graph_dir, 'actual_vs_predicted.png')
)

# Print summary
print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"\nBest Model: {real_best}")
print(f"Saved to: {real_model_path}")
print(f"\nTest Set Performance (2016 onward):")
print(f"  RMSE: {real_results[real_best]['rmse']:.2f} kg/ha")
print(f"  MAE:  {real_results[real_best]['mae']:.2f} kg/ha")
print(f"  R²:   {real_results[real_best]['r2']:.3f}")
print(f"\nCross-Validation Performance (Pre-2016, TimeSeriesSplit):")
print(f"  CV RMSE: {real_results[real_best]['cv_rmse']:.2f} kg/ha")
print(f"  CV MAE:  {real_results[real_best]['cv_mae']:.2f} kg/ha")
print(f"  CV R²:   {real_results[real_best]['cv_r2']:.3f}")

print(f"\nDetailed metrics saved to: {metrics_path}")
print(f"Actual vs Predicted plot: {os.path.join(graph_dir, 'actual_vs_predicted.png')}")
print("\n" + "=" * 80)
