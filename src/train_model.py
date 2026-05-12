"""Train and compare maize-yield regression models.

The dataset is a short annual time series, so model selection uses
chronological cross-validation instead of a random split. The final saved
model is refit on all available years after evaluation.
"""
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


FEATURES = ['Rainfall_mm', 'Average_Temperature_C', 'Fertilizer_kg_per_ha', 'Area_Harvested_Ha']
TARGET = 'Yield_kg_per_ha'


def load_data(path):
    return pd.read_csv(path)


def clean_training_data(df, target=TARGET, features=FEATURES):
    """Return numeric, complete training rows sorted by year when available."""
    df = df.copy()
    df.columns = [column.strip() for column in df.columns]

    required = list(features) + [target]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    for column in required:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    df = df.dropna(subset=required)
    if 'Year' in df.columns:
        df = df.sort_values('Year')

    if len(df) < 10:
        raise ValueError('At least 10 complete yearly records are required for training.')

    return df.reset_index(drop=True)


def build_model_candidates():
    """Create conservative candidates suited to a small tabular dataset."""
    linear_steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]

    return {
        'LinearRegression': Pipeline(linear_steps + [('model', LinearRegression())]),
        'RidgePolynomial': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler', StandardScaler()),
            ('model', RidgeCV(alphas=np.logspace(-3, 4, 20))),
        ]),
        'RandomForest': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestRegressor(
                n_estimators=500,
                max_depth=5,
                min_samples_leaf=2,
                random_state=42,
            )),
        ]),
        'ExtraTrees': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', ExtraTreesRegressor(
                n_estimators=500,
                max_depth=5,
                min_samples_leaf=2,
                random_state=42,
            )),
        ]),
        'GradientBoosting': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.03,
                max_depth=2,
                min_samples_leaf=2,
                subsample=0.85,
                random_state=42,
            )),
        ]),
    }


def _regression_metrics(y_true, preds):
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae = mean_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)
    return {'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2)}


def _make_cv(n_rows, chronological):
    if chronological and n_rows >= 20:
        return TimeSeriesSplit(n_splits=5)
    splits = min(5, max(2, n_rows // 5))
    return KFold(n_splits=splits, shuffle=True, random_state=42)


def _cross_validate_model(model, X, y, cv):
    actual_values = []
    predicted_values = []
    for train_index, test_index in cv.split(X):
        fold_model = clone(model)
        fold_model.fit(X.iloc[train_index], y.iloc[train_index])
        fold_preds = fold_model.predict(X.iloc[test_index])
        actual_values.extend(y.iloc[test_index].to_numpy())
        predicted_values.extend(fold_preds)
    return _regression_metrics(np.asarray(actual_values), np.asarray(predicted_values))


def train_models(df, target=TARGET):
    df = clean_training_data(df, target=target)
    X = df[FEATURES]
    y = df[target]

    chronological = 'Year' in df.columns
    cv = _make_cv(len(df), chronological=chronological)
    holdout_size = max(3, int(round(len(df) * 0.2)))
    train_end = len(df) - holdout_size
    X_train, X_test = X.iloc[:train_end], X.iloc[train_end:]
    y_train, y_test = y.iloc[:train_end], y.iloc[train_end:]

    candidates = build_model_candidates()
    models = {}
    results = {}

    for name, model in candidates.items():
        cv_metrics = _cross_validate_model(model, X, y, cv)

        model.fit(X_train, y_train)
        holdout_preds = model.predict(X_test)
        holdout_metrics = _regression_metrics(y_test, holdout_preds)

        final_model = build_model_candidates()[name]
        final_model.fit(X, y)
        models[name] = final_model
        results[name] = {
            'rmse': holdout_metrics['rmse'],
            'mae': holdout_metrics['mae'],
            'r2': holdout_metrics['r2'],
            'cv_rmse': cv_metrics['rmse'],
            'cv_mae': cv_metrics['mae'],
            'cv_r2': cv_metrics['r2'],
            'preds': holdout_preds,
        }

    return models, results, X_test, y_test


def save_best_model(models, results, out_path=os.path.join('..', 'models', 'maize_yield_model.pkl')):
    # Choose by cross-validation RMSE when available, with holdout RMSE as fallback.
    best = min(results.items(), key=lambda kv: kv[1].get('cv_rmse', kv[1]['rmse']))[0]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(models[best], out_path)
    return best, out_path


def save_metrics_report(results, out_file):
    rows = []
    for model_name, metrics in results.items():
        rows.append({
            'model': model_name,
            'holdout_rmse': metrics['rmse'],
            'holdout_mae': metrics['mae'],
            'holdout_r2': metrics['r2'],
            'cv_rmse': metrics.get('cv_rmse'),
            'cv_mae': metrics.get('cv_mae'),
            'cv_r2': metrics.get('cv_r2'),
        })
    report = pd.DataFrame(rows).sort_values('cv_rmse')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    report.to_csv(out_file, index=False)
    return out_file


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
    
    candidates = [
        os.path.join(project_root, 'data', 'zimbabwe_maize_yield_realvars_temp.csv'),
        os.path.join(project_root, 'data', 'zimbabwe_maize_yield_realvars.csv'),
        os.path.join(project_root, 'data', 'processed', 'zimbabwe_maize_yield_processed.csv'),
        os.path.join(project_root, 'data', 'zimbabwe_maize_yield.csv'),
    ]
    data_path = next((path for path in candidates if os.path.exists(path)), None)
    if data_path is None:
        raise FileNotFoundError('No training dataset found in the data directory.')
    
    df = load_data(data_path)
    models, results, X_test, y_test = train_models(df)
    model_out = os.path.join(project_root, 'models', 'maize_yield_model.pkl')
    best_name, model_path = save_best_model(models, results, out_path=model_out)
    metrics_path = save_metrics_report(results, os.path.join(project_root, 'reports', 'model_training_metrics.csv'))
    print(f'Trained on: {data_path}')
    print('Training results:')
    for k, v in results.items():
        print(
            f"{k}: "
            f"CV_RMSE={v['cv_rmse']:.2f}, CV_MAE={v['cv_mae']:.2f}, CV_R2={v['cv_r2']:.3f}; "
            f"Holdout_RMSE={v['rmse']:.2f}, Holdout_MAE={v['mae']:.2f}, Holdout_R2={v['r2']:.3f}"
        )
    print(f"Best model: {best_name} saved to {model_path}")
    print(f"Metrics report saved to {metrics_path}")
    # save plot
    graph_dir = os.path.join(project_root, 'reports', 'graphs')
    os.makedirs(graph_dir, exist_ok=True)
    plot_actual_vs_pred(y_test, results[best_name]['preds'], os.path.join(graph_dir, 'actual_vs_predicted.png'))
