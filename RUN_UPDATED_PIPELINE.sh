#!/usr/bin/env bash
# Quick Reference: Zimbabwe Maize Yield Prediction - Updated Pipeline
# 
# This script runs the complete updated data science pipeline with all improvements:
# ✅ Chronological train/test split (no data leakage)
# ✅ Regularized models (no overfitting)
# ✅ Removed confounding variables
# ✅ Empirical data only (no synthetic)
# ✅ Enhanced features (ENSO, Soil)

set -e  # Exit on error

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"

echo ""
echo "================================================================================================="
echo "ZIMBABWE MAIZE YIELD PREDICTION - COMPLETE PIPELINE"
echo "================================================================================================="
echo ""

# Step 1: Fetch new data sources
echo "[1/5] Fetching ENSO (El Niño/La Niña) index..."
python src/fetch_enso_index.py
echo ""

echo "[2/5] Fetching soil properties from SoilGrids..."
python src/fetch_soil_data.py
echo ""

# Step 2: Integrate all datasets
echo "[3/5] Integrating all datasets (rainfall, temperature, fertilizer, ENSO, soil, yield)..."
python src/integrate_datasets.py
echo ""

# Step 3: Train models with chronological split
echo "[4/5] Training models with chronological split (pre-2016 train, 2016+ test)..."
python src/compare_models.py
echo ""

# Step 4: Run tests
echo "[5/5] Running validation tests (chronological split, no NaN errors, feature checks)..."
python -m pytest tests/test_train_model.py -v --tb=short
echo ""

echo "================================================================================================="
echo "✅ PIPELINE COMPLETE!"
echo "================================================================================================="
echo ""
echo "Output files:"
echo "  📊 Best model:     models/maize_yield_model.pkl"
echo "  📈 Metrics:        reports/model_training_metrics.csv"
echo "  📉 Plot:           reports/graphs/actual_vs_predicted.png"
echo "  🧪 Real data:      data/zimbabwe_maize_yield_realvars.csv"
echo ""
echo "Key improvements:"
echo "  ✅ No data leakage (chronological split: pre-2016 train, 2016+ test)"
echo "  ✅ No overfitting (regularized models: Ridge/Lasso prioritized)"
echo "  ✅ No confounding (Area_Harvested removed)"
echo "  ✅ Empirical data (synthetic data eliminated)"
echo "  ✅ Enhanced features (ENSO Index, Soil properties added)"
echo ""
echo "View detailed methodology in: METHODOLOGY_IMPROVEMENTS.md"
echo ""
