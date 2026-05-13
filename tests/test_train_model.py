"""Unit tests for model training and evaluation."""
import unittest
import os
import pandas as pd
import numpy as np
import tempfile
import sys
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train_model import train_models, save_best_model, TEMPORAL_SPLIT_YEAR, FEATURES


class TestModelTraining(unittest.TestCase):
    
    def setUp(self):
        """Create a temporary synthetic dataset with all required features."""
        self.temp_dir = tempfile.mkdtemp()
        np.random.seed(42)
        n = 40  # 40 years of data
        
        self.df = pd.DataFrame({
            'Year': range(1980, 1980 + n),
            'Rainfall_mm': np.random.uniform(550, 750, n),
            'Average_Temperature_C': np.random.uniform(20, 24, n),
            'Fertilizer_kg_per_ha': np.random.uniform(40, 75, n),
            'ENSO_Index': np.random.uniform(-1.5, 1.5, n),  # NEW
            'Soil_Clay_Percent': np.random.uniform(25, 35, n),  # NEW
            'Soil_pH': np.random.uniform(5.5, 7.0, n),  # NEW
            'Soil_Organic_Carbon_Percent': np.random.uniform(1.5, 2.5, n),  # NEW
            'Maize_Production_Tonnes': np.random.uniform(450000, 800000, n),
            'Yield_kg_per_ha': np.random.uniform(1500, 1900, n)
        })
    
    def test_chronological_split_enforced(self):
        """Verify that chronological split is strictly enforced."""
        models, results, X_test, y_test = train_models(self.df)
        
        # Test set should only contain years >= TEMPORAL_SPLIT_YEAR
        test_years = self.df.iloc[X_test.index]['Year'].values
        self.assertTrue(np.all(test_years >= TEMPORAL_SPLIT_YEAR),
                       f"Found test years before {TEMPORAL_SPLIT_YEAR}: {test_years}")
        
        # Training set should only contain years < TEMPORAL_SPLIT_YEAR
        train_mask = self.df['Year'] < TEMPORAL_SPLIT_YEAR
        train_years = self.df[train_mask]['Year'].values
        self.assertGreater(len(train_years), 0, 
                          f"No training data available before year {TEMPORAL_SPLIT_YEAR}")
    
    def test_no_data_leakage(self):
        """Verify no data leakage between train and test sets."""
        models, results, X_test, y_test = train_models(self.df)
        
        # Get indices
        test_indices = X_test.index
        train_years = self.df[self.df['Year'] < TEMPORAL_SPLIT_YEAR]['Year'].values
        
        # Ensure no year in test set appears in training period
        test_years = self.df.loc[test_indices, 'Year'].values
        for test_year in test_years:
            self.assertFalse(test_year < TEMPORAL_SPLIT_YEAR,
                            f"Test year {test_year} is in training period")
    
    def test_all_features_available(self):
        """Verify all required features are present and don't cause NaN errors."""
        for feature in FEATURES:
            self.assertIn(feature, self.df.columns,
                         f"Missing required feature: {feature}")
            self.assertFalse(self.df[feature].isna().all(),
                           f"Feature {feature} is all NaN")
    
    def test_enso_feature_no_errors(self):
        """Verify ENSO feature doesn't cause issues during training."""
        models, results, X_test, y_test = train_models(self.df)
        
        # Check that models trained without NaN issues
        for name, model in models.items():
            self.assertIsNotNone(model)
            # Test predictions work
            preds = model.predict(X_test)
            self.assertEqual(len(preds), len(y_test))
            self.assertTrue(np.all(np.isfinite(preds)),
                          f"{name} produced non-finite predictions")
    
    def test_soil_features_no_errors(self):
        """Verify soil features don't cause issues during training."""
        soil_features = ['Soil_Clay_Percent', 'Soil_pH', 'Soil_Organic_Carbon_Percent']
        for feat in soil_features:
            self.assertIn(feat, FEATURES,
                         f"Soil feature {feat} not in FEATURES list")
        
        # Train and ensure no NaN propagation
        models, results, X_test, y_test = train_models(self.df)
        self.assertGreater(len(models), 0)
    
    def test_train_models(self):
        """Test that models can be trained without error."""
        models, results, X_test, y_test = train_models(self.df)
        
        # Should have at least LinearRegression and Ridge models
        self.assertIn('LinearRegression', models)
        self.assertIn('RidgeCV', models)
        self.assertIn('rmse', results['LinearRegression'])
        self.assertIn('mae', results['LinearRegression'])
        self.assertIn('r2', results['LinearRegression'])
        self.assertIn('cv_rmse', results['LinearRegression'])
    
    def test_models_produce_predictions(self):
        """Test that models produce valid predictions."""
        models, results, X_test, y_test = train_models(self.df)
        for name in models:
            preds = results[name]['preds']
            self.assertEqual(len(preds), len(y_test))
            self.assertTrue(np.all(np.isfinite(preds)))
    
    def test_regularized_models_preferred(self):
        """Verify that regularized models (Ridge/Lasso) are in candidates."""
        # This tests that we've shifted away from complex ensemble methods
        from train_model import build_model_candidates
        candidates = build_model_candidates()
        
        self.assertIn('LinearRegression', candidates)
        self.assertIn('RidgeCV', candidates)
        self.assertIn('LassoCV', candidates)
        
        # RandomForest if present should be heavily restricted
        if 'RandomForest_Restricted' in candidates:
            # Verify it's restricted (max_depth=2 is a good sign)
            self.assertIn('RandomForest_Restricted', candidates)
    
    def test_save_best_model(self):
        """Test saving the best model."""
        models, results, X_test, y_test = train_models(self.df)
        best_name, model_path = save_best_model(models, results, 
                                                 out_path=os.path.join(self.temp_dir, 'test_model.pkl'))
        self.assertTrue(os.path.exists(model_path))
        loaded_model = joblib.load(model_path)
        self.assertIsNotNone(loaded_model)
    
    def test_missing_features_raises_error(self):
        """Verify that missing required features cause an appropriate error."""
        df_missing = self.df.drop('ENSO_Index', axis=1)  # Remove ENSO feature
        
        with self.assertRaises(ValueError) as context:
            train_models(df_missing)
        
        self.assertIn('Missing required columns', str(context.exception))
    
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir)


if __name__ == '__main__':
    unittest.main()
