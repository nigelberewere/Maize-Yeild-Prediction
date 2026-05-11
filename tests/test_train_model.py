"""Unit tests for model training and evaluation."""
import unittest
import os
import pandas as pd
import numpy as np
import tempfile
import sys
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train_model import train_models, save_best_model


class TestModelTraining(unittest.TestCase):
    
    def setUp(self):
        """Create a temporary synthetic dataset."""
        self.temp_dir = tempfile.mkdtemp()
        np.random.seed(42)
        n = 30
        self.df = pd.DataFrame({
            'Year': range(1990, 1990 + n),
            'Rainfall_mm': np.random.uniform(550, 750, n),
            'Average_Temperature_C': np.random.uniform(20, 24, n),
            'Fertilizer_kg_per_ha': np.random.uniform(40, 75, n),
            'Area_Harvested_Ha': np.random.uniform(300000, 420000, n),
            'Maize_Production_Tonnes': np.random.uniform(450000, 800000, n),
            'Yield_kg_per_ha': np.random.uniform(1500, 1900, n)
        })
    
    def test_train_models(self):
        """Test that models can be trained without error."""
        models, results, X_test, y_test = train_models(self.df)
        self.assertIn('LinearRegression', models)
        self.assertIn('RandomForest', models)
        self.assertIn('rmse', results['LinearRegression'])
        self.assertIn('mae', results['LinearRegression'])
        self.assertIn('r2', results['LinearRegression'])
    
    def test_models_produce_predictions(self):
        """Test that models produce valid predictions."""
        models, results, X_test, y_test = train_models(self.df)
        for name in models:
            preds = results[name]['preds']
            self.assertEqual(len(preds), len(y_test))
            self.assertTrue(np.all(np.isfinite(preds)))
    
    def test_save_best_model(self):
        """Test saving the best model."""
        models, results, X_test, y_test = train_models(self.df)
        best_name, model_path = save_best_model(models, results, 
                                                 out_path=os.path.join(self.temp_dir, 'test_model.pkl'))
        self.assertTrue(os.path.exists(model_path))
        loaded_model = joblib.load(model_path)
        self.assertIsNotNone(loaded_model)
    
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir)


if __name__ == '__main__':
    unittest.main()
