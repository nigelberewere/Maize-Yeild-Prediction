"""Unit tests for data preprocessing module."""
import unittest
import os
import pandas as pd
import tempfile
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import load_data, basic_clean, save_processed


class TestDataPreprocessing(unittest.TestCase):
    
    def setUp(self):
        """Create a temporary test CSV."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_csv = os.path.join(self.temp_dir, 'test_data.csv')
        df = pd.DataFrame({
            'Year': [2020, 2021, 2022],
            'Rainfall_mm': [600, 700, 650],
            'Average_Temperature_C': [22.0, 21.5, 22.5],
            'Fertilizer_kg_per_ha': [50, 60, 55],
            'Area_Harvested_Ha': [400000, 410000, 405000],
            'Maize_Production_Tonnes': [600000, 697000, 648000],
            'Yield_kg_per_ha': [1500, 1700, 1600]
        })
        df.to_csv(self.test_csv, index=False)
    
    def test_load_data(self):
        """Test data loading."""
        df = load_data(self.test_csv)
        self.assertEqual(len(df), 3)
        self.assertIn('Year', df.columns)
    
    def test_basic_clean(self):
        """Test data cleaning."""
        df = load_data(self.test_csv)
        df_clean = basic_clean(df)
        self.assertFalse(df_clean.isnull().any().any())
    
    def test_save_processed(self):
        """Test saving processed data."""
        df = load_data(self.test_csv)
        df_clean = basic_clean(df)
        out_path = save_processed(df_clean, out_dir=self.temp_dir)
        self.assertTrue(os.path.exists(out_path))
        df_loaded = pd.read_csv(out_path)
        self.assertEqual(len(df_loaded), 3)
    
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir)


if __name__ == '__main__':
    unittest.main()
