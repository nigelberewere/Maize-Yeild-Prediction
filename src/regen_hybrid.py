"""Regenerate hybrid dataset (real rainfall + synthetic variables) and save as
`data/zimbabwe_maize_yield_hybrid.csv` so we can compare against real-vars dataset.
"""
import os
from create_hybrid_dataset import create_hybrid_dataset

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rainfall_path = os.path.join(project_root, 'Datasets', 'zwe-rainfall-subnat-full.csv')
output_path = os.path.join(project_root, 'data', 'zimbabwe_maize_yield_hybrid.csv')

print('Regenerating hybrid dataset...')
create_hybrid_dataset(rainfall_path, output_path)
print('Hybrid regenerated to', output_path)
