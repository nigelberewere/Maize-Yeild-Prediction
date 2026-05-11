"""
Create a hybrid real+synthetic dataset for Zimbabwe maize modeling.
Uses real rainfall data and estimates other variables based on agricultural patterns.
"""

import os
import pandas as pd
import numpy as np

def create_hybrid_dataset(rainfall_csv, output_path):
    """Create hybrid dataset from real rainfall data."""
    
    # Load real rainfall data
    rainfall_df = pd.read_csv(rainfall_csv)
    rainfall_df.columns = ['date', 'adm_level', 'adm_id', 'PCODE', 'n_pixels', 'rfh', 'rfh_avg', 'r1h', 'r1h_avg', 'r3h', 'r3h_avg', 'rfq', 'r1q', 'r3q', 'version']
    
    rainfall_df['date'] = pd.to_datetime(rainfall_df['date'])
    rainfall_df['year'] = rainfall_df['date'].dt.year
    
    # Aggregate rainfall by year
    yearly_rainfall = rainfall_df.groupby('year')['rfh_avg'].mean().reset_index()
    yearly_rainfall.columns = ['Year', 'Rainfall_mm']
    
    print(f"Real rainfall data: {len(yearly_rainfall)} years (1981-{yearly_rainfall['Year'].max()})")
    print(f"Rainfall range: {yearly_rainfall['Rainfall_mm'].min():.1f} - {yearly_rainfall['Rainfall_mm'].max():.1f} mm")
    
    # Ensure rainfall has realistic range (Zimbabwe: 350-850 mm annually)
    # The aggregated data seems to be daily/period data, so convert appropriately
    yearly_rainfall['Rainfall_mm'] = (yearly_rainfall['Rainfall_mm'] * 365 / 10).clip(350, 850)
    
    # Estimate temperature (°C) - Zimbabwe avg ~ 20-23°C, varies slightly
    yearly_rainfall['Average_Temperature_C'] = 22 + np.random.normal(0, 0.5, len(yearly_rainfall))
    yearly_rainfall['Average_Temperature_C'] = yearly_rainfall['Average_Temperature_C'].clip(20, 24)
    
    # Estimate fertilizer use (kg/ha) - generally increased over time in Zimbabwe, but also varies by rainfall
    base_fert = 45 + (yearly_rainfall['Year'] - 1981) * 0.3
    fert_rainfall_adj = (yearly_rainfall['Rainfall_mm'] - yearly_rainfall['Rainfall_mm'].min()) / (yearly_rainfall['Rainfall_mm'].max() - yearly_rainfall['Rainfall_mm'].min()) * 30
    yearly_rainfall['Fertilizer_kg_per_ha'] = (base_fert + fert_rainfall_adj).clip(30, 120)
    
    # Estimate area harvested (hectares) - typically 300k-450k for Zimbabwe
    yearly_rainfall['Area_Harvested_Ha'] = 350000 + (yearly_rainfall['Year'] - 1981) * 500 + np.random.normal(0, 20000, len(yearly_rainfall))
    yearly_rainfall['Area_Harvested_Ha'] = yearly_rainfall['Area_Harvested_Ha'].clip(250000, 480000)
    
    # Estimate yield based on RAINFALL (primary driver for Zimbabwe maize)
    # Use strong correlation: Rainfall is the most important factor
    rain_normalized = (yearly_rainfall['Rainfall_mm'] - yearly_rainfall['Rainfall_mm'].min()) / (yearly_rainfall['Rainfall_mm'].max() - yearly_rainfall['Rainfall_mm'].min())
    temp_effect = ((yearly_rainfall['Average_Temperature_C'] - 20) / 4) * 100  # optimal at 21-23°C
    fert_effect = (yearly_rainfall['Fertilizer_kg_per_ha'] - 30) / (120 - 30) * 200
    
    base_yield = 1200
    rainfall_yield = rain_normalized * 600  # Rain is biggest factor
    yearly_rainfall['Yield_kg_per_ha'] = base_yield + rainfall_yield + temp_effect + fert_effect + np.random.normal(0, 40, len(yearly_rainfall))
    yearly_rainfall['Yield_kg_per_ha'] = yearly_rainfall['Yield_kg_per_ha'].clip(800, 2000)
    
    # Production = Yield * Area / 1000 (convert kg to tonnes)
    yearly_rainfall['Maize_Production_Tonnes'] = (yearly_rainfall['Yield_kg_per_ha'] * yearly_rainfall['Area_Harvested_Ha']) / 1000
    
    # Reorder columns
    final_cols = ['Year', 'Rainfall_mm', 'Average_Temperature_C', 'Fertilizer_kg_per_ha', 'Area_Harvested_Ha', 'Maize_Production_Tonnes', 'Yield_kg_per_ha']
    result = yearly_rainfall[final_cols].sort_values('Year').reset_index(drop=True)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)
    
    print(f"\n✓ Hybrid dataset created: {output_path}")
    print(f"Shape: {result.shape}")
    print(f"Years: {result['Year'].min()} - {result['Year'].max()}")
    print(f"\nSummary Statistics:")
    print(result.describe())
    print(f"\nFirst 5 rows:")
    print(result.head())
    
    return result

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rainfall_path = os.path.join(project_root, 'Datasets', 'zwe-rainfall-subnat-full.csv')
    output_path = os.path.join(project_root, 'data', 'zimbabwe_maize_yield.csv')
    
    print("=" * 80)
    print("CREATING HYBRID REAL+SYNTHETIC ZIMBABWE MAIZE DATASET")
    print("Real rainfall data + estimated agricultural variables")
    print("=" * 80 + "\n")
    
    df = create_hybrid_dataset(rainfall_path, output_path)
    
    print("\n" + "=" * 80)
    print("✓ Dataset ready for model training!")
    print("=" * 80)
