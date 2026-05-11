"""
Clean and prepare real Zimbabwe maize data for model training.
Extracts Zimbabwe-specific maize yield records.
"""

import os
import pandas as pd
import numpy as np

def clean_real_data(raw_csv_path, output_csv_path):
    """Extract and clean Zimbabwe maize data."""
    print("Loading raw integrated data...")
    df = pd.read_csv(raw_csv_path)
    
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}\n")
    
    # Filter for Zimbabwe records
    if 'Area' in df.columns:
        df_zwe = df[df['Area'].str.contains('Zimbabwe', case=False, na=False)].copy()
        print(f"Zimbabwe records: {len(df_zwe)}")
    else:
        df_zwe = df.copy()
    
    # Filter for Maize Item
    if 'Item' in df.columns:
        df_maize = df_zwe[df_zwe['Item'].str.contains('Maize', case=False, na=False)].copy()
        print(f"Maize records: {len(df_maize)}")
    else:
        df_maize = df_zwe.copy()
    
    # Extract key metrics
    result_data = []
    
    for year in sorted(df_maize['Year'].unique()):
        year_data = df_maize[df_maize['Year'] == year]
        
        # Get values for different indicators
        year_dict = {'Year': int(year)}
        
        # Rainfall (from existing Rainfall_mm column if available)
        if 'Rainfall_mm' in df_maize.columns:
            rainfall_val = year_data['Rainfall_mm'].iloc[0] if len(year_data) > 0 else np.nan
            year_dict['Rainfall_mm'] = float(rainfall_val) if pd.notna(rainfall_val) else np.nan
        
        # Fertilizer
        if 'Fertilizer_kg_per_ha' in df_maize.columns:
            fert_val = year_data['Fertilizer_kg_per_ha'].iloc[0] if len(year_data) > 0 else np.nan
            year_dict['Fertilizer_kg_per_ha'] = float(fert_val) if pd.notna(fert_val) else np.nan
        
        # Extract yield from Indicator column
        for idx, row in year_data.iterrows():
            if pd.notna(row.get('Value', np.nan)) and row['Value'] > 0:
                indicator = str(row.get('Indicator', '')).lower()
                value = float(row['Value'])
                
                if 'production' in indicator and 'yield' not in indicator:
                    year_dict['Maize_Production_Tonnes'] = value
                elif 'area' in indicator and 'harvest' in indicator:
                    year_dict['Area_Harvested_Ha'] = value
                elif 'yield' in indicator:
                    year_dict['Yield_kg_per_ha'] = value
        
        if 'Yield_kg_per_ha' in year_dict:
            result_data.append(year_dict)
    
    result_df = pd.DataFrame(result_data)
    
    if result_df.empty:
        print("No maize yield data found. Using synthetic data instead.")
        return None
    
    # Fill missing values through interpolation
    for col in result_df.columns:
        if col != 'Year':
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
            result_df[col] = result_df[col].interpolate(method='linear', limit_direction='both')
    
    # Ensure we have the required columns
    required_cols = ['Year', 'Rainfall_mm', 'Fertilizer_kg_per_ha', 'Area_Harvested_Ha', 'Yield_kg_per_ha']
    for col in required_cols:
        if col not in result_df.columns:
            result_df[col] = np.nan
    
    # Reorder and keep only required columns
    result_df = result_df[required_cols]
    result_df = result_df.sort_values('Year')
    
    # Save cleaned data
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    result_df.to_csv(output_csv_path, index=False)
    
    print(f"\n✓ Cleaned data saved to {output_csv_path}")
    print(f"Shape: {result_df.shape}")
    print(f"\nData summary:")
    print(result_df.describe())
    print(f"\nFirst rows:")
    print(result_df.head(10))
    
    return result_df

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(project_root, 'data', 'zimbabwe_maize_yield_real.csv')
    clean_path = os.path.join(project_root, 'data', 'zimbabwe_maize_yield_clean.csv')
    
    print("=" * 80)
    print("CLEANING REAL ZIMBABWE MAIZE DATA")
    print("=" * 80 + "\n")
    
    df_clean = clean_real_data(raw_path, clean_path)
    
    if df_clean is None or df_clean.empty:
        print("\n⚠ Using synthetic data for training.")
    else:
        print("\n" + "=" * 80)
        print("✓ Data cleaning complete!")
        print("=" * 80)
