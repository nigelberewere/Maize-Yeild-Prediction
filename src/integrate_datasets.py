"""
Data Integration Script: Merge Zimbabwe Agriculture, Rainfall, and Fertilizer Data
Combines FAOSTAT, rainfall, and fertilizer datasets into a single training dataset.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def load_rainfall_data(filepath):
    """Load and aggregate rainfall data by year."""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    # Aggregate rainfall by year (sum annual rainfall - rainfall is cumulative, not averaged)
    yearly_rainfall = df.groupby('year')['rfh_avg'].sum().reset_index()
    yearly_rainfall.rename(columns={'rfh_avg': 'Rainfall_mm'}, inplace=True)
    return yearly_rainfall

def load_fertilizer_data(filepath):
    """Load and process fertilizer data."""
    df = pd.read_csv(filepath)
    
    # Filter for Zimbabwe
    df_zwe = df[df['Country Name'].str.contains('Zimbabwe', case=False, na=False)]
    
    # Filter for fertilizer use indicator
    df_fert = df_zwe[df_zwe['Indicator Name'].str.contains('Fertilizer', case=False, na=False)]
    
    if df_fert.empty:
        print("Warning: Could not find Zimbabwe fertilizer data")
        return pd.DataFrame()
    
    # Extract Year and Value columns
    result = df_fert[['Year', 'Value']].copy()
    result.columns = ['Year', 'Fertilizer_kg_per_ha']
    result['Year'] = pd.to_numeric(result['Year'], errors='coerce')
    result['Fertilizer_kg_per_ha'] = pd.to_numeric(result['Fertilizer_kg_per_ha'], errors='coerce')
    result = result.dropna()
    
    # Aggregate by year if multiple records
    result = result.groupby('Year')['Fertilizer_kg_per_ha'].mean().reset_index()
    
    print(f"   Found {len(result)} years of fertilizer data")
    return result

def load_faostat_maize_data(faostat_dir):
    """Load FAOSTAT maize data (yield, production, area harvested)."""
    csv_files = list(Path(faostat_dir).glob('*.csv'))
    print(f"   Found {len(csv_files)} FAOSTAT CSV files, searching for maize data...")
    
    all_data = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, encoding='latin-1')
            
            # Look for Zimbabwe and maize data
            if 'Area' in df.columns and 'Item' in df.columns and 'Indicator' in df.columns:
                # Filter for Zimbabwe
                df_zwe = df[df['Area'].str.contains('Zimbabwe', case=False, na=False)]
                
                # Filter for Maize
                df_maize = df_zwe[df_zwe['Item'].str.contains('Maize', case=False, na=False)]
                
                if not df_maize.empty:
                    print(f"     ✓ Found maize data in {csv_file.name}")
                    all_data.append(df_maize)
        except Exception as e:
            pass
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        
        # Extract relevant metrics: Production, Area Harvested, Yield
        production = result[result['Indicator'].str.contains('Production', case=False, na=False)][['Year', 'Value']]
        area = result[result['Indicator'].str.contains('Area Harvested', case=False, na=False)][['Year', 'Value']]
        yield_data = result[result['Indicator'].str.contains('Yield', case=False, na=False)][['Year', 'Value']]
        
        # Rename columns
        if not production.empty:
            production.columns = ['Year', 'Maize_Production_Tonnes']
        if not area.empty:
            area.columns = ['Year', 'Area_Harvested_Ha']
        if not yield_data.empty:
            yield_data.columns = ['Year', 'Yield_kg_per_ha']
        
        return production, area, yield_data
    
    print("   Warning: Could not extract specific FAOSTAT maize data")
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def create_synthetic_extension(real_data, start_year=1990, end_year=2024):
    """
    If real data is limited, extend it synthetically based on trends.
    This ensures we have continuous annual data for model training.
    """
    years = np.arange(start_year, end_year + 1)
    
    # If we have real data, fit a trend and extrapolate
    if not real_data.empty and 'Year' in real_data.columns:
        min_year = real_data['Year'].min()
        max_year = real_data['Year'].max()
        years = np.arange(min_year, max_year + 1)
    
    return years

def merge_datasets(rainfall_df, fertilizer_df, production_df, area_df, yield_df, output_path):
    """Merge all datasets into a unified training dataset."""
    
    # Start with rainfall as base
    merged = rainfall_df.copy()
    if 'year' in merged.columns:
        merged.rename(columns={'year': 'Year'}, inplace=True)
    
    # Ensure Year column is integer
    merged['Year'] = merged['Year'].astype(int)
    
    # Merge fertilizer data
    if not fertilizer_df.empty:
        merged = merged.merge(fertilizer_df, on='Year', how='left')
    else:
        merged['Fertilizer_kg_per_ha'] = np.nan
    
    # Merge production data
    if not production_df.empty:
        production_df['Year'] = production_df['Year'].astype(int)
        production_df = production_df.groupby('Year')['Maize_Production_Tonnes'].mean().reset_index()
        merged = merged.merge(production_df, on='Year', how='left')
    else:
        merged['Maize_Production_Tonnes'] = np.nan
    
    # Merge area data
    if not area_df.empty:
        area_df['Year'] = area_df['Year'].astype(int)
        area_df = area_df.groupby('Year')['Area_Harvested_Ha'].mean().reset_index()
        merged = merged.merge(area_df, on='Year', how='left')
    else:
        merged['Area_Harvested_Ha'] = np.nan
    
    # Merge yield data
    if not yield_df.empty:
        yield_df['Year'] = yield_df['Year'].astype(int)
        yield_df = yield_df.groupby('Year')['Yield_kg_per_ha'].mean().reset_index()
        merged = merged.merge(yield_df, on='Year', how='left')
    else:
        merged['Yield_kg_per_ha'] = np.nan
    
    # Keep only essential columns
    essential_cols = ['Year', 'Rainfall_mm', 'Average_Temperature_C', 'Fertilizer_kg_per_ha', 
                     'Area_Harvested_Ha', 'Maize_Production_Tonnes', 'Yield_kg_per_ha']
    available_cols = [c for c in essential_cols if c in merged.columns]
    merged = merged[available_cols]
    
    # Fill missing values with interpolation
    for col in merged.columns:
        if col != 'Year':
            merged[col] = pd.to_numeric(merged[col], errors='coerce')
            merged[col] = merged[col].interpolate(method='linear', limit_direction='both')
    
    # Drop rows with all NaN values (except Year)
    merged = merged.dropna(subset=['Yield_kg_per_ha'], how='all')
    
    # Save merged dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged = merged.sort_values('Year')
    merged.to_csv(output_path, index=False)
    
    print(f"   ✓ Merged dataset saved to {output_path}")
    print(f"   Shape: {merged.shape}")
    print(f"   Columns: {merged.columns.tolist()}")
    print(f"\n   Summary statistics:")
    print(merged.describe())
    
    return merged

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_dir = os.path.join(project_root, 'Datasets')
    
    print("=" * 80)
    print("ZIMBABWE MAIZE YIELD DATA INTEGRATION")
    print("=" * 80)
    
    # Load rainfall data
    print("\n1. Loading rainfall data...")
    rainfall_path = os.path.join(datasets_dir, 'zwe-rainfall-subnat-full.csv')
    rainfall_df = load_rainfall_data(rainfall_path)
    print(f"   ✓ Rainfall data: {rainfall_df.shape}")
    
    # Load fertilizer data
    print("\n2. Loading fertilizer data...")
    fert_path = os.path.join(datasets_dir, 'agriculture-and-rural-development_zwe.csv')
    fertilizer_df = load_fertilizer_data(fert_path)
    print(f"   ✓ Fertilizer data: {fertilizer_df.shape}")
    
    # Load FAOSTAT data
    print("\n3. Loading FAOSTAT maize production data...")
    faostat_dir = os.path.join(datasets_dir, 'FAOSTAT_T-Z_E')
    production_df, area_df, yield_df = load_faostat_maize_data(faostat_dir)
    print(f"   ✓ Production data: {production_df.shape}")
    print(f"   ✓ Area data: {area_df.shape}")
    print(f"   ✓ Yield data: {yield_df.shape}")
    
    # Merge all datasets
    print("\n4. Merging datasets...")
    output_path = os.path.join(project_root, 'data', 'zimbabwe_maize_yield_real.csv')
    merged_df = merge_datasets(rainfall_df, fertilizer_df, production_df, area_df, yield_df, output_path)
    
    print("\n" + "=" * 80)
    print("✓ Data integration complete!")
    print("=" * 80)
