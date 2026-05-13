"""
Data Integration Script: Merge Zimbabwe Agriculture, Rainfall, Fertilizer, ENSO, and Soil Data
Combines FAOSTAT, rainfall, fertilizer, El Niño/La Niña, and soil quality datasets 
into a single training dataset with enhanced features.
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


def load_enso_data(filepath):
    """Load ENSO (El Niño Southern Oscillation) index data."""
    if not os.path.exists(filepath):
        print(f"   Warning: ENSO file not found at {filepath}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(filepath)
        if 'Year' in df.columns and 'ENSO_Index' in df.columns:
            print(f"   ✓ Loaded ENSO data: {len(df)} years")
            return df[['Year', 'ENSO_Index']].copy()
    except Exception as e:
        print(f"   Warning: Could not load ENSO data: {e}")
    
    return pd.DataFrame()


def load_soil_data(filepath):
    """Load soil properties data (clay, sand, pH, organic carbon)."""
    if not os.path.exists(filepath):
        print(f"   Warning: Soil file not found at {filepath}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(filepath)
        if 'Year' in df.columns:
            # Select soil columns that match pattern
            soil_cols = ['Year'] + [col for col in df.columns if col.startswith('Soil_')]
            print(f"   ✓ Loaded soil data: {len(df)} years with columns {soil_cols[1:]}")
            return df[soil_cols].copy()
    except Exception as e:
        print(f"   Warning: Could not load soil data: {e}")
    
    return pd.DataFrame()


def load_real_base_dataset(filepath):
    """Load the empirical base dataset if it already exists."""
    if not os.path.exists(filepath):
        return pd.DataFrame()

    try:
        df = pd.read_csv(filepath)
    except Exception as exc:
        print(f"   Warning: Could not load base dataset: {exc}")
        return pd.DataFrame()

    if 'Year' not in df.columns:
        return pd.DataFrame()

    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Year']).copy()
    df['Year'] = df['Year'].astype(int)
    return df

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

def merge_datasets(rainfall_df, fertilizer_df, production_df, area_df, yield_df, 
                  enso_df=None, soil_df=None, output_path=None, base_df=None):
    """Merge all datasets into a unified training dataset.
    
    Args:
        rainfall_df: Rainfall data by year
        fertilizer_df: Fertilizer use by year
        production_df: Maize production by year
        area_df: Area harvested by year
        yield_df: Maize yield by year
        enso_df: ENSO index (El Niño/La Niña) by year
        soil_df: Soil properties by year
        output_path: Path to save merged CSV
    
    Returns:
        Merged DataFrame
    """
    
    if base_df is not None and not base_df.empty:
        merged = base_df.copy()
        if 'year' in merged.columns:
            merged.rename(columns={'year': 'Year'}, inplace=True)
        merged['Year'] = pd.to_numeric(merged['Year'], errors='coerce')
        merged = merged.dropna(subset=['Year']).copy()
        merged['Year'] = merged['Year'].astype(int)
    else:
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
    
    # Merge ENSO data (NEW)
    if enso_df is not None and not enso_df.empty:
        enso_df['Year'] = enso_df['Year'].astype(int)
        merged = merged.merge(enso_df, on='Year', how='left')
        print(f"   ✓ Merged ENSO data")
    
    # Merge Soil data (NEW)
    if soil_df is not None and not soil_df.empty:
        soil_df['Year'] = soil_df['Year'].astype(int)
        merged = merged.merge(soil_df, on='Year', how='left')
        print(f"   ✓ Merged soil properties data")
    
    # Keep only essential columns for training
    # (Remove Area_Harvested_Ha as it's a confounding variable for yield prediction)
    essential_cols = [
        'Year', 
        'Rainfall_mm', 
        'Average_Temperature_C', 
        'Fertilizer_kg_per_ha',
        'Maize_Production_Tonnes',
        'Yield_kg_per_ha',
        'ENSO_Index',  # NEW
        'Soil_Clay_Percent',  # NEW
        'Soil_Sand_Percent',  # NEW
        'Soil_pH',  # NEW
        'Soil_Organic_Carbon_Percent',  # NEW
    ]
    available_cols = [c for c in essential_cols if c in merged.columns]
    merged = merged[available_cols]
    
    # Fill missing values with interpolation
    for col in merged.columns:
        if col != 'Year':
            merged[col] = pd.to_numeric(merged[col], errors='coerce')
            merged[col] = merged[col].interpolate(method='linear', limit_direction='both')
    
    # Drop rows with all NaN values (except Year)
    merged = merged.dropna(subset=['Yield_kg_per_ha'], how='all')
    if base_df is not None and not base_df.empty:
        required_existing = ['Rainfall_mm', 'Average_Temperature_C', 'Fertilizer_kg_per_ha', 'Yield_kg_per_ha']
        existing_required = [column for column in required_existing if column in merged.columns]
        if existing_required:
            merged = merged.dropna(subset=existing_required, how='any')
    
    # Save merged dataset if output_path provided
    if output_path:
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
    base_path = os.path.join(project_root, 'data', 'zimbabwe_maize_yield_realvars_fixed.csv')
    base_df = load_real_base_dataset(base_path)
    
    print("=" * 80)
    print("ZIMBABWE MAIZE YIELD DATA INTEGRATION")
    print("=" * 80)
    
    if not base_df.empty:
        print("\n1. Loading empirical base dataset...")
        print(f"   ✓ Base data: {base_df.shape}")
        rainfall_df = pd.DataFrame()
        fertilizer_df = pd.DataFrame()
        production_df = pd.DataFrame()
        area_df = pd.DataFrame()
        yield_df = pd.DataFrame()
    else:
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
    
    # Load ENSO data (NEW)
    print("\n4. Loading ENSO (El Niño/La Niña) index data...")
    enso_path = os.path.join(datasets_dir, 'enso_index.csv')
    enso_df = load_enso_data(enso_path)
    
    # Load Soil data (NEW)
    print("\n5. Loading soil properties data...")
    soil_path = os.path.join(datasets_dir, 'soil_properties.csv')
    soil_df = load_soil_data(soil_path)
    
    # Merge all datasets
    print("\n6. Merging all datasets...")
    output_path = os.path.join(project_root, 'data', 'zimbabwe_maize_yield_realvars.csv')
    merged_df = merge_datasets(
        rainfall_df, fertilizer_df, production_df, area_df, yield_df,
        enso_df=enso_df, soil_df=soil_df, output_path=output_path, base_df=base_df
    )
    
    print("\n" + "=" * 80)
    print("✓ Data integration complete!")
    print("=" * 80)
