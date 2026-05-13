"""Fetch historical ENSO (El Niño Southern Oscillation) Index data.

El Niño and La Niña patterns strictly dictate Zimbabwe's drought cycles.
This script downloads the Oceanic Niño Index (ONI) from NOAA, which measures
sea surface temperature anomalies in the equatorial Pacific and is the official
indicator of El Niño and La Niña episodes.

Data source: NOAA Climate Prediction Center (CPC)
https://origin.cpc.ncei.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt
"""

import os
import pandas as pd
import numpy as np
import requests
from io import StringIO


def fetch_enso_data(output_path):
    """Fetch ENSO (Oceanic Niño Index) data from NOAA CPC.
    
    Returns annual average ENSO indices for years 1950-present.
    Positive values indicate El Niño (warm), negative values indicate La Niña (cool).
    """
    url = "https://origin.cpc.ncei.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt"
    
    try:
        print(f"Fetching ENSO data from NOAA CPC...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Parse the NOAA text file
        lines = response.text.split('\n')
        
        # Skip header comments and find data start
        data_lines = []
        for line in lines:
            if line.strip() and not line.startswith('Year'):
                data_lines.append(line)
        
        # Parse year-month-value format
        data_dict = {'Year': [], 'ENSO_Index': []}
        
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    year = int(parts[0])
                    # Get first anomaly value (could be monthly or seasonal)
                    anomaly = float(parts[1])
                    data_dict['Year'].append(year)
                    data_dict['ENSO_Index'].append(anomaly)
                except (ValueError, IndexError):
                    continue
        
        if not data_dict['Year']:
            print("Warning: Could not parse ENSO data. Using alternative method...")
            return fetch_enso_data_alternative(output_path)
        
        # Convert to DataFrame and aggregate to annual
        df = pd.DataFrame(data_dict)
        df = df.groupby('Year')['ENSO_Index'].mean().reset_index()
        
        # Filter for Zimbabwe's maize growing period (1980-2025)
        df = df[(df['Year'] >= 1980) & (df['Year'] <= 2025)]
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"   ✓ ENSO data saved to {output_path}")
        print(f"   Years: {df['Year'].min()}-{df['Year'].max()} (n={len(df)})")
        print(f"   ENSO Index range: [{df['ENSO_Index'].min():.2f}, {df['ENSO_Index'].max():.2f}]")
        print(f"   (Positive=El Niño/drought, Negative=La Niña/wet)")
        
        return df
        
    except Exception as e:
        print(f"Error fetching from NOAA: {e}")
        print("Using fallback synthetic method based on known El Niño years...")
        return fetch_enso_data_alternative(output_path)


def fetch_enso_data_alternative(output_path):
    """Fallback: Create ENSO index based on known El Niño/La Niña episodes.
    
    This is less accurate than real data but provides continuity if the API fails.
    Based on historical records from NOAA CPC.
    """
    # Known El Niño years (1982-83, 1987-88, 1991-92, 1997-98, 2002-03, 2009-10, 2015-16, 2023-24)
    # Known La Niña years (1988-89, 1995-96, 1999-00, 2010-11, 2020-21)
    el_nino_years = {
        1982: 1.5, 1983: 1.8, 1987: 1.2, 1988: 0.5,
        1991: 1.0, 1992: 1.3, 1997: 2.0, 1998: 1.5,
        2002: 1.0, 2003: 0.8, 2009: 0.7, 2010: 0.3,
        2015: 1.8, 2016: 1.5, 2023: 1.2, 2024: 0.8
    }
    
    la_nina_years = {
        1988: -0.5, 1989: -1.0, 1995: -0.8, 1996: -0.5,
        1999: -0.7, 2000: -1.2, 2010: -0.3, 2011: -1.0,
        2020: -0.7, 2021: -1.1
    }
    
    years = np.arange(1980, 2026)
    enso_index = np.zeros(len(years))
    
    for i, year in enumerate(years):
        if year in el_nino_years:
            enso_index[i] = el_nino_years[year]
        elif year in la_nina_years:
            enso_index[i] = la_nina_years[year]
        else:
            # Neutral year with small random variation
            enso_index[i] = np.random.normal(0, 0.2)
    
    df = pd.DataFrame({
        'Year': years,
        'ENSO_Index': enso_index
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"   ⚠ Using fallback ENSO data (based on known episodes)")
    print(f"   Saved to {output_path}")
    
    return df


if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    enso_output = os.path.join(project_root, 'Datasets', 'enso_index.csv')
    
    print("=" * 80)
    print("FETCH ENSO (El Niño Southern Oscillation) INDEX DATA")
    print("=" * 80)
    fetch_enso_data(enso_output)
    print("=" * 80)
