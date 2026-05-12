"""Merge actual Zimbabwe temperature into the real-variables dataset.

Prefers the Open-Meteo temperature files from `Datasets/` and falls back to
the NASA POWER multi-point series if the Open-Meteo output is unavailable.
"""
import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REALVARS_PATH = os.path.join(PROJECT_ROOT, 'data', 'zimbabwe_maize_yield_realvars.csv')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'zimbabwe_maize_yield_realvars_temp.csv')
OPEN_METEO_MONTHLY_PATH = os.path.join(PROJECT_ROOT, 'Datasets', 'zimbabwe_temperature_monthly_actual_1981_2025.csv')
NASA_POWER_PATH = os.path.join(PROJECT_ROOT, 'data', 'zimbabwe_temperature_nasa_power.csv')


def load_actual_temperature() -> pd.DataFrame:
    if os.path.exists(OPEN_METEO_MONTHLY_PATH):
        monthly = pd.read_csv(OPEN_METEO_MONTHLY_PATH)
        annual = (
            monthly.groupby('year', as_index=False)['mean_temp_c']
            .mean()
            .rename(columns={'year': 'Year', 'mean_temp_c': 'Average_Temperature_C'})
        )
        return annual.sort_values('Year').reset_index(drop=True)

    if os.path.exists(NASA_POWER_PATH):
        return pd.read_csv(NASA_POWER_PATH)

    raise FileNotFoundError(
        'No temperature source found. Expected Open-Meteo output in Datasets/ '
        'or NASA POWER output in data/.'
    )


def main() -> None:
    df = pd.read_csv(REALVARS_PATH)
    temp_df = load_actual_temperature()
    merged = df.drop(columns=['Average_Temperature_C'], errors='ignore').merge(temp_df, on='Year', how='left')
    # preserve the original column order
    desired = ['Year', 'Rainfall_mm', 'Average_Temperature_C', 'Fertilizer_kg_per_ha', 'Area_Harvested_Ha', 'Maize_Production_Tonnes', 'Yield_kg_per_ha']
    merged = merged[desired]
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f'Saved merged dataset to {OUTPUT_PATH}')
    print(f'Rows: {len(merged)}')
    print(merged.head())


if __name__ == '__main__':
    main()
