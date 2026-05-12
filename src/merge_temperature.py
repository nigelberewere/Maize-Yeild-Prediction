"""Merge NASA POWER temperature into the real-variables Zimbabwe dataset."""
import os
import pandas as pd
from fetch_temperature import fetch_monthly_temperature

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REALVARS_PATH = os.path.join(PROJECT_ROOT, 'data', 'zimbabwe_maize_yield_realvars.csv')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'zimbabwe_maize_yield_realvars_temp.csv')


def main() -> None:
    df = pd.read_csv(REALVARS_PATH)
    temp_df = fetch_monthly_temperature()
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
