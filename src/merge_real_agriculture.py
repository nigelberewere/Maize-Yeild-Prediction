"""Merge World Bank agriculture indicators into the hybrid Zimbabwe dataset.

This script reads `Datasets/agriculture-and-rural-development_zwe.csv`, pivots indicators
by year and replaces estimated variables in `data/zimbabwe_maize_yield.csv` where
real data exists. It writes `data/zimbabwe_maize_yield_realvars.csv`.
"""
import os
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datasets_dir = os.path.join(project_root, 'Datasets')
wb_file = os.path.join(datasets_dir, 'agriculture-and-rural-development_zwe.csv')
input_file = os.path.join(project_root, 'data', 'zimbabwe_maize_yield.csv')
output_file = os.path.join(project_root, 'data', 'zimbabwe_maize_yield_realvars.csv')

print("Loading World Bank agriculture dataset...")
df = pd.read_csv(wb_file)
# Keep only relevant columns
df = df[['Year', 'Indicator Name', 'Value']]
# Pivot to have indicators as columns
pivot = df.pivot_table(index='Year', columns='Indicator Name', values='Value', aggfunc='first')

print(f"Pivoted indicators: {len(pivot.columns)} columns, years {pivot.index.min()}-{pivot.index.max()}")

print("Loading hybrid dataset...")
h = pd.read_csv(input_file)
# ensure Year column is int
h['Year'] = h['Year'].astype(int)

# Mapping from World Bank indicators to our dataset columns
mapping = {
    'Fertilizer consumption (kilograms per hectare of arable land)': 'Fertilizer_kg_per_ha',
    'Cereal yield (kg per hectare)': 'Yield_kg_per_ha',
    'Land under cereal production (hectares)': 'Area_Harvested_Ha',
    'Cereal production (metric tons)': 'Maize_Production_Tonnes',
    'Average precipitation in depth (mm per year)': 'Rainfall_mm'
}

# For each year, replace if real value exists
replaced_counts = {v:0 for v in mapping.values()}
for idx, row in h.iterrows():
    yr = int(row['Year'])
    if yr in pivot.index:
        for wb_col, target_col in mapping.items():
            if wb_col in pivot.columns:
                val = pivot.at[yr, wb_col]
                if pd.notna(val):
                    # Some WB indicators (e.g., precipitation) may be aggregated differently,
                    # for fertilizer and yield we assume units match our columns or are close enough.
                    # Apply simple conversions if necessary (none applied here).
                    h.at[idx, target_col] = val
                    replaced_counts[target_col] += 1

print('\nReplaced counts per column:')
for k,v in replaced_counts.items():
    print(f'  - {k}: {v}')

# Save
os.makedirs(os.path.dirname(output_file), exist_ok=True)
h.to_csv(output_file, index=False)
print(f"\nSaved merged dataset to: {output_file}")
print(f"Shape: {h.shape}")
