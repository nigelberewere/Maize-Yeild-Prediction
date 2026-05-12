"""Replace the main hybrid CSV with the merged real-variables CSV.
This script reads `data/zimbabwe_maize_yield_realvars.csv` and writes it to
`data/zimbabwe_maize_yield.csv` safely.
"""
import os
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
real = os.path.join(project_root, 'data', 'zimbabwe_maize_yield_realvars.csv')
dest = os.path.join(project_root, 'data', 'zimbabwe_maize_yield.csv')

df = pd.read_csv(real)
# write atomically
tmp = dest + '.tmp'
df.to_csv(tmp, index=False)
os.replace(tmp, dest)
print(f"Replaced {dest} with {real} (rows={len(df)})")
