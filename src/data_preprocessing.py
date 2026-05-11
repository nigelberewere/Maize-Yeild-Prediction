"""Data preprocessing utilities for maize yield prediction."""
import os
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Trim whitespace from column names
    df.columns = [c.strip() for c in df.columns]
    # Convert types
    df = df.dropna(how='all')
    numeric_cols = [
        'Rainfall_mm', 'Average_Temperature_C', 'Fertilizer_kg_per_ha',
        'Area_Harvested_Ha', 'Maize_Production_Tonnes', 'Yield_kg_per_ha'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def save_processed(df: pd.DataFrame, out_dir: str = '../data/processed') -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'zimbabwe_maize_yield_processed.csv')
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == '__main__':
    df = load_data(os.path.join('..', 'data', 'zimbabwe_maize_yield.csv'))
    df = basic_clean(df)
    p = save_processed(df, out_dir=os.path.join('..', 'data', 'processed'))
    print(f"Processed data saved to {p}")
