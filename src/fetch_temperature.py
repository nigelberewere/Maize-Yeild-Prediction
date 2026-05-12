"""Fetch annual Zimbabwe temperature data from NASA POWER monthly T2M.

The script queries a single Zimbabwe reference point, aggregates monthly
T2M values into annual means, and writes a CSV with columns:
Year, Average_Temperature_C
"""
import os
import requests
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'zimbabwe_temperature_nasa_power.csv')

# Zimbabwe reference point near the country centroid; used as a stable climate proxy.
LATITUDE = -19.0154
LONGITUDE = 29.1549
START_YEAR = 1981
END_YEAR = 2025


def fetch_monthly_temperature() -> pd.DataFrame:
    url = (
        'https://power.larc.nasa.gov/api/temporal/monthly/point'
        f'?parameters=T2M&community=AG&longitude={LONGITUDE}&latitude={LATITUDE}'
        f'&start={START_YEAR}&end={END_YEAR}&format=JSON'
    )
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    payload = response.json()
    series = payload['properties']['parameter']['T2M']

    rows = []
    for key, value in series.items():
        year = int(key[:4])
        rows.append({'Year': year, 'Average_Temperature_C': float(value)})

    df = pd.DataFrame(rows)
    df = df.groupby('Year', as_index=False)['Average_Temperature_C'].mean()
    return df.sort_values('Year').reset_index(drop=True)


if __name__ == '__main__':
    df = fetch_monthly_temperature()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f'Saved temperature data to {OUTPUT_PATH}')
    print(df.head())
    print(df.tail())
