"""Fetch annual Zimbabwe temperature data from NASA POWER monthly T2M.

The script queries several Zimbabwe reference points, aggregates monthly
T2M values into annual means per point, then averages the points into a
country-level annual series.
"""
import os
import requests
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'zimbabwe_temperature_nasa_power.csv')

START_YEAR = 1981
END_YEAR = 2025

# Representative Zimbabwe reference points used to approximate a country average.
TEMP_POINTS = [
    {'name': 'Harare', 'latitude': -17.8252, 'longitude': 31.0335},
    {'name': 'Bulawayo', 'latitude': -20.1325, 'longitude': 28.6265},
    {'name': 'Mutare', 'latitude': -18.9707, 'longitude': 32.6709},
    {'name': 'Victoria Falls', 'latitude': -17.9243, 'longitude': 25.8567},
    {'name': 'Masvingo', 'latitude': -20.0636, 'longitude': 30.8270},
]


def _fetch_point_temperature(latitude: float, longitude: float) -> pd.DataFrame:
    url = (
        'https://power.larc.nasa.gov/api/temporal/monthly/point'
        f'?parameters=T2M&community=AG&longitude={longitude}&latitude={latitude}'
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
    return df.groupby('Year', as_index=False)['Average_Temperature_C'].mean()


def fetch_monthly_temperature() -> pd.DataFrame:
    frames = []
    for point in TEMP_POINTS:
        point_df = _fetch_point_temperature(point['latitude'], point['longitude'])
        point_df = point_df.rename(columns={'Average_Temperature_C': point['name']})
        frames.append(point_df)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on='Year', how='inner')

    temp_columns = [col for col in merged.columns if col != 'Year']
    merged['Average_Temperature_C'] = merged[temp_columns].mean(axis=1)
    merged = merged[['Year', 'Average_Temperature_C']]
    return merged.sort_values('Year').reset_index(drop=True)


if __name__ == '__main__':
    df = fetch_monthly_temperature()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f'Saved temperature data to {OUTPUT_PATH}')
    print(df.head())
    print(df.tail())
