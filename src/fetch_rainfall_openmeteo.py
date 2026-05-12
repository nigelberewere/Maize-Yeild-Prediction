"""Fetch annual Zimbabwe rainfall from Open-Meteo historical API.

Builds a country-level annual rainfall series by averaging monthly precipitation
across multiple Zimbabwe locations, then summing months by year.
"""
import os
import time
import requests
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'zimbabwe_rainfall_openmeteo_annual.csv')

LOCATIONS = [
    {"name": "Harare", "latitude": -17.8292, "longitude": 31.0522},
    {"name": "Bulawayo", "latitude": -20.1325, "longitude": 28.6265},
    {"name": "Mutare", "latitude": -18.9707, "longitude": 32.6709},
    {"name": "Gweru", "latitude": -19.4514, "longitude": 29.8167},
    {"name": "Masvingo", "latitude": -20.0637, "longitude": 30.8277},
    {"name": "Chiredzi", "latitude": -21.0500, "longitude": 31.6667},
    {"name": "Hwange", "latitude": -18.3645, "longitude": 26.4988},
    {"name": "Victoria Falls", "latitude": -17.9243, "longitude": 25.8572},
    {"name": "Beitbridge", "latitude": -22.2167, "longitude": 30.0000},
    {"name": "Karoi", "latitude": -16.8099, "longitude": 29.6925},
    {"name": "Gokwe", "latitude": -18.2048, "longitude": 28.9349},
    {"name": "Zvishavane", "latitude": -20.3267, "longitude": 30.0665},
]

START_DATE = "1981-01-01"
END_DATE = "2025-12-31"
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_with_retry(params, max_retries=5, initial_delay=10):
    delay = initial_delay
    for attempt in range(max_retries):
        response = requests.get(BASE_URL, params=params, timeout=180)
        if response.status_code != 429:
            response.raise_for_status()
            return response

        retry_after = response.headers.get("Retry-After")
        if retry_after:
            delay = max(delay, int(float(retry_after)))
        print(f"Rate limited. Waiting {delay}s before retry {attempt + 1}/{max_retries}...")
        time.sleep(delay)
        delay = min(delay * 2, 120)

    response.raise_for_status()


def fetch_annual_rainfall() -> pd.DataFrame:
    location_frames = []
    for place in LOCATIONS:
        print(f"Downloading monthly rainfall for {place['name']}...")
        params = {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "start_date": START_DATE,
            "end_date": END_DATE,
            "daily": "precipitation_sum",
            "timezone": "Africa/Harare",
        }

        response = fetch_with_retry(params)
        data = response.json()
        if "daily" not in data:
            raise ValueError(f"No daily data returned for {place['name']}: {data}")

        daily = pd.DataFrame(data["daily"])
        daily["date"] = pd.to_datetime(daily["time"])
        daily["Year"] = daily["date"].dt.year
        annual = daily.groupby("Year", as_index=False)["precipitation_sum"].sum()
        annual = annual.rename(columns={"precipitation_sum": place["name"]})
        location_frames.append(annual)

        time.sleep(3)

    merged = location_frames[0]
    for frame in location_frames[1:]:
        merged = merged.merge(frame, on="Year", how="inner")

    rain_cols = [c for c in merged.columns if c != "Year"]
    merged["Rainfall_mm"] = merged[rain_cols].mean(axis=1)
    out = merged[["Year", "Rainfall_mm"]].sort_values("Year").reset_index(drop=True)
    return out


if __name__ == "__main__":
    df = fetch_annual_rainfall()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved annual rainfall to {OUTPUT_PATH}")
    print(df.head())
    print(df.tail())
