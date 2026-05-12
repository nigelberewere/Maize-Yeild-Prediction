"""Simple prediction script for maize yield using saved model."""
import argparse
import os

import joblib
import pandas as pd


FEATURES = ['Rainfall_mm', 'Average_Temperature_C', 'Fertilizer_kg_per_ha', 'Area_Harvested_Ha']


def load_model(path=None):
    if path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(project_root, 'models', 'maize_yield_model.pkl')
    return joblib.load(path)


def predict(model, rainfall, temp, fert, area):
    X = pd.DataFrame([{
        'Rainfall_mm': rainfall,
        'Average_Temperature_C': temp,
        'Fertilizer_kg_per_ha': fert,
        'Area_Harvested_Ha': area,
    }], columns=FEATURES)
    pred = model.predict(X)
    return max(0.0, float(pred[0]))


def prompt_float(label, default=None):
    prompt = f"{label}"
    if default is not None:
        prompt += f" [{default}]"
    prompt += ": "

    while True:
        raw_value = input(prompt).strip()
        if not raw_value and default is not None:
            return float(default)
        try:
            return float(raw_value)
        except ValueError:
            print(f"Please enter a numeric value for {label.lower()}.")


def main():
    parser = argparse.ArgumentParser(description='Predict maize yield (kg/ha)')
    parser.add_argument('--rainfall', type=float, help='Annual rainfall in mm')
    parser.add_argument('--temp', type=float, help='Average temperature in C')
    parser.add_argument('--fert', type=float, help='Fertilizer use in kg/ha')
    parser.add_argument('--area', type=float, help='Area harvested in hectares')
    parser.add_argument('--interactive', action='store_true', help='Prompt for values step by step')
    args = parser.parse_args()

    interactive = args.interactive or any(
        value is None for value in (args.rainfall, args.temp, args.fert, args.area)
    )

    if interactive:
        print('Enter the maize yield inputs step by step.')
        rainfall = args.rainfall if args.rainfall is not None else prompt_float('Annual rainfall (mm)', 700)
        temp = args.temp if args.temp is not None else prompt_float('Average temperature (C)', 22)
        fert = args.fert if args.fert is not None else prompt_float('Fertilizer use (kg/ha)', 80)
        area = args.area if args.area is not None else prompt_float('Area harvested (hectares)', 380000)
    else:
        rainfall = args.rainfall
        temp = args.temp
        fert = args.fert
        area = args.area

    model = load_model()
    y = predict(model, rainfall, temp, fert, area)
    print(f"Predicted maize yield: {y:.2f} kg/ha")


if __name__ == '__main__':
    main()
