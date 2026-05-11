"""Simple prediction script for maize yield using saved model."""
import os
import joblib
import argparse
import numpy as np


def load_model(path=os.path.join('..', 'models', 'maize_yield_model.pkl')):
    return joblib.load(path)


def predict(model, rainfall, temp, fert, area):
    X = np.array([[rainfall, temp, fert, area]])
    pred = model.predict(X)
    return float(pred[0])


def main():
    parser = argparse.ArgumentParser(description='Predict maize yield (kg/ha)')
    parser.add_argument('--rainfall', type=float, required=True)
    parser.add_argument('--temp', type=float, required=True)
    parser.add_argument('--fert', type=float, required=True)
    parser.add_argument('--area', type=float, required=True)
    args = parser.parse_args()
    model = load_model()
    y = predict(model, args.rainfall, args.temp, args.fert, args.area)
    print(f"Predicted maize yield: {y:.2f} kg/ha")


if __name__ == '__main__':
    main()
