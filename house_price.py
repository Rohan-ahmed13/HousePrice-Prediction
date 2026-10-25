"""Train and use a house-price regression model with Housing.csv."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "price"
NUMERIC_FEATURES = (
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking",
)
CATEGORICAL_FEATURES = (
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus",
)
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate the Housing.csv schema and values."""
    data = pd.read_csv(csv_path)
    required = {*FEATURES, TARGET}
    missing_columns = sorted(required.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")
    if data.empty:
        raise ValueError("Dataset contains no rows.")
    if data[[*FEATURES, TARGET]].isnull().any().any():
        raise ValueError("Dataset contains missing values in required columns.")
    return data


def build_pipeline() -> Pipeline:
    """Create a preprocessing and random-forest regression pipeline."""
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessing),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=400,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_model(
    csv_path: str | Path = "Housing.csv",
    model_path: str | Path = "model/house_price_pipeline.joblib",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, float]:
    """Evaluate a holdout split, then fit and save a pipeline on all rows."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    data = load_dataset(csv_path)
    features = data[list(FEATURES)]
    target = data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )

    evaluation_pipeline = build_pipeline()
    evaluation_pipeline.fit(x_train, y_train)
    predictions = evaluation_pipeline.predict(x_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": math.sqrt(mean_squared_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }

    final_pipeline = build_pipeline()
    final_pipeline.fit(features, target)
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, destination)
    return metrics


def predict_csv(
    model_path: str | Path,
    input_csv: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    """Predict prices for rows in a CSV and write them alongside the inputs."""
    data = pd.read_csv(input_csv)
    missing_columns = sorted(set(FEATURES).difference(data.columns))
    if missing_columns:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing_columns)}")
    if data.empty:
        raise ValueError("Input CSV contains no rows.")
    if data[list(FEATURES)].isnull().any().any():
        raise ValueError("Input CSV contains missing values in required columns.")

    model = joblib.load(model_path)
    result = data.copy()
    result["predicted_price"] = model.predict(data[list(FEATURES)])
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="House-price model training and prediction.")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="Evaluate and train the model.")
    train_parser.add_argument("--data", default="Housing.csv", help="Path to the training CSV.")
    train_parser.add_argument(
        "--model",
        default="model/house_price_pipeline.joblib",
        help="Where to save the trained pipeline.",
    )
    train_parser.add_argument("--test-size", type=float, default=0.2)

    predict_parser = commands.add_parser("predict", help="Predict prices from an input CSV.")
    predict_parser.add_argument(
        "--model",
        default="model/house_price_pipeline.joblib",
        help="Path to a trained pipeline.",
    )
    predict_parser.add_argument("--input", required=True, help="CSV containing house features.")
    predict_parser.add_argument(
        "--output", default="predictions.csv", help="Where to save predictions."
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "train":
        metrics = train_model(args.data, args.model, args.test_size)
        print(f"Saved model to {args.model}")
        print(json.dumps(metrics, indent=2))
    else:
        predictions = predict_csv(args.model, args.input, args.output)
        print(f"Saved {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
