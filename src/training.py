import json
import logging

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def training(
    input_path: str, model_path: str, metrics_path: str, hyperparameters: dict
):
    """
    Trains the Random Forest model and saves artifacts.
    """
    logging.basicConfig(level=logging.INFO)
    try:
        logging.info("--- Starting Training ---")

        df = pd.read_csv(input_path)

        # Split features (X) and target (y) - assuming target is the last column
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Initialize and train model
        model = RandomForestRegressor(
            n_estimators=hyperparameters["n_estimators"],
            max_depth=hyperparameters["max_depth"],
            random_state=hyperparameters["random_state"],
        )
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Save model and metrics
        joblib.dump(model, model_path)

        metrics = {"mse": mse, "r2": r2}
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)

        logging.info(f"Model saved. Metrics: {metrics}")

    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise e
