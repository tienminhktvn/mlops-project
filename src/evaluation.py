import logging

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def evaluation(model_path: str, data_path: str, plot_path: str):
    """
    Generates evaluation plots (Actual vs. Predicted).
    """
    logging.basicConfig(level=logging.INFO)
    try:
        logging.info("--- Starting Evaluation ---")

        # Load artifacts
        model = joblib.load(model_path)
        df = pd.read_csv(data_path)

        X = df.iloc[:, :-1]
        y_true = df.iloc[:, -1]

        # Make predictions
        y_pred = model.predict(X)

        # Generate plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_true, y=y_pred)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted Housing Prices")

        # Save plot
        plt.savefig(plot_path)
        logging.info(f"Evaluation plot saved to {plot_path}")

    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise e
