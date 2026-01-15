import os
import time

from src.data_ingestion import data_ingestion
from src.evaluation import evaluation
from src.preprocessing import preprocessing
from src.training import training

# Define container paths
DATA_DIR = "/app/data"
OUTPUT_DIR = "/app/output"

# Define artifact paths
INPUT_CSV = os.path.join(DATA_DIR, "Housing.csv")
RAW_DATA_ARTIFACT = os.path.join(OUTPUT_DIR, "raw_data.csv")
PROCESSED_DATA_ARTIFACT = os.path.join(OUTPUT_DIR, "processed_data.csv")
SCALER_ARTIFACT = os.path.join(OUTPUT_DIR, "scaler.joblib")
MODEL_ARTIFACT = os.path.join(OUTPUT_DIR, "model.joblib")
METRICS_ARTIFACT = os.path.join(OUTPUT_DIR, "metrics.json")
PLOT_ARTIFACT = os.path.join(OUTPUT_DIR, "evaluation_plot.png")


def main():
    print(">>> PIPELINE EXECUTION STARTED...")
    start_time = time.time()

    # Step 1: Ingestion
    print("\n[1/4] Running Data Ingestion...")
    data_ingestion(input_path=INPUT_CSV, output_path=RAW_DATA_ARTIFACT)

    # Step 2: Preprocessing
    print("[2/4] Running Preprocessing...")
    preprocessing(
        input_path=RAW_DATA_ARTIFACT,
        output_data_path=PROCESSED_DATA_ARTIFACT,
        output_scaler_path=SCALER_ARTIFACT,
    )

    # Step 3: Training
    print("[3/4] Running Training...")
    training(
        input_path=PROCESSED_DATA_ARTIFACT,
        model_path=MODEL_ARTIFACT,
        metrics_path=METRICS_ARTIFACT,
        hyperparameters={"n_estimators": 100, "max_depth": 10, "random_state": 42},
    )

    # Step 4: Evaluation
    print("[4/4] Running Evaluation...")
    evaluation(
        model_path=MODEL_ARTIFACT,
        data_path=PROCESSED_DATA_ARTIFACT,
        plot_path=PLOT_ARTIFACT,
    )

    elapsed = time.time() - start_time
    print(f"\n>>> PIPELINE COMPLETED SUCCESSFULLY in {elapsed:.2f} seconds.")
    print(f">>> Check {OUTPUT_DIR} for artifacts.")


if __name__ == "__main__":
    main()
