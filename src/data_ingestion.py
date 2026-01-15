import logging
import os

import pandas as pd


def data_ingestion(input_path: str, output_path: str):
    """
    Loads raw data and saves it to the artifact location.
    """
    logging.basicConfig(level=logging.INFO)
    try:
        logging.info("--- Starting Ingestion ---")

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found at {input_path}")

        # Load dataset
        df = pd.read_csv(input_path)
        logging.info(f"Data loaded: {df.shape}")

        # Save raw data for the next step
        df.to_csv(output_path, index=False)
        logging.info(f"Raw data saved to {output_path}")

    except Exception as e:
        logging.error(f"Ingestion failed: {e}")
        raise e
