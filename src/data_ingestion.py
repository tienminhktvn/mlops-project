from kfp.v2.dsl import Dataset, Output, component


@component(
    base_image="europe-west1-docker.pkg.dev/gen-lang-client-0021096577/vertex-ai-pipeline/training:latest",
    output_component_file="data_ingestion.yaml",
)
def data_ingestion(dataset: Output[Dataset]):
    import logging

    import pandas as pd

    logging.basicConfig(level=logging.INFO)

    try:
        BUCKET_NAME = "mlops-gen-lang-client-0021096577"
        source_path = f"gs://{BUCKET_NAME}/data/Housing.csv"

        logging.info(f"Reading data from: {source_path}")
        df = pd.read_csv(source_path)

        df.to_csv(dataset.path, index=False)
        logging.info("Data ingestion completed.")

    except Exception as e:
        logging.error(f"Error during data ingestion: {e}")
        raise e
