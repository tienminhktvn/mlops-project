from google.cloud import aiplatform
from kfp.v2 import compiler, dsl

from src.data_ingestion import data_ingestion
from src.evaluation import evaluation
from src.preprocessing import preprocessing
from src.training import training

PROJECT_ID = "gen-lang-client-0021096577"
REGION = "europe-west1"
BUCKET_NAME = "gs://mlops-gen-lang-client-0021096577"
PIPELINE_ROOT = f"{BUCKET_NAME}/pipeline_root"


@dsl.pipeline(name="houseprice-pipeline", pipeline_root=PIPELINE_ROOT)
def houseprice_pipeline():
    ingestion_task = data_ingestion()

    preprocessing_task = preprocessing(input_dataset=ingestion_task.outputs["dataset"])

    training_task = training(
        preprocessed_dataset=preprocessing_task.outputs["preprocessed_dataset"],
        hyperparameters={"n_estimators": 100, "max_depth": 10, "random_state": 42},
    )

    evaluation_task = evaluation(
        model=training_task.outputs["model"],
        preprocessed_dataset=preprocessing_task.outputs["preprocessed_dataset"],
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=houseprice_pipeline, package_path="houseprice_pipeline.json"
    )

    aiplatform.init(project=PROJECT_ID, location=REGION)

    job = aiplatform.PipelineJob(
        display_name="houseprice_pipeline_job",
        template_path="houseprice_pipeline.json",
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False,
    )

    job.submit()
