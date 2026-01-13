from kfp.v2.dsl import Dataset, Input, Metrics, Model, Output, component


@component(
    base_image="europe-west1-docker.pkg.dev/gen-lang-client-0021096577/vertex-ai-pipeline/training:latest",
    output_component_file="training.yaml",
)
def training(
    preprocessed_dataset: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    hyperparameters: dict,
):
    import logging

    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    logging.basicConfig(level=logging.INFO)

    try:
        df = pd.read_csv(preprocessed_dataset.path)

        target_col = "price"
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        rf_model = RandomForestRegressor(
            n_estimators=hyperparameters.get("n_estimators", 100),
            max_depth=hyperparameters.get("max_depth", 10),
            random_state=hyperparameters.get("random_state", 42),
        )

        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_val)

        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        metrics.log_metric("mean_squared_error", mse)
        metrics.log_metric("r2_score", r2)

        joblib.dump(rf_model, model.path + ".joblib")
        joblib.dump(rf_model, model.path)

    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise e
