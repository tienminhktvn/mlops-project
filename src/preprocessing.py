from kfp.v2.dsl import Dataset, Input, Output, component


@component(
    base_image="europe-west1-docker.pkg.dev/gen-lang-client-0021096577/vertex-ai-pipeline/training:latest",
    output_component_file="preprocessing.yaml",
)
def preprocessing(
    input_dataset: Input[Dataset],
    preprocessed_dataset: Output[Dataset],
):
    import logging

    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    logging.basicConfig(level=logging.INFO)

    try:
        df = pd.read_csv(input_dataset.path)

        target_col = "price"
        X = df.drop(columns=[target_col])
        y = df[target_col]

        binary_cols = [
            "mainroad",
            "guestroom",
            "basement",
            "hotwaterheating",
            "airconditioning",
            "prefarea",
        ]

        def binary_map(x):
            return x.map({"yes": 1, "no": 0})

        X[binary_cols] = X[binary_cols].apply(binary_map)

        if "furnishingstatus" in X.columns:
            status = pd.get_dummies(X["furnishingstatus"], drop_first=True)
            X = pd.concat([X, status], axis=1)
            X.drop("furnishingstatus", axis=1, inplace=True)
            X = X.astype(int)

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        df_processed = pd.concat([X_scaled, y], axis=1)

        df_processed.to_csv(preprocessed_dataset.path, index=False)

    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise e
