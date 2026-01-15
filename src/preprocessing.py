import logging

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocessing(input_path: str, output_data_path: str, output_scaler_path: str):
    """
    Preprocesses data: Scales numeric features and encodes categorical ones.
    """
    logging.basicConfig(level=logging.INFO)
    try:
        logging.info("--- Starting Preprocessing ---")

        df = pd.read_csv(input_path)

        # Separate features and target
        target_col = "price"
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Identify column types
        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
        categorical_features = X.select_dtypes(include=["object"]).columns

        # Define transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ]
        )

        # Apply transformations
        X_processed = preprocessor.fit_transform(X)

        # Reconstruct DataFrame and attach target
        processed_data = pd.DataFrame(X_processed)
        processed_data["target"] = y.values

        # Save processed data and the scaler object
        processed_data.to_csv(output_data_path, index=False)
        joblib.dump(preprocessor, output_scaler_path)

        logging.info(f"Preprocessed data saved to {output_data_path}")

    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        raise e
