from kfp.v2.dsl import HTML, Dataset, Input, Metrics, Model, Output, component


@component(
    base_image="europe-west1-docker.pkg.dev/gen-lang-client-0021096577/vertex-ai-pipeline/training:latest",
    output_component_file="evaluation.yaml",
)
def evaluation(
    model: Input[Model],
    preprocessed_dataset: Input[Dataset],
    metrics: Output[Metrics],
    html: Output[HTML],
):
    import base64
    import logging
    from io import BytesIO

    import joblib
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from sklearn.metrics import mean_squared_error, r2_score

    logging.basicConfig(level=logging.INFO)

    try:
        rf_model = joblib.load(model.path)
        df = pd.read_csv(preprocessed_dataset.path)

        target_col = "price"
        X = df.drop(columns=[target_col])
        y = df[target_col]

        y_pred = rf_model.predict(X)

        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        metrics.log_metric("eval_mse", mse)
        metrics.log_metric("eval_r2", r2)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y, y=y_pred)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title(f"Actual vs Predicted Prices (R2: {r2:.2f})")

        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()

        html_content = f"""
        <html>
            <body>
                <h1>Model Performance Report</h1>
                <ul>
                    <li>MSE: {mse:.4f}</li>
                    <li>R2: {r2:.4f}</li>
                </ul>
                <img src="data:image/png;base64,{image_base64}" alt="Scatter Plot">
            </body>
        </html>
        """

        with open(html.path, "w") as f:
            f.write(html_content)

    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        raise e
