from src.config.configuration import EvaluationConfig
from src.utils.utlis import *
from src.constants import *
import mlflow
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score


class ModelEvaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.scores = {}

    def save_score(self):
        save_json(path=Path("artifacts/model_evaluation/metrics.json"), data=self.scores)

    def evaluate_model(self):
        test_data = pd.read_csv(self.config.test_data)
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        model = joblib.load(self.config.model_path)
        y_pred = model.predict(X_test)[:, 1]

        roc = roc_auc_score(y_test, y_pred)
        self.scores["ROC Score"] = roc
        return model, X_test  # Return X_test as well for later use

    def log_with_mlflow(self, model, X_test):
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        with mlflow.start_run():
            mlflow.log_param("Model Path", self.config.model_path)
            mlflow.log_metric("ROC Score", self.scores["ROC Score"])

            # Provide an example input (for a model that expects a 2D array like X_test)
            input_example = X_test[:5]  # Using the first 5 samples from X_test as an example

            mlflow.sklearn.log_model(model, "Keras Model", input_example=input_example)

    def evaluation(self):
        model, X_test = self.evaluate_model()  # Capture X_test from the evaluate_model method

        # Save scores locally
        self.save_score()

        # Log evaluation with MLflow
        self.log_with_mlflow(model, X_test)  # Pass X_test to the log_with_mlflow method
