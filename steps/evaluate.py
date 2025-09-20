import pandas as pd
from model.model_evaluation import AccuracyScore
from zenml import step
from zenml.client import Client
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer
from typing import Annotated
import mlflow
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: ClassifierMixin, vectorizer: CountVectorizer, X_test: pd.Series, y_test: pd.Series) -> Annotated[float, "Accuracy Score"]:
    """ Evaluate the trained model with the provided test data. """
    try:
        # mlflow.sklearn.autolog()
        model_predictions = model.predict(vectorizer.transform(X_test))
        accuracy_scorer = AccuracyScore()
        accuracy_score = accuracy_scorer.evaluate(y_test, model_predictions)
        mlflow.log_metric("Accuracy", accuracy_score)
        return accuracy_score
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise