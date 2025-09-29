import pandas as pd
from model.model_training import SVMTrainer
from zenml import step
from zenml.client import Client
from sklearn.pipeline import Pipeline
import mlflow
import logging

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(X_train: pd.Series, y_train: pd.Series) -> Pipeline:
    """ Train an SVM model with the provided training data. """
    try:
        mlflow.sklearn.autolog()
        trainer = SVMTrainer()
        pipeline = trainer.train(X_train, y_train)
        return pipeline
    except Exception as e:
        logging.error("Error during model training: %s", e)
        raise e
