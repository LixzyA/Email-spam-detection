import pandas as pd
from model.model_training import SVMTrainer
from zenml import step
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer
from typing import Tuple

@step
def train_model(X_train: pd.Series, y_train: pd.Series) -> Tuple[ClassifierMixin, CountVectorizer]:
    """ Train an SVM model with the provided training data. """
    trainer = SVMTrainer()
    model, vectorizer = trainer.train(X_train, y_train)
    return model, vectorizer
   