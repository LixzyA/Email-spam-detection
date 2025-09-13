import pandas as pd
from model.model_evaluation import AccuracyScore
from zenml import step
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@step
def evaluate_model(model: ClassifierMixin, vectorizer: CountVectorizer, X_test: pd.Series, y_test: pd.Series) -> None:
    """ Evaluate the trained model with the provided test data. """
    try:
        model_predictions = model.predict(vectorizer.transform(X_test))
        accuracy_scorer = AccuracyScore()
        accuracy_score = accuracy_scorer.evaluate(y_test, model_predictions)
        logging.info(f"Model Accuracy: {accuracy_score:.4f}")
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise