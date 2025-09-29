from abc import ABC, abstractmethod
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline, Pipeline

class ModelTrainer(ABC):
    """
    Abstract base class for model training.
    """
    @abstractmethod
    def train(self, X_train, y_train):
        pass

class SVMTrainer(ModelTrainer): 
    """
    Concrete implementation of ModelTrainer using Support Vector Machine.
    """
    def train(self, X_train, y_train) -> Pipeline:
        """ Train an SVM model with the provided training data. """
        pipeline = make_pipeline(
            ("Vectorizer", CountVectorizer()),
            ("Model", SVC(probability=True))
        )
        pipeline.fit()
        return pipeline
