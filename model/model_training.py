from abc import ABC, abstractmethod
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from typing import Tuple

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
    def train(self, X_train, y_train) -> Tuple[SVC, CountVectorizer]:
        """ Train an SVM model with the provided training data. """
        vectorizer = CountVectorizer()
        X_train = vectorizer.fit_transform(X_train)

        model = SVC(probability=True)
        model.fit(X_train, y_train)

        return (model, vectorizer)
