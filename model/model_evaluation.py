from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score
import numpy as np

class Evaluation(ABC):
    """
    Abstract base class for model evaluation.
    """
    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

class AccuracyScore(Evaluation):
    """
    Concrete implementation of Evaluation using accuracy score.
    """
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ Evaluate the model using accuracy score. """
        return accuracy_score(y_true, y_pred)
    
    

