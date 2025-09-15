from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """Model Configurations"""

    model_name: str = "SVM"
    fine_tuning: bool = False
