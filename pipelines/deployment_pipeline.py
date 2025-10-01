from steps.ingest_data import ingest_data
from steps.preprocess import preprocess_data
from steps.training import train_model
from steps.evaluate import evaluate_model
import os
import numpy as np
import pandas as pd
import json

from zenml import step, pipeline
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW, SKLEARN
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from pydantic import BaseModel
from .utils import get_data_for_test

docker_settings = DockerSettings(required_integrations=[MLFLOW, SKLEARN])

requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")

@step
def dynamic_importer() -> str:
    '''Loads the latest dataset'''
    data = get_data_for_test()
    return data

class DeploymentTriggerConfig(BaseModel):
    '''Parameteres for triggering the deployment'''
    min_acc: float = 0.9

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig
) -> bool:
    '''
    An implementation of the deployment trigger
    Only triggers the deployment if the accuracy satisfies the minimum accuracy
    '''
    return accuracy > config.min_acc

class MLFlowDeploymentLoaderStepParameters(BaseModel):
    """
    MLFlow deployment parameters

    Attributes:
        pipeline_name: name of the pipeline that deployed the MLFlow prediction server
        step_name: the name of the step that deployed the MLFlow server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    pipeline_name: str
    step_name: str
    running: bool = True
    model_name: str = 'model'

@step
def prediction_service_loader(
    parameters: MLFlowDeploymentLoaderStepParameters
) -> MLFlowDeploymentService:
    '''Get the prediction service started by the deployment pipeline'''
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    
    existing_services = model_deployer.find_model_server(
        running = parameters.running,
        pipeline_name=parameters.pipeline_name,
        pipeline_step_name = parameters.step_name,
        model_name= parameters.model_name
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{parameters.step_name} step in the {parameters.pipeline_name} "
            f"pipeline for the '{parameters.model_name}' model is currently "
            f"running."
        )
    print(existing_services)
    print(type(existing_services))
    return existing_services[0]

def predictor(
    service: MLFlowDeploymentService,
    data: np.ndarray
)->np.ndarray:
    """Run an inference request"""
    service.start(timeout=10)
    data = json.load(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = ['lemmatized_sent', 'is_spam']

    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction

@pipeline(enable_cache=True, settings={'docker': docker_settings})
def continuous_deployment_pipeline(
    data_path: str,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    pipeline = train_model(X_train=X_train, y_train=y_train)
    acc_score = evaluate_model(pipeline=pipeline, X_test=X_test, y_test=y_test)
    deployment_decision = deployment_trigger(acc_score, config = DeploymentTriggerConfig())
    mlflow_model_deployer_step(
        model = pipeline,
        deploy_decision= deployment_decision,
        workers=workers,
        timeout=timeout
    )

@pipeline(enable_cache=True, settings={'docker': docker_settings})
def inference_pipeline(
    pipeline_name: str, 
    pipeline_step_name: str
):
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        MLFlowDeploymentLoaderStepParameters(
            pipeline_name=pipeline_name,
            step_name=pipeline_step_name,
            model_name="model",
            running=False,
        )  
    )
    predictor(service=model_deployment_service, data=batch_data)