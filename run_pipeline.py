from pipelines.training_pipeline import pipeline_train
from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    pipeline_train(data_path='dataset/spam.csv')
