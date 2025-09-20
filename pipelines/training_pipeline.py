# Example pipeline for spam detection
from sklearn.pipeline import Pipeline
from zenml import pipeline

from steps.ingest_data import ingest_data
from steps.preprocess import preprocess_data
from steps.training import train_model
from steps.evaluate import evaluate_model

@pipeline
def pipeline_train(data_path:str):
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model, vectorizer = train_model(X_train=X_train, y_train=y_train)
    acc_score = evaluate_model(model=model, vectorizer=vectorizer, X_test=X_test, y_test=y_test)
