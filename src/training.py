import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import joblib
from zenml import step

@step
def train_model(X_train: pd.Series, y_train: pd.Series) -> tuple[SVC, CountVectorizer]:
    # X_train, X_test, y_train, y_test = train_test_split(df['lemmatized_sent'], df['is_spam'], test_size=0.3)

    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train)

    model = SVC(probability=True)
    model.fit(X_train, y_train)

    # joblib.dump(model, "artifacts/spam_count_lemma_onfly.joblib", compress=3)
    joblib.dump(
        {"model": model, "vectorizer": vectorizer},
        "artifacts/spam_count_lemma_bundle.joblib",
        compress=3
    )
    return (model, vectorizer)