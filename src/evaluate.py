import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_parquet('dataset/processed.parquet', engine='pyarrow')
X_train, X_test, y_train, y_test = train_test_split(df['lemmatized_sent'], df['is_spam'], test_size=0.3)
bundle = joblib.load("artifacts/spam_count_lemma_bundle.joblib")
model = bundle["model"]
vectorizer = bundle["vectorizer"]

X_test = vectorizer.transform(X_test)
pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, pred)}')
y_proba = model.predict_proba(X_test)[:,1]
print(f'Proba: {y_proba}')