# Example pipeline for spam detection
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import joblib
from src.preprocess import preprocess_data
from src.training import train_model
from src.evaluate import evaluate_model
from zenml import pipeline

@pipeline
def pipeline_train(data_path:str):
    # data = preprocess_data(data_path=data_path)
    X_train, X_test, y_train, y_test = preprocess_data(data_path=data_path)
    model, vectorizer = train_model(X_train=X_train, y_train=y_train)
    evaluate_model(model=model, vectorizer=vectorizer, X_test=X_test, y_test=y_test)


# # Custom text preprocessing function
# def preprocess_text(text):
#     tokens = nltk.word_tokenize(text)
#     tokens = [t for t in tokens if t.isalpha()]
#     tokens = [t.lower() for t in tokens if t.lower() not in stopwords.words('english')]
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(t) for t in tokens]
#     return ' '.join(tokens)

# # Pipeline definition
# pipeline = Pipeline([
#     ('preprocess', CountVectorizer(preprocessor=preprocess_text)),
#     ('svc', SVC())
# ])

# joblib.dump(pipeline, 'artifacts/spam_detection_pipeline.pkl')