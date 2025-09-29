import logging

import pandas as pd
from model.data_preprocessing import EmailDataPreprocessor


def get_data_for_test():
    try:
        df = pd.read_csv("./dataset/spam.csv")
        df = df.sample(n=100)
        preprocess_strategy = EmailDataPreprocessor()
        df = preprocess_strategy.handle_data(df)
        df.drop(["review_score"], axis=1, inplace=True)
        result = df.to_json(orient="split")
        return result
    except Exception as e:
        logging.error(e)
        raise e
