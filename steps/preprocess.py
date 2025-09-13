import pandas as pd
from typing import Tuple, Annotated
from zenml import step
from model.data_preprocessing import EmailDataPreprocessor, DataSplitter

@step
def preprocess_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.Series, "X_train"], 
    Annotated[pd.Series, "X_test"], 
    Annotated[pd.Series, "y_train"], 
    Annotated[pd.Series, "y_test"]
    ]:
    preprcessor = EmailDataPreprocessor()
    processed_df = preprcessor.handle_data(df)
    splitter = DataSplitter()
    X_train, X_test, y_train, y_test = splitter.handle_data(processed_df)
    return X_train, X_test, y_train, y_test
    