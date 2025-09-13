import pandas as pd
import logging
from zenml import step

class DataIngestion():
    """
    Concrete implementation of DataIngestion for email spam dataset.
    """
    def __init__(self) -> None:
        """ Initialize the data ingestion class. """
        pass

    def get_data(self, path: str = 'dataset/spam.csv') -> pd.DataFrame:
        """ Ingest data from the given path and return a DataFrame. """
        try:
            df = pd.read_csv(path, encoding='latin1')
            return df
        except Exception as e:
            logging.error(f"Error reading the data from {path}: {e}")
            raise e
        
@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Args:
        None
    Returns:
        df: pd.DataFrame
    """
    
    data_ingestion = DataIngestion()
    df = data_ingestion.get_data(data_path)
    return df
    