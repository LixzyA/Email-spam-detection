from abc import ABC, abstractmethod
import pandas as pd
import logging
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
from zenml import step
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Map NLTK POS to WordNet POS
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

class DataPreprocessor(ABC):
    """
    Abstract base class for data preprocessing.
    """
    @abstractmethod
    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame | pd.Series:
        pass

class EmailDataPreprocessor(DataPreprocessor):
    """
    Concrete implementation of DataPreprocessor for Email spam data.
    """
    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame | pd.Series:
        """ Preprocess the email data by removing duplicates, tokenizing, removing stopwords, and lemmatizing. """
        logging.info("Starting preprocessing of data")

        try:
            df.rename({'Category':'is_spam', 'Message':'email'}, axis =1, inplace=True)
            no_dup_df = df.drop_duplicates(['email'])
            tokens = no_dup_df['email'].apply(word_tokenize)

            stop_words = set(stopwords.words('english'))
            filtered_tokens = tokens.apply(lambda tokens: [word for word in tokens if word.lower() not in stop_words])

            lemmatizer = WordNetLemmatizer()
            lemmatized_series_pos = filtered_tokens.apply(
                lambda tokens: [
                    lemmatizer.lemmatize(word, get_wordnet_pos(pos))
                    for word, pos in pos_tag(tokens)
                ]
            )
            no_dup_df['lemmatized_sent'] = [' '.join(word) for word in lemmatized_series_pos]
            no_dup_df['is_spam'] = no_dup_df['is_spam'].map({'spam': 1, 'ham': 0})

            logging.info("Completed preprocessing of data")
            return no_dup_df
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            raise e

class DataSplitter(DataPreprocessor):
    """
        Strategy for splitting data into training and testing sets.
    """
    def handle_data(self, df: pd.DataFrame) -> pd.Series:

        """ Splits the data into training and testing sets. """

        logging.info("Starting data splitting")
        try:
            X_train, X_test, y_train, y_test = train_test_split(df['lemmatized_sent'], df['is_spam'], test_size=0.3)
            logging.info("Completed data splitting")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error during data splitting: {e}")
            raise e
