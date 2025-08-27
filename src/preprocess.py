import pandas as pd
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

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

df = pd.read_csv("dataset/spam.csv", encoding='latin1')
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
no_dup_df.to_parquet('dataset/processed.parquet', engine='pyarrow')