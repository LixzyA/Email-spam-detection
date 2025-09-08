# Spam Email Detection 📧

A machine learning project to classify emails as "spam" or "ham" using Natural Language Processing (NLP) techniques and a Support Vector Classifier (SVC).

## 📝 Overview

This project walks through the process of building a spam detection model. It covers data cleaning, exploratory data analysis (EDA) to understand the characteristics of spam vs. non-spam emails, text preprocessing using both stemming and lemmatization, and finally, training and evaluating an SVC model.

## 📊 Dataset

The project utilizes the `spam.csv` dataset, which contains two primary columns:
* `v1`: Label indicating whether the email is 'spam' or 'ham'.
* `v2`: The raw text content of the email.

## 🛠️ Technologies & Libraries

* **Python 3**
* **Core Libraries:**
    * `pandas`: For data manipulation and analysis.
    * `numpy`: For numerical operations.
    * `matplotlib` : For data visualization.
* **NLP & Machine Learning:**
    * `nltk`: For text preprocessing tasks including:
        * Tokenization (`word_tokenize`, `sent_tokenize`).
        * Stop word removal (`stopwords`).
        * Stemming (`PorterStemmer`).
        * Lemmatization (`WordNetLemmatizer`).
        * Part-of-Speech (POS) tagging (`pos_tag`).
    * `scikit-learn`: For:
        * Splitting data (`train_test_split`).
        * Feature extraction (`CountVectorizer`).
        * Model implementation (`SVC`).
        * Performance evaluation (`accuracy_score`).

## ⚙️ Project Workflow

1.  **Data Loading and Initial Cleaning**:
    * Load the `spam.csv` dataset.
    * Rename columns for clarity (e.g., `v1` to `is_spam`, `v2` to `email`).
    * Drop irrelevant columns.

2.  **Data Integrity**:
    * Check for and handle missing values (none found in critical columns after initial cleanup).
    * Identify and remove duplicate email entries to ensure data quality.

3.  **Exploratory Data Analysis (EDA)**:
    * Visualize the class distribution (spam vs. ham) using a pie chart.
    * Engineer new features:
        * `email_length`: Character count per email.
        * `word_count`: Word count per email.
        * `sentence_count`: Sentence count per email.
    * Compare the average of these engineered features across spam and ham emails using bar charts to identify potential patterns.

4.  **Text Preprocessing**:
    * **Tokenization**: Break down email text into individual words.
    * **Stop Word Removal**: Filter out common English stop words.
    * **Normalization Showdown**:
        * **Stemming**: Apply `PorterStemmer` to reduce words to their base form.
        * **Lemmatization**: Employ `WordNetLemmatizer` with POS tagging for more contextually accurate base form conversion.
    * The target variable `is_spam` is converted to a binary format (1 for spam, 0 for ham).

5.  **Feature Extraction**:
    * Utilize `CountVectorizer` to transform the preprocessed (stemmed and lemmatized) email texts into numerical feature matrices suitable for model training.

6.  **Model Training & Evaluation**:
    * Split the data into training (70%) and testing (30%) sets for both stemmed and lemmatized versions.
    * Train a Support Vector Classifier (SVC) model.
    * Evaluate the model's performance on the test sets using accuracy as the primary metric.

## 📈 Results

The model achieved the following accuracy scores on the respective test sets:

* **With Lemmatized Text**: ~98.07%
* **With Stemmed Text**: ~97.74%

Lemmatization yielded a slightly higher accuracy, suggesting its more nuanced approach to text normalization was beneficial for this particular dataset and model.

## 🚀 Setup & Usage

1.  **Prerequisites**:
    * Python 3.x
    * Jupyter Notebook or any Python IDE

2.  **Installation**:
    Clone the repository and install the required packages:
    ```bash
    pip install pandas numpy matplotlib seaborn nltk scikit-learn
    ```
    You will also need to download NLTK resources. Run the following in a Python interpreter after installing NLTK:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger')
    ```
    *(These download commands are also present in the first cell of the notebook)*

3.  **Running the Notebook**:
    * Ensure the `spam.csv` file is in the same directory as the `main.ipynb` notebook.
    * Open and run the `main.ipynb` notebook in Jupyter.

---

Feel free to explore the notebook for a detailed implementation!
