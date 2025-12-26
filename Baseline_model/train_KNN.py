import joblib
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score


def download_nltk_resources():
    resources = ['punkt', 'punkt_tab']
    for res in resources:
        try:
            if res == 'punkt': nltk.data.find('tokenizers/punkt')
            else: nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download(res, quiet=True)
download_nltk_resources()

def custom_preprocessor(text):
    if not isinstance(text, str): return ""
    return re.sub(r'\s+', ' ', text.lower().strip())

def custom_tokenizer(text):
    tokens = word_tokenize(text)
    filter_punct = string.punctuation.replace('!', '').replace('?', '')
    return [t for t in tokens if t not in filter_punct]


def run_knn():
    dataset = load_dataset("SetFit/sst5")
    X_train, y_train = dataset['train']['text'], dataset['train']['label']
    X_val, y_val = dataset['validation']['text'], dataset['validation']['label']
    
    label_map = {0: 'Very Negative', 1: 'Negative', 2: 'Neutral', 3: 'Positive', 4: 'Very Positive'}
    

    tfidf = TfidfVectorizer(
        preprocessor=custom_preprocessor, tokenizer=custom_tokenizer,
        ngram_range=(1, 3), min_df=2, max_features=25000
    )
    
 
    knn = KNeighborsClassifier(n_neighbors=25, metric='cosine', algorithm='brute', n_jobs=-1)
    
    pipeline = Pipeline([('vectorizer', tfidf), ('knn', knn)])


    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_val)
    
    print("-" * 30)
    print(f"Validation Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print("-" * 30)
    print("Classification Report:")
    print(classification_report(y_val, y_pred, target_names=label_map.values()))
    
    joblib.dump(pipeline, 'sst5_knn.joblib')


if __name__ == "__main__":
    run_knn()