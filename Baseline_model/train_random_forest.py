import joblib
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
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
    text = text.lower().strip()
    return re.sub(r'\s+', ' ', text)

def custom_tokenizer(text):
    tokens = word_tokenize(text)
    filter_punct = string.punctuation.replace('!', '').replace('?', '')
    return [t for t in tokens if t not in filter_punct]

# --- MAIN ---
def train_random_forest():

    dataset = load_dataset("SetFit/sst5")
    X_train, y_train = dataset['train']['text'], dataset['train']['label']
    X_val, y_val = dataset['validation']['text'], dataset['validation']['label']
    
    label_map = {0: 'Very Negative', 1: 'Negative', 2: 'Neutral', 3: 'Positive', 4: 'Very Positive'}

 
    

    tfidf = TfidfVectorizer(
        preprocessor=custom_preprocessor, tokenizer=custom_tokenizer,
        ngram_range=(1, 3), min_df=2, max_features=25000
    )
    

    rf = RandomForestClassifier(
        n_estimators=100, 
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1,
        verbose=1
    )
    
    pipeline = Pipeline([('vectorizer', tfidf), ('rf', rf)])

    print("⚙️  Đang training Random Forest (Có thể tốn RAM hơn SVM)...")
    pipeline.fit(X_train, y_train)
    
    print("📊 Đang đánh giá trên tập VALIDATION...")
    y_pred = pipeline.predict(X_val)
    
    print("-" * 30)
    print(f"Validation Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print("-" * 30)
    print("Classification Report:")
    print(classification_report(y_val, y_pred, target_names=label_map.values()))
    
    joblib.dump(pipeline, 'sst5_random_forest.joblib')


if __name__ == "__main__":
    train_random_forest()