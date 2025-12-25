import joblib
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# --- XỬ LÝ NLTK & DATA PREPROCESSING ---
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
def train_xgboost():
    dataset = load_dataset("SetFit/sst5")
    X_train, y_train = dataset['train']['text'], dataset['train']['label']
    X_val, y_val = dataset['validation']['text'], dataset['validation']['label']
    
    label_map = {0: 'Very Negative', 1: 'Negative', 2: 'Neutral', 3: 'Positive', 4: 'Very Positive'}

    
    tfidf = TfidfVectorizer(
        preprocessor=custom_preprocessor, tokenizer=custom_tokenizer,
        ngram_range=(1, 3), min_df=2, max_features=15000
    )
    
    # XGBoost Setup
    # objective='multi:softmax': Bắt buộc cho bài toán đa lớp
    # num_class=5: Số lượng lớp
    # tree_method='hist': Tăng tốc độ train trên dữ liệu lớn
    xgb = XGBClassifier(
        objective='multi:softmax',
        num_class=5,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        tree_method='hist' # Giúp train nhanh hơn
    )
    
    pipeline = Pipeline([('vectorizer', tfidf), ('xgb', xgb)])


    pipeline.fit(X_train, y_train)
    
   
    y_pred = pipeline.predict(X_val)
    
    print("-" * 30)
    print(f"Validation Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print("-" * 30)
 
    print(classification_report(y_val, y_pred, target_names=label_map.values()))
    
    joblib.dump(pipeline, 'sst5_xgboost.joblib')
    print("✅ Đã lưu model: sst5_xgboost.joblib")

if __name__ == "__main__":
    train_xgboost()