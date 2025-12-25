import joblib
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

def download_nltk_resources():
    resources = ['punkt', 'punkt_tab']
    for res in resources:
        try:
            if res == 'punkt':
                nltk.data.find('tokenizers/punkt')
            else:
                nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            print(f"⬇️ Đang tải NLTK resource: {res}...")
            nltk.download(res, quiet=True)

download_nltk_resources()

def custom_preprocessor(text):
    """Làm sạch văn bản cơ bản."""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def custom_tokenizer(text):
    """
    Tokenizer thông minh hơn của NLTK:
    - Tách 'don't' -> 'do', "n't" (giúp bắt phủ định tốt hơn)
    - Giữ lại dấu '!' và '?' (quan trọng cho cảm xúc mạnh)
    """
    tokens = word_tokenize(text)
    filter_punct = string.punctuation.replace('!', '').replace('?', '')
    
    clean_tokens = []
    for t in tokens:
        if t not in filter_punct:
            clean_tokens.append(t)
    return clean_tokens

# --- 3. MAIN FUNCTION ---
def train_logistic_regression():
    dataset = load_dataset("SetFit/sst5")
    
    X_train = dataset['train']['text']
    y_train = dataset['train']['label']
    
    X_val = dataset['validation']['text']
    y_val = dataset['validation']['label']
    
    label_map = {0: 'Very Negative', 1: 'Negative', 2: 'Neutral', 3: 'Positive', 4: 'Very Positive'}

 

    tfidf = TfidfVectorizer(
        preprocessor=custom_preprocessor,
        tokenizer=custom_tokenizer,
        ngram_range=(1, 3),
        min_df=2, 
        max_features=25000 
    )
    

    clf = LogisticRegression(
        solver='saga', 
        multi_class='multinomial', 
        max_iter=2000, 
        random_state=42,
        n_jobs=-1 
    )
    
    pipeline = Pipeline([
        ('vectorizer', tfidf), 
        ('clf', clf)
    ])


    pipeline.fit(X_train, y_train)
    

    y_pred = pipeline.predict(X_val)
    
    print("-" * 30)
    print(f"Validation Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print("-" * 30)
    print("Classification Report (Validation):")
    print(classification_report(y_val, y_pred, target_names=label_map.values()))
    

    model_filename = 'sst5_logistic_regression_pro.joblib'
    joblib.dump(pipeline, model_filename)


if __name__ == "__main__":
    train_logistic_regression()