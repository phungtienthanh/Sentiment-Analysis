import joblib
import re
import string
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.tokenize import word_tokenize


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

def custom_preprocessor(text):
    """
    Hàm làm sạch dữ liệu cơ bản:
    - Chuyển về chữ thường
    - Xử lý khoảng trắng thừa
    """
    if not isinstance(text, str): return ""
    
    text = text.lower() 
    text = text.strip() 
    text = re.sub(r'\s+', ' ', text) 
    
    return text

def custom_tokenizer(text):
    """
    Tokenizer dùng NLTK để giữ lại dấu câu quan trọng (!, ?)
    và tách từ tốt hơn (ví dụ: don't -> do n't)
    """
    tokens = word_tokenize(text)
    filter_punct = string.punctuation.replace('!', '').replace('?', '')  
    clean_tokens = []
    for t in tokens:
        if t not in filter_punct:
            clean_tokens.append(t)
            
    return clean_tokens

# --------------------------------------------------

def train_svm_pro():

    dataset = load_dataset("SetFit/sst5")
    
    X_train = dataset['train']['text']
    y_train = dataset['train']['label']
    X_val = dataset['validation']['text']
    y_val = dataset['validation']['label']
    
    label_map = {0: 'Very Negative', 1: 'Negative', 2: 'Neutral', 3: 'Positive', 4: 'Very Positive'}


    tfidf = TfidfVectorizer(
        preprocessor=custom_preprocessor,
        tokenizer=custom_tokenizer, 
        ngram_range=(1, 3), # Unigram + Bigram + Trigram
        min_df=2, 
        max_features=25000 
    )
    
    svm = LinearSVC(class_weight='balanced', random_state=42, max_iter=5000, C=0.5)
    
    pipeline = Pipeline([
        ('vectorizer', tfidf),
        ('svm', svm)
    ])

 
    pipeline.fit(X_train, y_train)
    

    y_pred = pipeline.predict(X_val)
    
    print("-" * 30)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")
    print("-" * 30)
    print("Classification Report:")
    print(classification_report(y_val, y_pred, target_names=label_map.values()))
    
    model_filename = 'sst5_svm_pro.joblib'
    joblib.dump(pipeline, model_filename)


if __name__ == "__main__":
    train_svm_pro()