import joblib
import re
import string
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

def custom_preprocessor(text):
    if not isinstance(text, str): return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def custom_tokenizer(text):
    tokens = word_tokenize(text)
    filter_punct = string.punctuation.replace('!', '').replace('?', '')
    clean_tokens = []
    for t in tokens:
        if t not in filter_punct:
            clean_tokens.append(t)
    return clean_tokens

def evaluate_and_save_results(pipeline, X_val, y_val, label_map):
    output_dir = "results_SVM"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    y_pred = pipeline.predict(X_val)
    labels = list(label_map.keys())
    target_names = list(label_map.values())

    report = classification_report(y_val, y_pred, target_names=target_names)
    print(report)

    with open(os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_val, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    df_results = pd.DataFrame({'text': X_val, 'true': y_val, 'pred': y_pred})
    df_errors = df_results[df_results['true'] != df_results['pred']].copy()
    
    if not df_errors.empty:
        df_errors['length'] = df_errors['text'].apply(lambda x: len(custom_tokenizer(custom_preprocessor(x))))
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df_errors, x='length', kde=True, bins=30, color='red', alpha=0.6)
        plt.title('Error Sentence Length Distribution')
        plt.xlabel('Word Count')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "error_length_histogram.png"))
        plt.close()
        
        df_errors.to_csv(os.path.join(output_dir, "error_samples.csv"), index=False, encoding='utf-8')

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
        ngram_range=(1, 3), 
        min_df=2, 
        max_features=25000 
    )
    
    svm = LinearSVC(class_weight='balanced', random_state=42, max_iter=5000, C=0.5)
    
    pipeline = Pipeline([
        ('vectorizer', tfidf),
        ('svm', svm)
    ])

    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, 'sst5_svm_pro.joblib')
    evaluate_and_save_results(pipeline, X_val, y_val, label_map)

if __name__ == "__main__":
    train_svm_pro()