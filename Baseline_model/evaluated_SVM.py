import joblib
import re
import string
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
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

def evaluate_and_save_results(pipeline, X_test, y_test, label_map):
    output_dir = "results_SVM"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    y_pred = pipeline.predict(X_test)
    labels = list(label_map.keys())
    target_names = list(label_map.values())

    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)

    with open(os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # --- BẮT ĐẦU PHẦN VẼ BIỂU ĐỒ THEO CODE THAM KHẢO ---
    all_preds = np.array(y_pred)
    all_labels = np.array(y_test)
    
    # Tính độ dài câu (word count)
    all_lengths = np.array([len(custom_tokenizer(custom_preprocessor(t))) for t in X_test])
    
    # 1. Tạo bins
    max_len = np.max(all_lengths) if len(all_lengths) > 0 else 0
    # Tạo 16 điểm mốc -> 15 khoảng (bins)
    bins = np.linspace(0, max_len, 16) 
    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = (bins[1] - bins[0]) * 0.8

    correct_counts = []
    incorrect_counts = []
    
    for i in range(len(bins)-1):
        # Mask lọc các câu có độ dài trong khoảng bin hiện tại
        mask = (all_lengths >= bins[i]) & (all_lengths < bins[i+1])
        if np.any(mask):
            bin_preds = all_preds[mask]
            bin_labels = all_labels[mask]
            
            correct = np.sum(bin_preds == bin_labels)
            incorrect = np.sum(bin_preds != bin_labels)
            
            correct_counts.append(correct)
            incorrect_counts.append(incorrect)
        else:
            correct_counts.append(0)
            incorrect_counts.append(0)

    # 2. Vẽ biểu đồ
    plt.figure(figsize=(14, 8), facecolor='white')
    
    color_correct = '#27ae60'   # Emerald Green
    color_incorrect = '#e74c3c' # Alizarin Red
    
    # Cột Đúng (dưới)
    plt.bar(bin_centers, correct_counts, width=width, 
            label='Correct Predictions', color=color_correct, 
            alpha=0.85, edgecolor='white', linewidth=0.5)
    
    # Cột Sai (trên)
    plt.bar(bin_centers, incorrect_counts, width=width, 
            bottom=correct_counts, label='Misclassified', 
            color=color_incorrect, alpha=0.9, 
            edgecolor='white', linewidth=0.5)

    # 3. Tinh chỉnh hiển thị
    plt.title('Stacked Analysis: Accuracy vs. Error by Sentence Length (SVM)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Sentence Length (Number of words)', fontsize=13)
    plt.ylabel('Number of Sentences', fontsize=13)
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(frameon=True, shadow=True, fontsize=11, loc='upper right')

    # Hiển thị tỷ lệ % lỗi
    for i in range(len(bin_centers)):
        total = correct_counts[i] + incorrect_counts[i]
        if total > 5: 
            err_rate = (incorrect_counts[i] / total) * 100
            plt.text(bin_centers[i], total + 1, f'{err_rate:.0f}%', 
                     ha='center', va='bottom', fontsize=9, color='#c0392b', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stacked_error_analysis.png"), dpi=300)
    plt.close()

def train_svm_pro():
    dataset = load_dataset("SetFit/sst5")
    X_train = dataset['train']['text']
    y_train = dataset['train']['label']
    X_test = dataset['test']['text']
    y_test = dataset['test']['label']
    
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
    evaluate_and_save_results(pipeline, X_test, y_test, label_map)

if __name__ == "__main__":
    train_svm_pro()