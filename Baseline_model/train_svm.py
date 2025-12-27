import joblib
import re
import string
import os
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.exceptions import ConvergenceWarning
import nltk
from nltk.tokenize import word_tokenize
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

def custom_preprocessor(text):
    if not isinstance(text, str): return ""
    text = text.lower().strip() 
    text = re.sub(r'\s+', ' ', text) 
    return text

def custom_tokenizer(text):
    tokens = word_tokenize(text)
    filter_punct = string.punctuation.replace('!', '').replace('?', '')  
    return [t for t in tokens if t not in filter_punct]

def calculate_approx_loss(model, X, y):
    from sklearn.metrics import hinge_loss
    decision = model.decision_function(X)
    return hinge_loss(y, decision)

def train_svm_exact_curve():
    plot_dir = "results_SVM"
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)

    dataset = load_dataset("SetFit/sst5")
    X_train_raw = dataset['train']['text']
    y_train = dataset['train']['label']
    X_val_raw = dataset['validation']['text']
    y_val = dataset['validation']['label']
    
    label_map = {0: 'Very Negative', 1: 'Negative', 2: 'Neutral', 3: 'Positive', 4: 'Very Positive'}

    tfidf = TfidfVectorizer(
        preprocessor=custom_preprocessor,
        tokenizer=custom_tokenizer, 
        ngram_range=(1, 3),
        min_df=2, 
        max_features=25000 
    )
    
    X_train_vec = tfidf.fit_transform(X_train_raw)
    X_val_vec = tfidf.transform(X_val_raw)

    epochs = 40
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for i in range(1, epochs + 1):
        clf = LinearSVC(
            class_weight='balanced', 
            random_state=42, 
            C=0.5, 
            max_iter=i,
            loss='squared_hinge',
            dual=True
        )
        
        clf.fit(X_train_vec, y_train)
        
        train_acc = clf.score(X_train_vec, y_train)
        val_acc = clf.score(X_val_vec, y_val)
        
        train_loss = calculate_approx_loss(clf, X_train_vec, y_train)
        val_loss = calculate_approx_loss(clf, X_val_vec, y_val)
        
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

    epochs_range = range(1, epochs + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history['train_acc'], 'b-o', label='Training Accuracy')
    plt.plot(epochs_range, history['val_acc'], 'r-o', label='Validation Accuracy')
    plt.title('LinearSVC Learning Curve (Accuracy per Iteration)')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(plot_dir, "svm_accuracy_curve.png"))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history['train_loss'], 'b-o', label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], 'r-o', label='Validation Loss')
    plt.title('LinearSVC Learning Curve (Loss per Iteration)')
    plt.xlabel('Iterations')
    plt.ylabel('Hinge Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(plot_dir, "svm_loss_curve.png"))
    plt.close()
    
    final_svm = LinearSVC(
        class_weight='balanced', 
        random_state=42, 
        C=0.5, 
        loss='squared_hinge', 
        max_iter=5000,
        dual=True
    )
    
    final_svm.fit(X_train_vec, y_train)
    
    full_pipeline = Pipeline([('vectorizer', tfidf), ('svm', final_svm)])
    joblib.dump(full_pipeline, 'sst5_svm_pro.joblib')
    
    y_pred = final_svm.predict(X_val_vec)
    print(classification_report(y_val, y_pred, target_names=label_map.values()))

if __name__ == "__main__":
    train_svm_exact_curve()