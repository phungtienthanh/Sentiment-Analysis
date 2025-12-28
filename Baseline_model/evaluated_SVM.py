import joblib
import re
import string
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.tokenize import word_tokenize

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
    clean_tokens = []
    for t in tokens:
        if t not in filter_punct:
            clean_tokens.append(t)
    return clean_tokens

def evaluate_and_save_results(pipeline, X_data, y_data, label_map, dataset_name="test"):
    output_dir = "results_SVM"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    display_name = dataset_name.upper()
    print(f"Evaluating on {display_name} set...")

    y_pred = pipeline.predict(X_data)
    labels = list(label_map.keys())
    target_names = list(label_map.values())

    report = classification_report(y_data, y_pred, target_names=target_names)
    
    with open(os.path.join(output_dir, f"{dataset_name}_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_data, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix - {display_name} Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_confusion_matrix.png"))
    plt.close()

    all_preds = np.array(y_pred)
    all_labels = np.array(y_data)
    
    all_lengths = np.array([len(custom_tokenizer(custom_preprocessor(t))) for t in X_data])
    
    max_len = np.max(all_lengths) if len(all_lengths) > 0 else 0
    if max_len == 0: max_len = 10 
    
    bins = np.linspace(0, max_len, 16) 
    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = (bins[1] - bins[0]) * 0.8

    correct_counts = []
    incorrect_counts = []
    
    for i in range(len(bins)-1):
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

    plt.figure(figsize=(14, 8), facecolor='white')
    
    color_correct = '#27ae60' 
    color_incorrect = '#e74c3c' 
    
    plt.bar(bin_centers, correct_counts, width=width, 
            label='Correct Predictions', color=color_correct, 
            alpha=0.85, edgecolor='white', linewidth=0.5)
    
    plt.bar(bin_centers, incorrect_counts, width=width, 
            bottom=correct_counts, label='Misclassified', 
            color=color_incorrect, alpha=0.9, 
            edgecolor='white', linewidth=0.5)

    plt.title(f'Accuracy vs. Sentence Length (SVM - {display_name})', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Sentence Length (Number of words)', fontsize=13)
    plt.ylabel('Number of Sentences', fontsize=13)
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(frameon=True, shadow=True, fontsize=11, loc='upper right')

    for i in range(len(bin_centers)):
        total = correct_counts[i] + incorrect_counts[i]
        if total > 5: 
            err_rate = (incorrect_counts[i] / total) * 100
            plt.text(bin_centers[i], total + 1, f'{err_rate:.0f}%', 
                     ha='center', va='bottom', fontsize=9, color='#c0392b', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_stacked_error_analysis.png"), dpi=300)
    plt.close()

def run_evaluation_full():
    print("Loading SST-5 dataset...")
    dataset = load_dataset("SetFit/sst5")
    
    X_test = dataset['test']['text']
    y_test = dataset['test']['label']
    
    X_dev = dataset['validation']['text']
    y_dev = dataset['validation']['label']
    
    label_map = {0: 'Very Negative', 1: 'Negative', 2: 'Neutral', 3: 'Positive', 4: 'Very Positive'}

    model_path = 'sst5_svm_pro.joblib' 
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    print(f"Loading model from '{model_path}'...")
    try:
        pipeline = joblib.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    evaluate_and_save_results(pipeline, X_dev, y_dev, label_map, dataset_name="dev")
    
    evaluate_and_save_results(pipeline, X_test, y_test, label_map, dataset_name="test")
    
    print(f"Done! Results saved in 'results_SVM' directory.")

if __name__ == "__main__":
    run_evaluation_full()