import joblib
import os
import re
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report


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
    text = re.sub(r'\s+', ' ', text)
    return text

def custom_tokenizer(text):
    tokens = word_tokenize(text)
    filter_punct = string.punctuation.replace('!', '').replace('?', '')
    return [t for t in tokens if t not in filter_punct]


model_files = {
    "Logistic Regression": "sst5_logistic_regression_pro.joblib",
    "SVM (Linear)":        "sst5_svm_pro.joblib",
    "Random Forest":       "sst5_random_forest.joblib",
    "XGBoost":             "sst5_xgboost.joblib",
    "k-NN":                "sst5_knn.joblib",
    "Naive Bayes (CountVec)": "countvec_model.joblib"
}


OUTPUT_FOLDER = "comparison_val"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


def main():
    dataset = load_dataset("SetFit/sst5")
    
    X_val = dataset['validation']['text']
    y_val = dataset['validation']['label']
    
    label_map = {0: 'Very Negative', 1: 'Negative', 2: 'Neutral', 3: 'Positive', 4: 'Very Positive'}
    results = []
    
    
    for model_name, filename in model_files.items():
        print(f"🔹 Đang xử lý: {model_name}...")
        

        if not os.path.exists(filename):
            print(f"   ⚠️ Lỗi: Không tìm thấy file '{filename}'. Bỏ qua.")
            continue

        try:
            y_pred = []
            
            if model_name == "Naive Bayes (CountVec)":
                vec_filename = "countvec_vectorizer.joblib"
                if not os.path.exists(vec_filename):
                    print(f"   ⚠️ Thiếu file vectorizer '{vec_filename}'.")
                    continue
                
                print("   -> Loading Vectorizer & Model rời...")
                vectorizer = joblib.load(vec_filename)
                model = joblib.load(filename)
                

                X_val_vec = vectorizer.transform(X_val)
                y_pred = model.predict(X_val_vec)
                

            else:
                pipeline = joblib.load(filename)
                y_pred = pipeline.predict(X_val)

   
            acc = accuracy_score(y_val, y_pred)
            f1_macro = f1_score(y_val, y_pred, average='macro')
            f1_weighted = f1_score(y_val, y_pred, average='weighted')
            
         
            report = classification_report(y_val, y_pred, target_names=label_map.values())
            report_path = os.path.join(OUTPUT_FOLDER, f"report_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.txt")
            with open(report_path, "w", encoding='utf-8') as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Validation Accuracy: {acc:.4f}\n")
                f.write("-" * 30 + "\n")
                f.write(report)
            
            results.append({
                "Model": model_name,
                "Accuracy": acc,
                "Macro F1": f1_macro,
                "Weighted F1": f1_weighted
            })
            print(f"   ✅ Done. Acc: {acc:.4f} | Macro F1: {f1_macro:.4f}")
            
        except Exception as e:
            print(f"   ❌ Lỗi ngoại lệ: {str(e)}")


    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by="Macro F1", ascending=False)
        
        print("\n" + "="*50)
        print("🏆 BẢNG XẾP HẠNG MODEL (Trên tập VALIDATION)")
        print("="*50)
        print(df_results.to_string(index=False))
        
        # Lưu CSV
        csv_path = os.path.join(OUTPUT_FOLDER, "val_comparison.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"\n💾 Đã lưu bảng so sánh: {csv_path}")
        
        # VẼ BIỂU ĐỒ
        plt.figure(figsize=(12, 6))
        sns.set_style("whitegrid")
        
        df_melted = df_results.melt(id_vars="Model", value_vars=["Accuracy", "Macro F1"], var_name="Metric", value_name="Score")
        
        chart = sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette="viridis")
        plt.title("So sánh hiệu suất các mô hình trên tập VALIDATION SST-5")
        plt.ylim(0, 0.7) 
        plt.xticks(rotation=15)
        plt.legend(loc='lower right')
        
        for container in chart.containers:
            chart.bar_label(container, fmt='%.3f', padding=3, fontsize=9)
            
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_FOLDER, "val_benchmark_chart.png")
        plt.savefig(plot_path)
        print(f"📊 Đã lưu biểu đồ: {plot_path}")
    else:
        print("\n⚠️ Không có kết quả nào được ghi nhận.")

if __name__ == "__main__":
    main()