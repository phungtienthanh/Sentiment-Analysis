

print("--- Bắt đầu huấn luyện mô hình Baseline (CountVectorizer) ---")

import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# 1. Tải dữ liệu
print("Đang tải dữ liệu SST-5...")
dataset = load_dataset("SetFit/sst5")
X_train_raw = dataset['train']['text'] + dataset['validation']['text']
y_train = dataset['train']['label'] + dataset['validation']['label']
X_test_raw = dataset['test']['text']
y_test = dataset['test']['label']

# 2. Vector hóa văn bản bằng CountVectorizer
print("Vector hóa văn bản bằng CountVectorizer...")
vectorizer = CountVectorizer(stop_words='english', max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train_raw)
X_test_vec = vectorizer.transform(X_test_raw)

# 3. Huấn luyện mô hình Naive Bayes
print("Huấn luyện mô hình Naive Bayes...")
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 4. Đánh giá
print("Đánh giá mô hình...")
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['very negative', 'negative', 'neutral', 'positive', 'very positive'])

print(f"\nĐộ chính xác (Accuracy): {accuracy:.4f}")
print("Báo cáo phân loại:\n", report)

# 5. Lưu kết quả và model
print("Lưu model và báo cáo...")
joblib.dump(model, 'countvec_model.joblib')
joblib.dump(vectorizer, 'countvec_vectorizer.joblib')

with open('report_countvec.txt', 'w', encoding='utf-8') as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(report)

print("--- Hoàn tất huấn luyện mô hình Baseline ---")