# File: predict.py

import joblib

MODEL_TYPE = 'countvec' 


print(f"--- Tải mô hình {MODEL_TYPE.upper()} ---")
try:
    model = joblib.load(f'{MODEL_TYPE}_model.joblib')
    vectorizer = joblib.load(f'{MODEL_TYPE}_vectorizer.joblib')
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file model của {MODEL_TYPE}. Hãy chạy file train_{MODEL_TYPE}.py trước.")
    exit()


new_reviews = [
    "This movie was absolutely fantastic! A masterpiece.",
    "It was a complete waste of my time.",
    "The acting was okay, but the plot was predictable.",
    "A true masterpiece of cinema... if your goal is to fall asleep in ten minutes." # Câu mỉa mai
]

target_names = ['very negative', 'negative', 'neutral', 'positive', 'very positive']


new_reviews_vec = vectorizer.transform(new_reviews)
predictions = model.predict(new_reviews_vec)

print("\n--- Kết quả dự đoán ---")
for review, prediction in zip(new_reviews, predictions):
    print(f"Bình luận: '{review}'")
    print(f"  -> Dự đoán: {target_names[prediction].upper()}\n")