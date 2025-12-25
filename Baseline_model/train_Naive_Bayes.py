import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score


dataset = load_dataset("SetFit/sst5")

X_train_raw = dataset['train']['text']
y_train = dataset['train']['label']


X_val_raw = dataset['validation']['text']
y_val = dataset['validation']['label']


vectorizer = CountVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))

X_train_vec = vectorizer.fit_transform(X_train_raw)
X_val_vec = vectorizer.transform(X_val_raw)


model = MultinomialNB()
model.fit(X_train_vec, y_train)


y_pred = model.predict(X_val_vec)
acc = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy : {acc:.4f}")


joblib.dump(model, 'sst5_countvec_model.joblib')
joblib.dump(vectorizer, 'sst5_countvec_vectorizer.joblib')
