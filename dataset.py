"""
data_preparation.py
-------------------
Module dùng để:
1️⃣ Làm sạch văn bản (HTML, emoji, stopwords, lemmatize)
2️⃣ Tải và lưu tokenizer của BERT (bert-base-uncased)
3️⃣ Chuyển text thành input tensor (padding, truncation)
4️⃣ Chuẩn bị embedding matrix từ GloVe cho LSTM
"""

import re
import string
import os
import nltk
import numpy as np
from bs4 import BeautifulSoup
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer

# --------------------------------------------------
# 1️⃣ Cấu hình và tải tài nguyên NLP
# --------------------------------------------------
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --------------------------------------------------
# 2️⃣ Hàm làm sạch văn bản
# --------------------------------------------------
def clean_text(text: str) -> str:
    """Làm sạch văn bản: xóa HTML, emoji, URL, stopword, lemmatize."""
    if not isinstance(text, str):
        return ""

    # 1. Xóa HTML
    text = BeautifulSoup(text, "html.parser").get_text()

    # 2. Chuẩn hóa contractions (don’t → do not)
    text = contractions.fix(text)

    # 3. Xóa emoji
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # 4. Xóa URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 5. Xóa dấu câu, chuyển lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()

    # 6. Lemmatize + loại stopword + giữ từ alphabet
    tokens = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word.isalpha() and word not in stop_words
    ]

    return " ".join(tokens)

# --------------------------------------------------
# 3️⃣ Tokenizer BERT (tải và lưu local)
# --------------------------------------------------
def load_bert_tokenizer(save_dir: str = "./tokenizer"):
    """
    Tải tokenizer của BERT và lưu local.
    Nếu đã có local tokenizer, tự động load lại.
    """
    if not os.path.exists(save_dir):
        print(f"[INFO] Downloading BERT tokenizer → {save_dir}")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenizer.save_pretrained(save_dir)
    else:
        print(f"[INFO] Loading local BERT tokenizer from {save_dir}")
        tokenizer = BertTokenizer.from_pretrained(save_dir)
    return tokenizer

# --------------------------------------------------
# 4️⃣ Hàm encode văn bản
# --------------------------------------------------
def encode_texts(texts, tokenizer, max_len: int = 150, framework: str = 'pt'):
    """
    Biến danh sách văn bản thành tensor input.
    framework: 'tf' hoặc 'pt' (TensorFlow hoặc PyTorch)
    """
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors=framework
    )

# --------------------------------------------------
# 5️⃣ Chuẩn bị embedding matrix (nếu dùng GloVe)
# --------------------------------------------------
def load_glove_embeddings(glove_path: str, tokenizer_vocab: dict, embedding_dim: int = 100):
    """
    Tạo embedding matrix từ file GloVe và vocab của tokenizer.
    glove_path: đường dẫn đến glove.6B.100d.txt
    tokenizer_vocab: tokenizer.vocab (từ → id)
    """
    print(f"[INFO] Loading GloVe embeddings from {glove_path} ...")
    embedding_index = {}
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs

    embedding_matrix = np.zeros((len(tokenizer_vocab), embedding_dim))
    for word, i in tokenizer_vocab.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    print(f"[INFO] Embedding matrix created: {embedding_matrix.shape}")
    return embedding_matrix

# --------------------------------------------------
# 6️⃣ Nếu chạy riêng file này
# --------------------------------------------------
if __name__ == "__main__":
    import pandas as pd

    # Ví dụ test nhỏ
    data = {
        "review": [
            "I loved the movie! It's absolutely fantastic <br> 😍",
            "Worst film ever... boring and too long! http://example.com",
        ],
        "sentiment": ["positive", "negative"]
    }
    df = pd.DataFrame(data)

    print("🔹 Cleaning text ...")
    df['clean_review'] = df['review'].apply(clean_text)
    print(df[['review', 'clean_review']])

    print("\n🔹 Loading tokenizer ...")
    tokenizer = load_bert_tokenizer()

    print("\n🔹 Encoding text ...")
    encoded = encode_texts(df['clean_review'].tolist(), tokenizer)
    print("Input IDs shape:", encoded['input_ids'].shape)

    # Nếu muốn tạo embedding từ GloVe (khi dùng LSTM)
    # glove_path = './embeddings/glove.6B.100d.txt'
    # embedding_matrix = load_glove_embeddings(glove_path, tokenizer.vocab)
