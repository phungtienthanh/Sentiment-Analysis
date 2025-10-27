# src/config.py

import torch

# --- 1. Cấu hình Dữ liệu & Mô hình ---
BERT_MODEL_NAME = 'bert-base-uncased'
DATASET_NAME = 'SetFit/sst5'

# Quyết định từ file 1.0-data-exploration.ipynb
MAX_LENGTH = 64 
NUM_CLASSES = 5

# --- 2. Cấu hình Kiến trúc ---
LSTM_HIDDEN_SIZE = 256 # Kích thước lớp ẩn LSTM
LSTM_LAYERS = 2      # Số lớp BiLSTM (stacked)
DROPOUT_RATE = 0.3

# --- 3. Cấu hình Training ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32      # Tăng batch size cho training (thay vì 8 để test)
NUM_EPOCHS = 10      # Số epoch để huấn luyện
LEARNING_RATE = 2e-5 # Learning rate phổ biến cho AdamW

# --- 4. Cấu hình Đường dẫn ---
MODEL_SAVE_PATH = "../saved_models/"