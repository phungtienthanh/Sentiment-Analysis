# src/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
from tqdm import tqdm # Thư viện để vẽ thanh tiến trình (progress bar)
import numpy as np
import os

# Import các file .py của chúng ta
import config
from dataset import SST5Dataset
from model import BertLSTMClassifier
from utils import save_model, calculate_metrics

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    """
    Hàm thực hiện một epoch training.
    """
    model.train() # Chuyển model sang chế độ training
    total_loss = 0
    
    # tqdm bọc data_loader để hiển thị progress bar
    for batch in tqdm(data_loader, desc="Training"):
        # Chuyển data lên device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # 1. Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 2. Tính loss
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()
        
        # 3. Backward pass và optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return total_loss / len(data_loader)

def evaluate(model, data_loader, loss_fn, device):
    """
    Hàm đánh giá mô hình trên tập validation.
    """
    model.eval() # Chuyển model sang chế độ evaluation
    total_loss = 0
    
    all_labels = []
    all_preds = []
    
    with torch.no_grad(): # Không tính gradient khi đánh giá
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 1. Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # 2. Tính loss
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            # Lấy dự đoán (lớp có xác suất cao nhất)
            _, preds = torch.max(outputs, dim=1)
            
            # Thu thập tất cả labels và preds để tính metrics
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    avg_loss = total_loss / len(data_loader)
    metrics = calculate_metrics(all_labels, all_preds)
    
    return avg_loss, metrics

# --- HÀM MAIN ĐỂ CHẠY ---
def main():
    print(f"Using device: {config.DEVICE}")

    # --- 1. Tải Dữ liệu và Tokenizer ---
    print("Loading tokenizer and dataset...")
    tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    raw_datasets = load_dataset(config.DATASET_NAME)

    # --- 2. Tạo Datasets (Train/Val) ---
    print("Creating datasets...")
    train_dataset = SST5Dataset(
        data=raw_datasets['train'],
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH
    )
    
    val_dataset = SST5Dataset(
        data=raw_datasets['validation'],
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH
    )
    
    # (Tùy chọn: test_dataset, nếu em muốn chạy đánh giá cuối cùng)
    
    # --- 3. Tạo DataLoaders ---
    print("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True, # Xáo trộn dữ liệu train
        num_workers=2 # (Tùy chọn) Tăng tốc độ load data
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False, # Không cần xáo trộn val/test
        num_workers=2
    )
    
    # --- 4. Khởi tạo Mô hình ---
    print("Initializing model...")
    model = BertLSTMClassifier(
        bert_model_name=config.BERT_MODEL_NAME,
        lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
        lstm_layers=config.LSTM_LAYERS,
        dropout_rate=config.DROPOUT_RATE,
        num_classes=config.NUM_CLASSES
    ).to(config.DEVICE)
    
    # sử dụng DataParallel trên nhiều GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # --- 5. Khởi tạo Loss Function và Optimizer ---
    print("Initializing loss function and optimizer...")
    # CrossEntropyLoss đã bao gồm Softmax
    loss_fn = nn.CrossEntropyLoss()
    
    # AdamW là optimizer tốt nhất cho các mô hình Transformer
    # Chỉ optimize các tham số *không* bị đóng băng (của LSTM, Attention, Classifier)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE
    )
    
    # --- 6. Check CUDA ---
    if torch.cuda.is_available():
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}") # For device 0
    
    # --- 7. Vòng lặp Training ---
    print("--- Starting Training ---")
    best_val_f1 = -1.0 # Lưu lại F1 macro tốt nhất

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, config.DEVICE)
        print(f"  Train Loss: {train_loss:.4f}")
        
        # Evaluate
        val_loss, val_metrics = evaluate(model, val_loader, loss_fn, config.DEVICE)
        
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Val F1 (Macro): {val_metrics['f1_macro']:.4f}")
        
        # Lưu lại model tốt nhất (dựa trên F1 Macro)
        current_f1 = val_metrics['f1_macro']
        if current_f1 > best_val_f1:
            print(f"  New best F1 Macro! ({current_f1:.4f} > {best_val_f1:.4f}). Saving model...")
            best_val_f1 = current_f1
            save_model(
                model=model,
                save_path=config.MODEL_SAVE_PATH,
                epoch=epoch,
                filename="best_model.pt" # Sẽ ghi đè model tốt nhất
            )

    print("--- Training Finished ---")
    print(f"Best validation F1 (Macro): {best_val_f1:.4f}")

# Đoạn này đảm bảo hàm main() chỉ chạy khi em thực thi file này
if __name__ == "__main__":
    # Tạo thư mục lưu model nếu chưa có
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    main()