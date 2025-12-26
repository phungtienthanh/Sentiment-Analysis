# src/utils.py

import torch
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def save_model(model, save_path, epoch, filename="best_model.pt"):
    """
    Lưu checkpoint của mô hình.
    """
    # Đảm bảo thư mục tồn tại
    os.makedirs(save_path, exist_ok=True)
    
    save_file = os.path.join(save_path, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, save_file)
    print(f"Model saved to {save_file}")

def calculate_metrics(labels, preds):
    """
    Tính toán các chỉ số đánh giá (Accuracy và F1-Score).
    """
    accuracy = accuracy_score(labels, preds)
    
    # Tính F1-Score (Macro) - Quan trọng cho dữ liệu mất cân bằng
    f1_macro = f1_score(labels, preds, average='macro')
    
    # Tính F1-Score (Weighted) - Cũng hữu ích
    f1_weighted = f1_score(labels, preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }