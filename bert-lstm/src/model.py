# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class BertLSTMClassifier(nn.Module):
    """
    Kiến trúc mô hình chính:
    1. BERT (đóng băng) làm bộ trích xuất đặc trưng.
    2. Stacked BiLSTM để học ngữ cảnh tuần tự.
    3. Lớp Attention để tổng hợp thông tin.
    4. Lớp Classifier (Linear) để phân loại.
    """
    # __init__ sẽ nhận các tham số từ file config
    def __init__(self, bert_model_name, lstm_hidden_size, lstm_layers, dropout_rate, num_classes):
        
        super(BertLSTMClassifier, self).__init__()
        
        # --- 1. Lớp BERT (Feature Extractor) ---
        self.bert = BertModel.from_pretrained(bert_model_name)
        # **ĐÓNG BĂNG BERT**
        for param in self.bert.parameters():
            param.requires_grad = False
            
        bert_output_size = self.bert.config.hidden_size # (Đây là 768)
        
        # --- 2. Lớp Stacked BiLSTM (Encoder) ---
        self.lstm = nn.LSTM(
            input_size=bert_output_size,        # 768
            hidden_size=lstm_hidden_size, # 256
            num_layers=lstm_layers,       # 2
            bidirectional=True,               # BiLSTM
            batch_first=True,                 # [Batch, Seq, Feature]
            dropout=dropout_rate if lstm_layers > 1 else 0
        )
        
        lstm_output_size = lstm_hidden_size * 2 # (256 * 2 = 512)
        
        # --- 3. Lớp Attention ---
        self.attention_weights = nn.Linear(lstm_output_size, 1)
        
        # --- 4. Lớp Phân loại (Classifier) ---
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(
            lstm_output_size, # 512
            num_classes       # 5
        )

    def forward(self, input_ids, attention_mask):
        # 1. BERT: [B, S] -> [B, S, 768]
        bert_output = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        bert_features = bert_output.last_hidden_state
        
        # 2. BiLSTM: [B, S, 768] -> [B, S, 512]
        # Bỏ qua (h_n, c_n) vì chúng ta dùng attention
        lstm_output, _ = self.lstm(bert_features)
        
        # 3. Attention: [B, S, 512] -> [B, 512]
        # (B, S, 512) -> (B, S, 1)
        attn_logits = self.attention_weights(lstm_output)
        
        # Mask các vị trí [PAD]
        attn_mask = attention_mask.unsqueeze(2) # [B, S] -> [B, S, 1]
        attn_logits = attn_logits.masked_fill(attn_mask == 0, -1e9)
        
        # Softmax để ra trọng số
        attn_scores = F.softmax(attn_logits, dim=1) # [B, S, 1]
        
        # Tính vector ngữ cảnh (weighted sum)
        # (B, S, 512) * (B, S, 1) -> (B, S, 512)
        # .sum(dim=1) -> (B, 512)
        context_vector = torch.sum(lstm_output * attn_scores, dim=1)
        
        # 4. Classifier: [B, 512] -> [B, 5]
        context_vector = self.dropout(context_vector)
        logits = self.classifier(context_vector)
        
        return logits