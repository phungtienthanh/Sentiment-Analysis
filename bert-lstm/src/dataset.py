# src/dataset.py

import torch
from torch.utils.data import Dataset

class SST5Dataset(Dataset):
    """
    Class Dataset, nhận dữ liệu từ HF Datasets và tokenizer, sau đó trả về các tensors đã xử lý
    """
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Lấy mẫu dữ liệu
        sample = self.data[idx]
        text = sample['text']
        label = sample['label']
        
        # Tokenize văn bản
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # .squeeze() để loại bỏ chiều batch (vì tokenizer trả về [1, max_length])
        # .to(torch.long) là cần thiết cho label
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }