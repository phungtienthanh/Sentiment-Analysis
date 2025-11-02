#!/bin/bash

#----------------------------------------------------------------#
# Script để chạy training mô hình SST-5
#----------------------------------------------------------------#

# Dừng script ngay lập tức nếu có bất kỳ lệnh nào thất bại
set -e

# (TÙY CHỌN) Chỉ định GPU nào sẽ được sử dụng
export CUDA_VISIBLE_DEVICES=0

echo "============================================="
echo "STARTING TRAINING..."
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================="

# 1. Tạo thư mục 'logs' nếu nó chưa tồn tại
# Cờ '-p' đảm bảo nó không báo lỗi nếu thư mục đã tồn tại
mkdir -p logs

# 2. Lấy ngày giờ và trỏ LOG_FILE vào thư mục 'logs'
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_log_${TIMESTAMP}.log" 

# Kích hoạt môi trường ảo (ví dụ: .venv)
# (Hãy đảm bảo đường dẫn này đúng)
source .venv/bin/activate
echo "Virtual environment activated."

# Lệnh 'tee' sẽ vừa in ra màn hình, vừa ghi vào file log
python src/train.py | tee $LOG_FILE

echo "============================================="
echo "TRAINING FINISHED."
# Thông báo này sẽ tự động cập nhật vì $LOG_FILE đã có đường dẫn mới
echo "Log file saved to: $LOG_FILE"
echo "============================================="