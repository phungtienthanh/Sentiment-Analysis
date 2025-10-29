#!/bin/bash

#----------------------------------------------------------------#
# Script để chạy training mô hình SST-5
#
# Cách dùng (trên Ubuntu/Linux):
# 1. Mở terminal, cd vào thư mục GỐC của project (SST5_LSTM_BERT/)
# 2. Chạy: chmod +x scripts/run_training.sh  (Chỉ cần chạy 1 lần)
# 3. Chạy: ./scripts/run_training.sh
#----------------------------------------------------------------#

# Dừng script ngay lập tức nếu có bất kỳ lệnh nào thất bại
set -e

# (TÙY CHỌN) Chỉ định GPU nào sẽ được sử dụng (ví dụ: GPU 0)
# Rất hữu ích nếu máy có nhiều GPU, với máy em chỉ có 1 GPU thì đây là 0
export CUDA_VISIBLE_DEVICES=0

echo "============================================="
echo "STARTING TRAINING..."
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================="

# Lấy ngày giờ hiện tại để đặt tên file log
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="training_log_${TIMESTAMP}.log"

# Chạy script train.py
# Chúng ta giả định file này được chạy từ thư mục GỐC của project
#
# Lệnh 'tee' rất hữu ích:
# 1. Nó sẽ in output ra terminal để em theo dõi (stdout)
# 2. Đồng thời, nó ghi toàn bộ output vào file $LOG_FILE
python src/train.py | tee $LOG_FILE

echo "============================================="
echo "TRAINING FINISHED."
echo "Log file saved to: $LOG_FILE"
echo "============================================="