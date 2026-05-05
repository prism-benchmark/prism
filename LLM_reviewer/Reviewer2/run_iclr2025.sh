#!/bin/bash
#SBATCH --job-name=rv_icl2025
#SBATCH --output=/datastore/npl/luannt/IHSD/Reviewer2/logs/reviewer2_iclr2025.out
#SBATCH --error=/datastore/npl/luannt/IHSD/Reviewer2/logs/reviewer2_iclr2025.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=mps:l40:1 
#SBATCH --mem=4G
#SBATCH --time=72:00:00

# =========================================================
# VRAM CONFIGURATION
# =========================================================
# vLLM + Qwen3-14B: ~20-24GB VRAM
# Transformers + Qwen3-14B: ~32GB VRAM
# 
# Hiện tại script sử dụng vLLM (--use_vllm flag)
# Nên REQUIRED_VRAM có thể giảm xuống
REQUIRED_VRAM=37000  # Đồng bộ với gpu_memory_utilization=0.45 để tránh chọn GPU chỉ còn ~30GB rồi fail ngay

# =========================================================
# CHUẨN BỊ MÔI TRƯỜNG
# =========================================================
module clear -f

module load shared python312
source /datastore/npl/luannt/IHSD/.cache/venv/bin/activate
export PATH="/datastore/npl/luannt/IHSD/.cache/venv/bin:$PATH"
export PYTHONPATH="/datastore/npl/luannt/IHSD/.cache/venv/lib/python3.12/site-packages:$PYTHONPATH"
cd /datastore/npl/luannt/IHSD/Reviewer2

# Xóa biến môi trường Slurm để tự chọn GPU
unset CUDA_VISIBLE_DEVICES
# --- GỌI HELPER --- (Quan trọng, cần gọi hàm này (có sẵn) để tìm GPU có vRAM trống >= REQUIRED_VRAM, nếu không tìn thấy GPU đủ vRAM thì hàm CHECK_OUT sẽ đưa job vào lại hàng đợi để chờ tìm slot khác; sau 5 lần requeue mà vẫn chưa tìm được slot thì sẽ trả về mã lỗi để kết thúc job)
CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
EXIT_CODE=$?
if [ $EXIT_CODE -eq 10 ]; then
    echo "$CHECK_OUT"
    exit 0 # Thoát để Slurm đưa vào queue lại
elif [ $EXIT_CODE -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1 # Lỗi thật sự (sau 5 lần requeue), dừng Job
fi
BEST_GPU=$CHECK_OUT
echo "✅ Job $SLURM_JOB_ID bắt đầu trên GPU: $BEST_GPU"
# =========================================================
# KHỞI TẠO PRIVATE MPS SERVER
# =========================================================
# Khởi tạo Private MPS với Pipe riêng cho từng JOB ID (để dễ quit)
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID

# Xóa sạch dấu vết cũ nếu có và tạo mới (quan trọng!)
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

export CUDA_VISIBLE_DEVICES=$BEST_GPU # Quan trọng

# =========================================================
# CHẠY CODE với vLLM
# =========================================================
# vLLM settings:
# - --use_vllm: Sử dụng vLLM backend (10-15x nhanh hơn Transformers)
# - --batch_size 4: Batch 4 papers tại 1 lần inference
# - --paper_ids: Filter để chỉ xử lý papers có ID trong file (optional)
# - --skip_existing: Bỏ qua papers đã xử lý rồi (resume mode)
#
# Expected performance (for ~3000 papers in subset):
# - vLLM (batch_size=4): 5-15s/paper → ~4-12 giờ cho 3000 papers subset
# 
# CRITICAL: Set multiprocessing method for vLLM CUDA compatibility
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python /datastore/npl/luannt/IHSD/Reviewer2/demo_iclr2025_vllm.py \
    --grobid_dir /datastore/npl/luannt/IHSD/Reviewer2/ICLR2025/grobid_fulltext \
    --output_dir /datastore/npl/luannt/IHSD/Reviewer2/output_reviewer2_iclr2025_fix_empty_review \
    --batch_size 8 \
    --gpu_memory_utilization 0.8 \
    --max_model_len 38000 \
    --max_num_batched_tokens 2048 \
    --paper_ids /datastore/npl/luannt/IHSD/Reviewer2/ICLR2025/data_subset/paper_ids_200.txt \
    --force_reprocess