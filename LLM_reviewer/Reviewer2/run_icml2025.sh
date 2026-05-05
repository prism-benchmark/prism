#!/bin/bash
#SBATCH --job-name=rv_icml2025
#SBATCH --output=/datastore/npl/luannt/IHSD/Reviewer2/logs/reviewer2_icml2025.out
#SBATCH --error=/datastore/npl/luannt/IHSD/Reviewer2/logs/reviewer2_icml2025.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=mps:a100:1
#SBATCH --mem=4G
#SBATCH --time=72:00:00

# =========================================================
# VRAM CONFIGURATION
# =========================================================
REQUIRED_VRAM=35000

# =========================================================
# CHUẨN BỊ MÔI TRƯỜNG
# =========================================================
module clear -f

module load shared python312
source /datastore/npl/luannt/IHSD/.cache/venv/bin/activate
export PATH="/datastore/npl/luannt/IHSD/.cache/venv/bin:$PATH"
export PYTHONPATH="/datastore/npl/luannt/IHSD/.cache/venv/lib/python3.12/site-packages:$PYTHONPATH"
cd /datastore/npl/luannt/IHSD/Reviewer2

unset CUDA_VISIBLE_DEVICES
CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
EXIT_CODE=$?
if [ $EXIT_CODE -eq 10 ]; then
    echo "$CHECK_OUT"
    exit 0
elif [ $EXIT_CODE -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1
fi
BEST_GPU=$CHECK_OUT
echo "✅ Job $SLURM_JOB_ID bắt đầu trên GPU: $BEST_GPU"

# =========================================================
# KHỞI TẠO PRIVATE MPS SERVER
# =========================================================
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID

rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY

export CUDA_VISIBLE_DEVICES=$BEST_GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =========================================================
# CHẠY CODE với vLLM
# =========================================================
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python /datastore/npl/luannt/IHSD/Reviewer2/demo_icml2025_vllm.py \
    --grobid_dir /datastore/npl/luannt/IHSD/Reviewer2/ICML2025/grobid_fulltext \
    --output_dir /datastore/npl/luannt/IHSD/Reviewer2/output_reviewer2_icml2025 \
    --paper_ids /datastore/npl/luannt/IHSD/Reviewer2/ICML2025/data_subset/paper_ids_200.txt \
    --batch_size 8 \
    --gpu_memory_utilization 0.48 \
    --max_model_len 36864 \
    --max_num_batched_tokens 1024 \
    --max_num_seqs 32