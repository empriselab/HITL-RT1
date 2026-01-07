#!/bin/bash
#SBATCH -J bbox_inference
#SBATCH -o slurm_outputs/%j.out
#SBATCH -e slurm_outputs/%j.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=98304
#SBATCH -t 24:00:00
#SBATCH --partition=bhattacharjee
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-mid"
#SBATCH --requeue

# Load Anaconda properly (so `conda` works)
source /share/apps/software/anaconda3/etc/profile.d/conda.sh

# Activate your new environment
conda activate tf11-noembed

# Set up CUDA environment for G2 cluster
export CUDA_HOME=/usr/local/cuda-11.2
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0


# API URL - UPDATE THIS with your API server URL
API_URL="https://b66c5e1f834a.ngrok-free.app"

# Print some useful information
echo "=== Batch API Inference Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "Python executable: $(which python)"
echo "API URL: $API_URL"
echo "=========================="

# Install required packages if needed
pip install requests matplotlib numpy --quiet

# Create output directory
mkdir -p results

# Run batch inference
echo "Starting batch API inference..."
python batch_api_inference.py \
    --api_url "$API_URL" \
    --input_dir inference/inference_images \
    --output_file results/batch_inference_results

if [ $? -eq 0 ]; then
    echo "✅ Batch inference completed successfully!"
else
    echo "❌ Batch inference failed!"
    exit 1
fi

echo "Job completed at $(date)" 