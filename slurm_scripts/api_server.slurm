#!/bin/bash
#SBATCH -J bbox_inference
#SBATCH -o slurm_outputs/%j.out
#SBATCH -e slurm_outputs/%j.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=98304
#SBATCH -t 120:00:00
#SBATCH --partition=bhattacharjee
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-mid"
#SBATCH --requeue

# Load Anaconda properly (so `conda` works)
source /share/apps/software/anaconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate tf11-noembed

# Set up CUDA environment for G2 cluster
export CUDA_HOME=/usr/local/cuda-11.2
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Set Flask environment variables
export FLASK_ENV=production
export PORT=8080
export CHECKPOINT_PATH=output_checkpoints27-smooth_l1
export CONFIG_FILE=configs/transformer_mixin.gin
export LOSS_TYPE=smooth_l1
export USE_NGROK=true
export NGROK_AUTHTOKEN=3317JggsR4HRTqlZk8l9XEPLEuk_6fr4S3BZN4kNn3J96pfTs

# Print some useful information
echo "=== RT-1 API Server Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "Python executable: $(which python)"
echo "CUDA device: $CUDA_VISIBLE_DEVICES"
echo "Port: $PORT"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "=========================="

# Install required packages
pip install flask flask-cors pyngrok

# Create output directory for logs
mkdir -p slurm_outputs

# Run the API server
echo "Starting RT-1 API server..."
python api_batch_server.py

if [ $? -eq 0 ]; then
    echo "✅ API server completed successfully!"
else
    echo "❌ API server failed!"
    exit 1
fi

echo "Job completed at $(date)"
