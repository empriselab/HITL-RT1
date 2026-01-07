#!/bin/bash
#SBATCH -J training_test
#SBATCH -o slurm_outputs/%j.out
#SBATCH -e slurm_outputs/%j.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=98304
#SBATCH -t 2:00:00
#SBATCH --partition=bhattacharjee
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-mid"
#SBATCH --requeue

# Load Anaconda properly (so `conda` works)
source /share/apps/anaconda3/2022.10/etc/profile.d/conda.sh

# Activate your environment
conda activate tf11-tfp19

# Set up CUDA environment for G2 cluster
export CUDA_HOME=/usr/local/cuda-11.2
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# Print versions for debugging
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import tensorflow_probability as tfp; print('TFP:', tfp.__version__)"
python -c "import tf_agents; print('TF-Agents:', tf_agents.__version__)"

# Test CUDA/CuDNN availability
python -c "import tensorflow as tf; print('CuDNN available:', tf.test.is_built_with_cuda()); print('GPU devices:', tf.config.list_physical_devices('GPU'))"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Print some useful information
echo "=== Training Mode Test Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "Python executable: $(which python)"
echo "CUDA device: $CUDA_VISIBLE_DEVICES"
echo "=================================="

# Run the training mode test
echo "Starting training mode test..."
python test_training_mode.py

if [ $? -eq 0 ]; then
    echo "✅ Training mode test completed successfully!"
else
    echo "❌ Training mode test failed!"
    exit 1
fi

echo "Job completed at $(date)" 