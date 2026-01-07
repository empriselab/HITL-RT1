#!/bin/bash
#SBATCH -J inspect_checkpoint
#SBATCH -o slurm_outputs/%j.out
#SBATCH -e slurm_outputs/%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=36384
#SBATCH -t 01:00:00
#SBATCH --partition=bhattacharjee
#SBATCH --requeue

# Load Anaconda properly (so `conda` works)
source /share/apps/anaconda3/2022.10/etc/profile.d/conda.sh

# Activate your new environment
conda activate tf11-tfp19

# Print some useful information
echo "=== Checkpoint Inspection Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "Python executable: $(which python)"
echo "=========================="

# Check if h5py is available
python -c "import h5py; print('h5py version:', h5py.__version__)"

# Run the checkpoint inspection
echo "Starting checkpoint inspection..."
python inspect_h5_checkpoint.py

if [ $? -eq 0 ]; then
    echo "✅ Checkpoint inspection completed successfully!"
else
    echo "❌ Checkpoint inspection failed!"
    exit 1
fi

echo "Job completed at $(date)" 