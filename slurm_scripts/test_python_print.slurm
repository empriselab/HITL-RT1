#!/bin/bash
#SBATCH -J python_print_test
#SBATCH -o slurm_outputs/%j_printtest.out
#SBATCH -e slurm_outputs/%j_printtest.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=1024
#SBATCH -t 00:05:00
#SBATCH --partition=bhattacharjee

# Load Anaconda properly
source /share/apps/anaconda3/2022.10/etc/profile.d/conda.sh
conda activate tf11-tfp19

# Print job info
echo "=== Python Print Test Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "Python executable: $(which python)"
echo "============================="

# Run the minimal Python print test
python slurm_scripts/test_python_print.py

if [ $? -eq 0 ]; then
    echo "✅ Python print test completed successfully!"
else
    echo "❌ Python print test failed!"
    exit 1
fi

echo "Job completed at $(date)" 