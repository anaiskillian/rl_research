#!/bin/bash
#SBATCH --job-name=eval_math
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --partition=kempner_h100
#SBATCH --output=%x_%j/output_%j.out
#SBATCH --error=%x_%j/error_%j.out
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --mem=256000
#SBATCH --exclusive

# Load necessary modules
module load python/3.10.12-fasrc01
module load intelpython/3.9.16-fasrc01
module load cuda
module load cudnn

# Activate your conda environment
mamba activate demo

# Run your Python script
python eval_math_batched_v2.py
