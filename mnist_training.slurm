#!/bin/bash
#SBATCH --job-name=mnist_conventional_training
#SBATCH --mail-user="dp.wuensch@gmail.com"
#SBATCH --mail-type=ALL
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-short               # adjust if ALICE uses gpu_short / gpu_long
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --gres=gpu         # adapt to your available GPU type

# start with a clean module environment OPTIONAL
# module purge

#  load modules you need for CUDA & conda
module load ALICE/default
module load slurm
module load CUDA/12.3.2
module load GCC/11.3.0
module load Miniconda3/24.7.1-0         # check 'module avail' for correct version

# make sure autoverify looks at correct paths
export CONDA_ENVS_PATH=/home/s3665534/.conda/envs
export CONDA_PREFIX=/home/s3665534/.conda
export CUDA_HOME=/easybuild/software/CUDA/12.3.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/home/s3665534/.conda/envs/__av__abcrown/lib:$LD_LIBRARY_PATH

# move to project folder
cd $HOME/adversarial-training-box-FORK
TEST_DIR=$(pwd)
echo "## Current dircectory $TEST_DIR"

# activate conda env
eval "$(conda shell.bash hook)"
conda activate adversarial-training-box

# run your training script
python example_scripts/mnist-net_256x2-standard-training.py

echo "## Job finished. Goodbye"
