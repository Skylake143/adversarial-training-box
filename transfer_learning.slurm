#!/bin/bash
#SBATCH --job-name=transfer_learning
#SBATCH --mail-user="dp.wuensch@gmail.com"
#SBATCH --mail-type=ALL
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-short  # gpu-short;gpu-2080ti-11g; gpu-mig-40g; gpu-a100-80g
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --gres=gpu:l4:1        # 4g.40gb:1; a100; l4

#  load modules you need for CUDA & conda
module load ALICE/default
module load CUDA/12.3.2
module load GCC/11.3.0
module load Miniconda3/24.7.1-0

# make sure autoverify looks at correct paths
export CONDA_ENVS_PATH=/home/s3665534/.conda/envs
export CONDA_PREFIX=/home/s3665534/.conda
export CUDA_HOME=/easybuild/software/CUDA/12.3.2
export PATH=$CUDA_HOME/bin:$PATH

# move to project folder
cd $HOME/adversarial-training-box-FORK
TEST_DIR=$(pwd)
echo "## Current dircectory $TEST_DIR"

echo "## GPU Information:"
nvidia-smi

# activate conda env
eval "$(conda shell.bash hook)"
conda activate adversarial-training-box

# run your training script
python example_scripts/transfer_learning_mnist_emnist.py

echo "## Job finished. Goodbye"