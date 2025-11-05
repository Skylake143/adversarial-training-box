#!/bin/bash
#SBATCH --job-name=mnist_training_testing
#SBATCH --mail-user="dp.wuensch@gmail.com"
#SBATCH --mail-type=ALL
#SBATCH --time=00:05:00
#SBATCH --partition=testing
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12
#SBATCH --gres=gpu

#  load modules you need for CUDA & conda
module load ALICE/default
module load CUDA/12.3.2
module load GCC/11.3.0
module load Miniconda3/24.7.1-0         # check 'module avail' for correct version

# make sure autoverify looks at correct paths
export CONDA_ENVS_PATH=/home/s3665534/.conda/envs
export CONDA_PREFIX=/home/s3665534/.conda
export CUDA_HOME=/easybuild/software/CUDA/12.3.2
export PATH=$CUDA_HOME/bin:$PATH

# move to project folder
cd $HOME/adversarial-training-box-FORK
TEST_DIR=$(pwd)
echo "## Current dircectory $TEST_DIR"

# activate conda env
eval "$(conda shell.bash hook)"
conda activate adversarial-training-box

# run your training script
python example_scripts/emnist-pgd-training.py --network CNN_SMALL --experiment_name cnn_small-pgd-training

echo "## Job finished. Goodbye"