#!/usr/bin/env python3
"""
Minimal SLURM Job Submitter
A simple script to automate SLURM batch job submissions with different parameters.
"""

import subprocess
import time
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class SlurmJobConfig:
    """Configuration for a SLURM job"""
    job_name: str
    script_path: str
    partition: str = "gpu-short" # gpu-short;gpu-2080ti-11g; gpu-mig-40g; gpu-a100-80g
    time_limit: str = "01:00:00"
    memory: str = "16G"
    cpus_per_task: int = 1
    gpus: int = 1
    mail_user: str = "dp.wuensch@gmail.com"
    additional_params: Dict[str, Any] = None


class SlurmJobSubmitter:
    """Handles automated SLURM job submission"""
    
    def __init__(self, workspace_path: str = "/home/s3665534/adversarial-training-box-FORK"):
        self.workspace_path = Path(workspace_path)
        self.logs_dir = self.workspace_path / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
    def create_slurm_script(self, config: SlurmJobConfig, script_args: str = "") -> str:
        """Generate a SLURM script from configuration"""
        # TODO: fixed to a100 now
        additional_params_str = ""
        if config.additional_params:
            for key, value in config.additional_params.items():
                additional_params_str += f"#SBATCH --{key}={value}\n"
        slurm_script = f"""#!/bin/bash
#SBATCH --job-name={config.job_name}
#SBATCH --mail-user="{config.mail_user}"
#SBATCH --mail-type=ALL
#SBATCH --time={config.time_limit}
#SBATCH --partition={config.partition}
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={config.cpus_per_task}
#SBATCH --mem={config.memory}
#SBATCH --gres=gpu:{config.gpus}
{additional_params_str}
# Load required modules
module load ALICE/default
module load CUDA/12.3.2
module load GCC/11.3.0
module load Miniconda3/24.7.1-0

# Set environment variables
export CONDA_ENVS_PATH=/home/s3665534/.conda/envs
export CONDA_PREFIX=/home/s3665534/.conda
export CUDA_HOME=/easybuild/software/CUDA/12.3.2
export PATH=$CUDA_HOME/bin:$PATH

# Navigate to workspace
cd {self.workspace_path}
echo "## Current directory $(pwd)"

echo "## GPU Information:"
nvidia-smi

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate adversarial-training-box

# Run the script
python {config.script_path} {script_args}

echo "## Job finished. Goodbye"
"""
        return slurm_script
    
    def submit_job(self, config: SlurmJobConfig, script_args: str = "", dry_run: bool = False) -> Optional[str]:
        """Submit a job to SLURM"""
        # Create temporary SLURM script
        temp_script_path = self.workspace_path / f"temp_{config.job_name}.slurm"
        
        try:
            # Write SLURM script
            slurm_content = self.create_slurm_script(config, script_args)
            with open(temp_script_path, 'w') as f:
                f.write(slurm_content)
            
            if dry_run:
                print(f"DRY RUN - Would submit job '{config.job_name}' with script:")
                print(slurm_content)
                return None
            
            # Submit job
            result = subprocess.run(['sbatch', str(temp_script_path)], 
                                  capture_output=True, text=True, cwd=self.workspace_path)
            
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                print(f"✓ Submitted job '{config.job_name}' with ID: {job_id}")
                return job_id
            else:
                print(f"✗ Failed to submit job '{config.job_name}': {result.stderr}")
                return None
                
        finally:
            # Clean up temporary script
            if temp_script_path.exists():
                temp_script_path.unlink()
    
    def submit_multiple_jobs(self, job_configs: List[tuple], delay_seconds: int = 5, dry_run: bool = False) -> List[str]:
        """Submit multiple jobs with optional delay between submissions"""
        job_ids = []
        
        for i, (config, script_args) in enumerate(job_configs):
            print(f"\n[{i+1}/{len(job_configs)}] Submitting job: {config.job_name}")
            
            job_id = self.submit_job(config, script_args, dry_run)
            if job_id:
                job_ids.append(job_id)
            
            # Add delay between submissions (except for last job)
            if i < len(job_configs) - 1 and delay_seconds > 0 and not dry_run:
                print(f"Waiting {delay_seconds} seconds before next submission...")
                time.sleep(delay_seconds)
        
        return job_ids
    
    def check_job_status(self, job_id: str) -> Dict[str, str]:
        """Check the status of a submitted job"""
        result = subprocess.run(['squeue', '-j', job_id, '--format=%T,%R'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0 and len(result.stdout.strip().split('\n')) > 1:
            status_line = result.stdout.strip().split('\n')[1]
            status, reason = status_line.split(',')
            return {'status': status, 'reason': reason}
        else:
            return {'status': 'NOT_FOUND', 'reason': 'Job not in queue'}


def main():
    submitter = SlurmJobSubmitter()
    networks = ["MNIST_RELU_4_256", "MNIST_RELU_4_1024"] #"MNIST_RELU_6_200", "MNIST_RELU_5_256", "MNIST_RELU_6_256", 
    job_configs = []
    
    for network in networks:
        config = SlurmJobConfig(
            job_name=f"emnist_adversarial_training_{network.lower()}",
            script_path="example_scripts/emnist-pgd-training.py",
            partition="gpu-l4-24g", # gpu-short;gpu-2080ti-11g; gpu-mig-40g; gpu-a100-80g; gpu-l4-24g
            time_limit="2-04:00:00"
        )
        args = f"--network {network} --experiment_name {network.lower()}-pgd-training"
        job_configs.append((config, args))
    
    job_ids = submitter.submit_multiple_jobs(job_configs, delay_seconds=1, dry_run=False)
    print(f"\nSubmitted {len(job_ids)} jobs")

if __name__ == "__main__":
    main()