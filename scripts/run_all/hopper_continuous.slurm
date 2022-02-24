#!/bin/bash
#SBATCH --job-name=run_exp_seeded
#SBATCH --open-mode=append
#SBATCH --output=./output/%j_%x_%a.out
#SBATCH --error=./output/%j_%x_%a.err
#SBATCH --export=ALL
#SBATCH --time=5:00:00
#SBATCH --mem=16G
#SBATCH -c 8
#SBATCH --gres=gpu:1

singularity exec --nv --overlay /scratch/bne215/overlay-50G-10M.ext3:ro     --overlay /scratch/work/public/singularity/mujoco200-dep-cuda11.1-cudnn8-ubunutu18.04.sqf:ro     /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "
source /ext3/env.sh
conda activate
echo 'task id $SLURM_ARRAY_TASK_ID'
python run_exp.py -c da_configs/greene/hopper/continuous/hopper_continuous_$SLURM_ARRAY_TASK_ID.yaml
"
    