#!/bin/bash
#SBATCH --job-name=run_exp
#SBATCH --open-mode=append
#SBATCH --output=./output/%j_%x.out
#SBATCH --error=./output/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=5:00:00
#SBATCH --mem=16G
#SBATCH -c 8
#SBATCH --gres=gpu:1

singularity exec --nv --overlay /scratch/bne215/overlay-50G-10M.ext3:ro \
    --overlay /scratch/work/public/singularity/mujoco200-dep-cuda11.1-cudnn8-ubunutu18.04.sqf:ro \
    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "
source /ext3/env.sh
conda activate
python run_exp.py -c da_configs/greene/slidepuck/continuous/slidepuck_continuous_0.yaml
"
