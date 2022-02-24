#!/bin/bash
#SBATCH --job-name=run_exp_cpu
#SBATCH --open-mode=append
#SBATCH --output=./output/%j_%x.out
#SBATCH --error=./output/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH -c 32

python -m notify.me info "starting training on $HOSTNAME"

singularity exec --nv --overlay /scratch/bne215/overlay-50G-10M.ext3:ro \
    --overlay /scratch/work/public/singularity/mujoco200-dep-cuda11.1-cudnn8-ubunutu18.04.sqf:ro \
    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif -c "
    
    "

python run_exp.py -c da_configs/proto_unsup.yaml --device cpu
python -m notify.me info "training on $HOSTNAME done"
