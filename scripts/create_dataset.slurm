#!/bin/bash
#SBATCH --job-name=create_dataset
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH -c 48

python -m notify.me info "starting dataset generation $HOSTNAME"
conda init bash
conda activate base
singularity exec --overlay /scratch/work/public/singularity/mujoco200-dep-cuda11.1-cudnn8-ubunutu18.04.sqf:ro /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif python create_da_dataset_push_box.py
python -m notify.me info "done generating dataset"